//! Context packs: reuse the work done on a text when it turns up again inside a different prompt.
//!
//! A prefix cache only helps when the shared text comes first. Retrieval and agent prompts put the
//! same documents in changing order behind changing instructions, so the cache never hits. A *pack*
//! is the cache of one text prefilled alone, kept on a [`Shelf`]. [`compose`] puts several packs
//! one after another in a fresh state:
//!
//! 1. each pack's rows are copied in and their keys rotated to the new positions (exact, see
//!    [`Model::splice_cache`]);
//! 2. what is still wrong is that a row computed alone never saw the packs before it. The damage is
//!    worst at the start of each pack (where the text leans most on what came before) and fades.
//!    So the start of every later pack is recomputed in place, window by window, and the
//!    *measured* change of the rows is the stopping signal: when a window moves its rows by less
//!    than `tolerance` the rest of that pack is left as it is. A pack that does not depend on its
//!    neighbours is repaired by one small window; one that does is repaired as far as it needs;
//! 3. the question (the tail) is always computed fresh, attending to the repaired rows.
//!
//! `tolerance = 0` recomputes everything and gives exactly what a full prefill gives. Models whose
//! cache cannot be edited (sliding window, evicted, accelerator) are refused with an error and the
//! caller falls back to an ordinary prefill.

use crate::model::{KvKind, Model, ModelError, State};
use crate::snapshot::{SnapOptions, SnapshotError, SnapshotStore};

/// How [`compose`] repairs the joins.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ComposeOptions {
    /// Rows recomputed at the start of each later pack before the first measurement, and the size
    /// the window doubles from.
    pub probe: usize,
    /// Stop repairing a pack once the last measured rows changed by less than this (mean relative
    /// change of key and value over layers). `0.0` repairs everything.
    pub tolerance: f32,
    /// Most rows recomputed in one pack (`usize::MAX` for no limit).
    pub max_refresh: usize,
}
impl Default for ComposeOptions {
    fn default() -> Self {
        ComposeOptions {
            probe: 8,
            tolerance: 0.02,
            max_refresh: usize::MAX,
        }
    }
}

/// What [`compose`] did.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ComposeReport {
    /// Rows copied from packs.
    pub spliced: usize,
    /// Rows of packs recomputed to repair the joins.
    pub refreshed: usize,
    /// Rows computed for the tail.
    pub fresh: usize,
    /// Per later pack: rows recomputed and the change measured at the end of the last window.
    pub joins: Vec<(usize, f32)>,
}
impl ComposeReport {
    /// Rows that did not have to go through the whole network: `spliced - refreshed`.
    pub fn saved(&self) -> usize {
        self.spliced.saturating_sub(self.refreshed)
    }
}

/// Mean of a slice, `0` if empty.
fn mean(v: &[f32]) -> f32 {
    if v.is_empty() {
        0.0
    } else {
        v.iter().sum::<f32>() / v.len() as f32
    }
}

/// Repair the rows `lo..hi` of `st` (all moved in from elsewhere), front to back, in growing windows.
/// Returns rows recomputed and the change measured at the end of the last window.
fn repair(
    model: &Model,
    st: &mut State,
    lo: usize,
    hi: usize,
    o: &ComposeOptions,
) -> Result<(usize, f32), ModelError> {
    let cap = o.max_refresh.min(hi - lo);
    let (mut done, mut win, mut last) = (0usize, o.probe.max(1), f32::INFINITY);
    while done < cap {
        let n = win.min(cap - done);
        let d = model.refresh_slots(st, lo + done, n)?;
        done += n;
        // the end of a window is the part furthest from the join, the best sign of whether the damage is over
        last = mean(&d[d.len().saturating_sub(n.min(4))..]);
        if last <= o.tolerance {
            break;
        }
        win = win.saturating_mul(2);
    }
    Ok((done, if last.is_finite() { last } else { 0.0 }))
}

/// Build the state for `packs` followed by `tail` and return it with the logits for the next token.
/// `packs` are states made from position 0 by this model (see [`Shelf::pack`]) and must be exact;
/// the first is copied whole, the others are spliced and repaired. `tail` must not be empty.
pub fn compose(
    model: &Model,
    packs: &[&State],
    tail: &[usize],
    opts: &ComposeOptions,
) -> Result<(State, Vec<f32>, ComposeReport), ModelError> {
    if tail.is_empty() {
        return Err(ModelError::NoTokens);
    }
    let mut rep = ComposeReport::default();
    let mut st = match packs.first() {
        Some(p) if p.is_exact() => p.fork()?,
        Some(_) => return Err(ModelError::Shape("a pack must be exact".into())),
        None => model.new_state_with(KvKind::F32),
    };
    rep.spliced = st.tokens().len();
    for p in packs.iter().skip(1) {
        let (lo, n) = (st.tokens().len(), p.tokens().len());
        if n == 0 {
            continue;
        }
        model.splice_cache(&mut st, p, 0..n)?;
        rep.spliced += n;
        let (r, d) = repair(model, &mut st, lo, lo + n, opts)?;
        rep.refreshed += r;
        rep.joins.push((r, d));
    }
    let mut all = st.tokens().to_vec();
    all.extend_from_slice(tail);
    let logits = model.prefill_cached(&mut st, &all)?;
    rep.fresh = all.len() - rep.spliced;
    Ok((st, logits, rep))
}

/// Like [`compose`], but the packs were written by **another model** (`src`) and are carried over to
/// `model` through `map` ([`crate::kvmap`]). Every pack, the first included, comes in as a guess and
/// is repaired with the same adaptive rule: recompute from its start until the measured drift falls
/// under `opts.tolerance`. With `tolerance: 0.0` everything is recomputed, the answer equals a plain
/// prefill by `model`, and nothing is saved; how much a real map saves depends on how well the two
/// models' caches relate, which only `corbul map-check` can tell on real models. `joins` has one entry
/// per pack. The packs must be exact states of `src`.
pub fn compose_mapped(
    map: &crate::kvmap::KvMap,
    src: &Model,
    packs: &[&State],
    model: &Model,
    tail: &[usize],
    opts: &ComposeOptions,
) -> Result<(State, Vec<f32>, ComposeReport), crate::kvmap::KvMapError> {
    if tail.is_empty() {
        return Err(ModelError::NoTokens.into());
    }
    let mut rep = ComposeReport::default();
    let mut st = model.new_state_with(KvKind::F32);
    for p in packs {
        let (lo, n) = (st.tokens().len(), p.tokens().len());
        if n == 0 {
            continue;
        }
        map.import(src, p, model, &mut st)?;
        rep.spliced += n;
        let (r, d) = repair(model, &mut st, lo, lo + n, opts)?;
        rep.refreshed += r;
        rep.joins.push((r, d));
    }
    let mut all = st.tokens().to_vec();
    all.extend_from_slice(tail);
    let logits = model.prefill_cached(&mut st, &all)?;
    rep.fresh = all.len() - rep.spliced;
    Ok((st, logits, rep))
}

/// How [`shrink_pack`] shortens a pack.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShrinkOptions {
    /// The last `window` tokens of the pack are re-read to vote for the rows to keep (and, with
    /// `refit`, become the probes the kept values are fitted to).
    pub window: usize,
    /// Smoothing of the votes over neighbouring rows (SnapKV).
    pub pool: usize,
    /// The first rows are always kept: they are where attention piles up.
    pub sinks: usize,
    /// Refit the kept values (see `kvfit.rs`).
    pub refit: bool,
    /// How strongly the refit stays near the old values.
    pub ridge: f32,
}
impl Default for ShrinkOptions {
    fn default() -> Self {
        ShrinkOptions {
            window: 16,
            pool: 5,
            sinks: 4,
            refit: true,
            ridge: 0.01,
        }
    }
}

/// SnapKV that always keeps the first `sinks` rows.
struct Sinked {
    inner: crate::kvpolicy::SnapKv,
    sinks: usize,
}
impl crate::kvpolicy::Policy for Sinked {
    fn select_keep(&self, scores: &[f32], len: usize, budget: usize) -> Vec<usize> {
        let s = self.sinks.min(budget / 2).min(len);
        let mut k = self
            .inner
            .select_keep(scores, len, budget.saturating_sub(s));
        k.extend(0..s);
        k.sort_unstable();
        k.dedup();
        k
    }
    fn needs_scores(&self) -> bool {
        true
    }
    fn name(&self) -> &'static str {
        "snapkv+sinks"
    }
}

/// A shorter version of `pack`: about `ratio` of its rows are kept, chosen by what its own last
/// `window` tokens attend to, optionally with the kept values refitted, and the keys rotated to
/// their new slots ([`Model::squeeze_cache`]). The result is the cache of the pack's text *with the
/// dropped tokens left out* (its `tokens()` are the kept ones), so it can be spliced and repaired
/// like any pack. It is an approximation: the kept rows were computed with the dropped tokens in
/// view. `ratio >= 1`, or a pack too short to vote, returns a copy.
///
/// Costs a re-read of `window` tokens, plus the fit when `refit` is on. Needs a cache that
/// [`Model::splice_cache`] accepts (no sliding window, no accelerator, not Q8).
pub fn shrink_pack(
    model: &Model,
    pack: &State,
    ratio: f32,
    o: &ShrinkOptions,
) -> Result<State, ModelError> {
    let n = pack.tokens().len();
    let w = o.window.min(n.saturating_sub(1));
    let budget = ((n as f64) * ratio.clamp(0.0, 1.0) as f64).ceil() as usize;
    let budget = budget.max(w + o.sinks + 1);
    if !pack.is_exact() {
        return Err(ModelError::Shape("a pack must be exact".into()));
    }
    if ratio.is_nan() || budget >= n || w == 0 {
        return pack.fork();
    }
    let tokens = pack.tokens().to_vec();
    let mut st = pack.fork()?;
    st.truncate(n - w)?;
    st.track_scores(true);
    if o.refit {
        st.record_queries(w);
    }
    model.forward_batch(&mut st, &tokens[n - w..])?;
    let policy = Sinked {
        inner: crate::kvpolicy::SnapKv {
            window: w,
            pool: o.pool,
        },
        sinks: o.sinks,
    };
    let layers = crate::kvpolicy::LayerBudgets::Uniform;
    if o.refit {
        st.apply_policy_fitted(&policy, budget, &layers, o.ridge)?;
    } else {
        st.apply_policy(&policy, budget, &layers)?;
    }
    st.track_scores(false);
    st.record_queries(0);
    model.squeeze_cache(&mut st)?;
    Ok(st)
}

/// Packs kept on disk (an LRU-capped [`SnapshotStore`], optionally signed), made on demand.
///
/// Packs are F32 by default. [`Shelf::with_kind`] keeps them in F16 instead, half the size on disk
/// and in memory, at the price of the rounding F16 brings to every row (what a model run with an
/// F16 cache has anyway); [`Shelf::with_compression`] compresses the files without any loss. A pack
/// stored in the other precision is not used: it is made again and replaces the old one.
pub struct Shelf {
    store: SnapshotStore,
    kind: KvKind,
    compress: bool,
}
impl Shelf {
    pub fn new(store: SnapshotStore) -> Shelf {
        Shelf {
            store,
            kind: KvKind::F32,
            compress: false,
        }
    }
    /// Cache precision of the packs: `F32` or `F16` (`Q8` cannot be spliced and is refused).
    pub fn with_kind(mut self, kind: KvKind) -> Result<Shelf, SnapshotError> {
        if kind == KvKind::Q8 {
            return Err(SnapshotError::Unsupported(
                "packs cannot be kept in Q8: splicing needs F32 or F16".into(),
            ));
        }
        self.kind = kind;
        Ok(self)
    }
    /// Byte-shuffle and compress the pack files (lossless; costs time on save and load).
    pub fn with_compression(mut self, on: bool) -> Shelf {
        self.compress = on;
        self
    }
    pub fn kind(&self) -> KvKind {
        self.kind
    }
    pub fn store(&self) -> &SnapshotStore {
        &self.store
    }
    /// The pack for `tokens`: loaded if shelved, else prefilled from empty and shelved. The bool is
    /// true when it was already there.
    pub fn pack(&self, model: &Model, tokens: &[usize]) -> Result<(State, bool), SnapshotError> {
        if tokens.is_empty() {
            return Err(SnapshotError::Unsupported("a pack needs tokens".into()));
        }
        if let Some(s) = self.store.get(model, tokens)?
            && s.kv_kind() == self.kind
        {
            return Ok((s, true));
        }
        let mut st = model.new_state_with(self.kind);
        model.prefill_cached(&mut st, tokens)?;
        self.store.put(
            model,
            &st,
            SnapOptions {
                compress: self.compress,
            },
        )?;
        Ok((st, false))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::tests::{reference_logits, tiny};

    fn toks(seed: usize, n: usize, vocab: usize) -> Vec<usize> {
        (0..n)
            .map(|i| (seed * 31 + i * 7 + i * i) % vocab)
            .collect()
    }
    fn close(a: &[f32], b: &[f32], tol: f32) -> bool {
        a.len() == b.len()
            && a.iter()
                .zip(b)
                .all(|(x, y)| (x - y).abs() <= tol * (1.0 + y.abs()))
    }
    fn packed(m: &Model, t: &[usize]) -> State {
        let mut s = m.new_state();
        m.prefill_cached(&mut s, t).unwrap();
        s
    }

    #[test]
    fn full_repair_equals_full_prefill() {
        let m = tiny(4, 2, false);
        let v = m.spec.vocab as usize;
        let (a, b, c, q) = (toks(1, 9, v), toks(2, 13, v), toks(3, 6, v), toks(4, 3, v));
        let (pa, pb, pc) = (packed(&m, &a), packed(&m, &b), packed(&m, &c));
        let o = ComposeOptions {
            tolerance: 0.0,
            ..Default::default()
        };
        let (st, logits, rep) = compose(&m, &[&pa, &pb, &pc], &q, &o).unwrap();
        let all: Vec<usize> = [a, b, c, q].concat();
        assert!(close(&logits, &reference_logits(&m, &all), 2e-3));
        assert_eq!(rep.refreshed, 13 + 6);
        assert!(st.is_exact());
        assert_eq!(st.tokens(), &all[..]);
    }

    #[test]
    fn no_repair_is_cheap_but_approximate_and_flagged() {
        let m = tiny(4, 2, false);
        let v = m.spec.vocab as usize;
        let (a, b, q) = (toks(5, 10, v), toks(6, 12, v), toks(7, 2, v));
        let (pa, pb) = (packed(&m, &a), packed(&m, &b));
        let o = ComposeOptions {
            probe: 1,
            tolerance: f32::INFINITY,
            max_refresh: 0,
        };
        let (st, logits, rep) = compose(&m, &[&pa, &pb], &q, &o).unwrap();
        assert_eq!((rep.refreshed, rep.saved()), (0, 22));
        assert!(!st.is_exact() && st.approx_slots() == 12);
        let all: Vec<usize> = [a, b, q].concat();
        assert!(logits.iter().all(|x| x.is_finite()));
        // unrepaired differs from the truth (otherwise the repair would be pointless in this test)
        assert!(!close(&logits, &reference_logits(&m, &all), 1e-6));
    }

    #[test]
    fn first_pack_and_independent_prefix_are_exact_without_repair() {
        let m = tiny(2, 2, true);
        let v = m.spec.vocab as usize;
        let (a, q) = (toks(8, 11, v), toks(9, 4, v));
        let pa = packed(&m, &a);
        let (st, logits, rep) = compose(&m, &[&pa], &q, &ComposeOptions::default()).unwrap();
        assert!(st.is_exact() && rep.refreshed == 0 && rep.joins.is_empty());
        let all: Vec<usize> = [a, q].concat();
        assert!(close(&logits, &reference_logits(&m, &all), 2e-3));
    }

    #[test]
    fn adaptive_window_grows_until_drift_falls() {
        let m = tiny(4, 2, false);
        let v = m.spec.vocab as usize;
        let (a, b, q) = (toks(10, 6, v), toks(11, 22, v), toks(12, 2, v));
        let (pa, pb) = (packed(&m, &a), packed(&m, &b));
        let tight = ComposeOptions {
            probe: 2,
            tolerance: 1e-9,
            ..Default::default()
        };
        let (_, _, r) = compose(&m, &[&pa, &pb], &q, &tight).unwrap();
        assert_eq!(r.joins.len(), 1);
        assert!(r.refreshed >= 2 && r.refreshed <= 22);
        let loose = ComposeOptions {
            probe: 2,
            tolerance: 1e9,
            ..Default::default()
        };
        let (_, _, r2) = compose(&m, &[&pa, &pb], &q, &loose).unwrap();
        assert_eq!(r2.refreshed, 2);
    }

    #[test]
    fn rotation_is_exact_keys_land_where_a_prefill_puts_them() {
        let m = tiny(2, 2, false);
        let v = m.spec.vocab as usize;
        let (a, b) = (toks(13, 5, v), toks(14, 7, v));
        let (pa, pb) = (packed(&m, &a), packed(&m, &b));
        let mut st = pa.fork().unwrap();
        m.splice_cache(&mut st, &pb, 0..7).unwrap();
        // layer 0 keys depend only on the token and its position (no earlier context): exact after rotation
        let direct = packed(&m, &[a.clone(), b.clone()].concat());
        for i in 0..7 {
            let (x, y) = (
                st.key_row(0, 5 + i).unwrap(),
                direct.key_row(0, 5 + i).unwrap(),
            );
            assert!(close(&x, &y, 1e-4), "slot {i}");
        }
    }

    #[test]
    fn refresh_of_exact_rows_changes_nothing() {
        let m = tiny(4, 2, false);
        let v = m.spec.vocab as usize;
        let t = toks(15, 16, v);
        let mut st = packed(&m, &t);
        let d = m.refresh_slots(&mut st, 3, 9).unwrap();
        assert!(d.iter().all(|x| *x < 1e-4), "{d:?}");
        assert!(st.is_exact());
    }

    #[test]
    fn hostile_inputs_are_refused_without_damage() {
        let m = tiny(2, 2, false);
        let v = m.spec.vocab as usize;
        let t = toks(16, 6, v);
        let p = packed(&m, &t);
        let mut st = p.fork().unwrap();
        assert!(m.splice_cache(&mut st, &p, 3..3).is_err());
        assert!(m.splice_cache(&mut st, &p, 0..99).is_err());
        assert!(m.refresh_slots(&mut st, 4, 5).is_err());
        assert!(m.refresh_slots(&mut st, 0, 0).is_err());
        assert_eq!(st.tokens(), p.tokens());
        let f16 = m.new_state_with(KvKind::F16);
        let mut d = m.new_state();
        assert!(m.splice_cache(&mut d, &f16, 0..1).is_err());
        assert!(compose(&m, &[&p], &[], &ComposeOptions::default()).is_err());
        // an approximate state is never written to disk
        let mut ap = p.fork().unwrap();
        m.splice_cache(&mut ap, &p, 0..6).unwrap();
        assert!(!ap.is_exact());
        assert!(crate::snapshot::encode(&m, &ap, SnapOptions::default()).is_err());
        // and truncating back removes the doubt
        ap.truncate(6).unwrap();
        assert!(ap.is_exact());
    }

    #[test]
    fn shelf_makes_then_reuses_packs() {
        let m = tiny(2, 2, false);
        let v = m.spec.vocab as usize;
        let dir = std::env::temp_dir().join(format!("ctxpack-shelf-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let shelf = Shelf::new(SnapshotStore::open(&dir, 1 << 26).unwrap());
        let t = toks(17, 10, v);
        let (s1, hit1) = shelf.pack(&m, &t).unwrap();
        let (s2, hit2) = shelf.pack(&m, &t).unwrap();
        assert!(!hit1 && hit2);
        assert_eq!(s1.tokens(), s2.tokens());
        assert!(shelf.pack(&m, &[]).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }
    #[test]
    fn f16_and_compressed_shelves_work_and_never_mix_precisions() {
        let m = tiny(2, 2, false);
        let v = m.spec.vocab as usize;
        let dir = std::env::temp_dir().join(format!("ctxpack-f16-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let open = || SnapshotStore::open(&dir, 1 << 26).unwrap();
        assert!(Shelf::new(open()).with_kind(KvKind::Q8).is_err());
        let (a, b, q) = (toks(5, 9, v), toks(6, 7, v), toks(7, 3, v));
        let s16 = Shelf::new(open())
            .with_kind(KvKind::F16)
            .unwrap()
            .with_compression(true);
        let (pa, h1) = s16.pack(&m, &a).unwrap();
        let (pb, _) = s16.pack(&m, &b).unwrap();
        assert!(!h1 && pa.kv_kind() == KvKind::F16);
        assert!(
            s16.pack(&m, &a).unwrap().1,
            "second time it is on the shelf"
        );
        // composed from F16 packs with everything repaired = a full F16 prefill
        let o = ComposeOptions {
            tolerance: 0.0,
            ..Default::default()
        };
        let (_, logits, _) = compose(&m, &[&pa, &pb], &q, &o).unwrap();
        let mut full = m.new_state_with(KvKind::F16);
        let all: Vec<usize> = [a.clone(), b.clone(), q.clone()].concat();
        let want = m.prefill_cached(&mut full, &all).unwrap();
        assert!(close(&logits, &want, 2e-2));
        // an F32 shelf over the same folder does not take the F16 file: it makes its own
        let s32 = Shelf::new(open());
        let (p32, hit) = s32.pack(&m, &a).unwrap();
        assert!(!hit && p32.kv_kind() == KvKind::F32);
        // and back: the F16 shelf finds an F32 file, remakes it in F16
        let (back, hit) = s16.pack(&m, &a).unwrap();
        assert!(!hit && back.kv_kind() == KvKind::F16);
        let _ = std::fs::remove_dir_all(&dir);
    }
    #[test]
    fn a_shrunk_pack_is_the_cache_of_its_shortened_text() {
        let m = tiny(4, 2, false);
        let v = m.spec.vocab as usize;
        let (a, b, q) = (toks(21, 10, v), toks(22, 14, v), toks(23, 3, v));
        let (pa, pb) = (packed(&m, &a), packed(&m, &b));
        for refit in [false, true] {
            let o = ShrinkOptions {
                window: 4,
                sinks: 2,
                refit,
                ..Default::default()
            };
            let sb = shrink_pack(&m, &pb, 0.5, &o).unwrap();
            assert!(sb.is_exact() && !sb.is_compacted());
            assert_eq!(sb.pos, sb.tokens().len());
            assert!(sb.pos < 14 && sb.pos >= 7);
            assert_eq!(sb.kv_kind(), pb.kv_kind());
            // the kept tokens are a subsequence of the pack's, the first ones always, the last window too
            let mut it = b.iter();
            assert!(sb.tokens().iter().all(|t| it.any(|x| x == t)));
            assert_eq!(&sb.tokens()[..2], &b[..2]);
            assert_eq!(sb.tokens()[sb.pos - 4..], b[10..]);
            // layer-0 keys depend only on the token and its slot: the rotation put them where a
            // prefill of the shortened text puts them
            let direct = packed(&m, sb.tokens());
            for i in 0..sb.pos {
                let (x, y) = (sb.key_row(0, i).unwrap(), direct.key_row(0, i).unwrap());
                assert!(close(&x, &y, 1e-3), "refit {refit} slot {i}");
            }
            // composed with everything repaired = a full prefill of the shortened text
            let full = ComposeOptions {
                tolerance: 0.0,
                ..Default::default()
            };
            let (st, logits, rep) = compose(&m, &[&pa, &sb], &q, &full).unwrap();
            let all: Vec<usize> = [a.clone(), sb.tokens().to_vec(), q.clone()].concat();
            assert!(close(&logits, &reference_logits(&m, &all), 2e-3));
            assert!(st.is_exact() || st.approx_slots() == 0);
            assert_eq!(rep.spliced, 10 + sb.pos);
        }
        // nothing to shrink: a copy
        let o = ShrinkOptions::default();
        for r in [1.0, 2.0, f32::NAN] {
            let c = shrink_pack(&m, &pb, r, &o).unwrap();
            assert_eq!(c.tokens(), pb.tokens());
        }
        let tiny_pack = packed(&m, &toks(24, 3, v));
        assert_eq!(shrink_pack(&m, &tiny_pack, 0.1, &o).unwrap().pos, 3);
        // an approximate pack is refused
        let mut st = pa.fork().unwrap();
        m.splice_cache(&mut st, &pb, 0..14).unwrap();
        assert!(shrink_pack(&m, &st, 0.5, &o).is_err());
    }
}
