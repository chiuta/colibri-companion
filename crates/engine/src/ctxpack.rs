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

/// Packs kept on disk (an LRU-capped [`SnapshotStore`], optionally signed), made on demand.
pub struct Shelf {
    store: SnapshotStore,
}
impl Shelf {
    pub fn new(store: SnapshotStore) -> Shelf {
        Shelf { store }
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
        if let Some(s) = self.store.get(model, tokens)? {
            return Ok((s, true));
        }
        let mut st = model.new_state_with(KvKind::F32);
        model.prefill_cached(&mut st, tokens)?;
        self.store.put(model, &st, SnapOptions::default())?;
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
}
