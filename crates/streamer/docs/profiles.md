# How concentrated is expert routing? Measured on the histories colibri ships

Generated with `cargo run --release -p expert-streamer --example profiles -- LABEL=FILE ...`
(runs in well under a second). Source files, taken 2026-11-16 from the colibri repository:
`c/profiles/qwen36-35b.coli_usage`, `c/profiles/qwen38-flash-next.coli_usage`, and the session
histories `docs/experiments/*/history.usage.gz`. Four of the five session files are
byte-identical, so there are only **four distinct histories** (two aggregate profiles, two
short sessions), and only the three Qwen3.8 ones can be scored against each other.

A history is counts only. It says which experts were used how often, never in which order.
Everything below is therefore popularity; nothing here speaks for recency, prefetching or
speculation.

## Concentration (best-case fixed set of the hottest experts, on the history itself)

| history | layers x experts | uses | never used | Gini | effective experts | top 1% | 5% | 10% | 25% | 50% | experts for 50% / 80% / 90% / 95% of uses |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| a | 40x256 | 796160 |   1.1% | 0.49 | 6726 (65.7%) |   6.0% |  21.2% |  34.2% |  59.3% |  83.5% | 18.5% / 45.3% / 60.5% / 71.5% |
| b | 48x512 | 1194240 |   6.1% | 0.60 | 12874 (52.4%) |   8.3% |  26.4% |  41.1% |  68.7% |  90.7% | 13.9% / 35.0% / 48.7% / 59.8% |
| c | 48x512 | 134880 |  54.1% | 0.86 | 4196 (17.1%) |  27.4% |  58.7% |  74.7% |  93.7% | 100.0% | 3.4% / 12.6% / 20.2% / 27.4% |
| d | 48x512 | 158400 |  48.3% | 0.80 | 6112 (24.9%) |  19.9% |  46.2% |  63.0% |  88.8% |  99.7% | 5.9% / 18.1% / 26.3% / 33.1% |

## Per layer: does a per-layer budget beat one global ranking?

A global ranking is optimal for plain coverage, so the useful number is what an EQUAL split per layer would lose.

| history | layer top-10% share (min / median / max) | global 10% | equal per layer 10% | global 25% | equal per layer 25% |
|---|---|---:|---:|---:|---:|
| a | 21.0% / 33.3% / 42.0% |  34.2% |  33.4% |  59.3% |  59.2% |
| b | 26.1% / 42.1% / 52.4% |  41.1% |  40.8% |  68.7% |  68.6% |
| c | 47.6% / 74.6% / 90.5% |  74.7% |  74.2% |  93.7% |  92.9% |
| d | 38.7% / 64.3% / 85.8% |  63.0% |  62.7% |  88.8% |  87.9% |

## Held out: rank on one history, score on another of the same model shape

| ranked on | scored on | top 1% | 5% | 10% | 25% | 50% |
|---|---|---:|---:|---:|---:|---:|
| b | c |   1.7% |   5.8% |  11.2% |  28.2% |  54.8% |
| b | d |   4.7% |  17.6% |  29.6% |  56.3% |  82.4% |
| c | b |   0.9% |   5.6% |  12.2% |  30.6% |  55.2% |
| c | d |   0.5% |   7.3% |  17.2% |  40.1% |  64.1% |
| d | b |   2.4% |  12.8% |  24.2% |  49.4% |  73.4% |
| d | c |   1.7% |   6.2% |  12.5% |  31.4% |  62.8% |

## Spread over the histories (own coverage, no hold-out)

| pinned share of all experts | min | median | max |
|---:|---:|---:|---:|
| 1% | 6.0% | 19.9% | 27.4% |
| 5% | 21.2% | 46.2% | 58.7% |
| 10% | 34.2% | 63.0% | 74.7% |
| 25% | 59.3% | 88.8% | 93.7% |
| 50% | 83.5% | 99.7% | 100.0% |

## Range from SKEW_SAMPLES (what other crates can use)

| pinned share of all experts | own history: min / median / max | held out on another history: min / median / max |
|---:|---|---|
| 5% | 21.2% / 46.2% / 58.7% | 5.6% / 9.5% / 17.6% |
| 10% | 34.2% / 63.0% / 74.7% | 11.2% / 18.4% / 29.6% |
| 25% | 59.3% / 88.8% / 93.7% | 28.2% / 40.4% / 56.3% |
| 50% | 83.5% / 99.7% / 100.0% | 54.8% / 68.1% / 82.4% |


## What the numbers say

1. **Routing is skewed, but how much depends on the workload, not on the model.** One short
   session leaves 48-54% of the experts unused and puts 63-75% of the uses on the top 10%.
   The aggregate profile of the same model (many prompts) puts 41% on its top 10%, and the
   Qwen3.6 profile 34%. A pinned set built from one narrow session is a bad guess for the next.
2. **Coverage measured on the history a set was built from overstates what it serves on new
   prompts by a factor of about 2 to 4** at 5-10% pinned. Ranked on one Qwen3.8 history and
   scored on another, the top 10% serves 11-30% of the uses, against 41-75% on its own history.
   Colibri's own documentation reports the same effect (70% on its own history, 57.7% on a
   separate session, at about 26% of experts); the 25% row above gives 28-56% for the same
   cross-scoring.
3. **A per-layer budget is not worth it.** Layers differ in how concentrated they are (the
   top-10% share of a layer ranges 21-42% on Qwen3.6 and 26-52% on Qwen3.8 aggregate), but an
   equal number of experts per layer loses between 0.1 and 0.9 points against one global
   ranking at 10% and 25% pinned. A global ranking is optimal for plain coverage by construction.
4. **There is no universal PIN fraction.** The spread between histories is wider than the
   effect of any single setting. Use the range, not a point: at 10% of all experts pinned,
   expect between 11% and 75% of the uses served, depending on how far the prompts resemble the
   history the set came from. For an estimator, take the held-out range as the conservative
   value and the own-history range as the optimistic one.
5. **Pin as much as memory allows, then measure.** Coverage never falls as the pinned share grows
   (a short session saturates early: 94% at 25%), so a smaller pin is never better for the hit
   rate; the limit is the swap threshold, not a magic fraction. The auto-tune search should start from the
   largest pin that does not push the machine into swap and search downwards, not from a fixed
   percentage.

### From a pinned share to a RAM tier

The share is `pinned experts / all experts`, i.e. `pin_gb / expert_gb` for one model. As an
illustration only: Qwen3.6-35B at int4 has 10,240 experts of about 1.7 MB (colibri's
`docs/tuning.md`), 17.4 GB of experts in all, so a 4 / 8 / 12 GB pin is 23% / 46% / 69% of them.
How much RAM a tier really leaves for a pin depends on the dense weights, the context and the
operating system; that is not measured here.

## What cannot be judged from this data

* **Stability over time.** Records carry no order, so a history cannot be split into halves to
  see whether popularity drifts inside a session. The cross-history scoring above is the only
  stability evidence, and it compares an aggregate with narrow sessions, which is not a
  same-distribution test.
* **Other models and engines.** Only Qwen3.6 and Qwen3.8 are shipped. Nothing is known here
  about DeepSeek, GLM, Kimi or others; colibri itself notes that some models (OLMoE) have no hot
  experts at all and show a nearly straight curve.
* **Who produced the histories.** They come from the colibri authors' machines and prompts;
  some sessions use MTP. Your prompts may be flatter or narrower.
* **Hit rate.** Coverage of a fixed set is a lower bound on a hit rate (a recency cache serves
  part of the rest) and ignores the cost of misses.
