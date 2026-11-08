# Bot design: Short Deck PLO5 heads-up (team_max)

The bot follows the recipe of the strongest poker AIs (Libratus/Pluribus), scaled down to a 1-CPU, 8-seconds-per-1000-hands budget:

1. **Offline equilibrium blueprint.** JIT-compiled Monte Carlo CFR over the exact betting tree, with a potential-aware card abstraction.
2. **Real-time card reading** with a vectorised equity engine.
3. **Safe exploitation** in the style of Restricted Nash Response:
   - against balanced opponents the bot plays the blueprint;
   - it shifts to an exploitative policy only as far as the opponent measurably deviates from equilibrium behaviour.
   - Opponent ranges are calibrated from the hands they show down.

Everything is built from scratch on top of the kit's own evaluator, so hand rankings match the referee exactly.

## Files

| File | Role |
|---|---|
| `agent.py` (YOUR CODE section only) | Loads engine and strategy, wraps every call in try/except with a legal fallback, hard time cap. |
| `strategy.py` | Per-decision logic: opponent model → range estimate → equity → blueprint/exploit mix → action. All tunable numbers are in `PARAMS`. |
| `plo_engine.py` | Vectorised Omaha evaluator. `T[board-triple, hole-pair]` (7140 × 630) is built at startup (~1.5 s) from `evaluator.get_table()`, verified identical to `omaha_score`. Also computes per-sample equity, current made-hand strength and opponent strength keys. |
| `betting.py` | Exact replica of `game.py` betting rules (fuzz-tested on 20k hands: identical legal actions, raise sizes, payoffs). |
| `blueprint2.pkl` | **Shipped blueprint:** 31,100 infosets, 12M MCCFR iterations. |
| `params_tuned.json` | The tuned mixing thresholds (dev0/dev1), validated on two independent seed sets. |
| `preflop_equity.npy`, `preflop_range.npz` | Preflop equity vs random hands and vs top-X% ranges for all 376,992 hands (19,998 suit classes). |
| `nn_policy.py`, `nn_policy.npz` | Deep-RL policy for `BOT_MODE=nn` (numpy features + MLP; the `.npz` holds a 9-member population). This is the default mode of the submitted bot. See "Deep RL track" below. |
| `offline/` | Everything used to build and evaluate the bot (excluded from the Docker image). Details below. |

## 1. Blueprint v2: potential-aware abstraction + JIT MCCFR

- **Betting tree.** The real tree is small: 41,490 nodes and 1,016 public betting states. It is flattened once into arrays (`offline/tree.py`). No action abstraction is needed, because only 4 actions exist and there is a 4-raise cap.
- **Card abstraction (potential-aware).** Plain equity groups would lump a big draw together with a medium made hand of the same equity, but they need different play. So hands are grouped by equity × draw potential:

  | Street | Groups |
  |---|---|
  | Preflop | 40 equity quantiles (exact table) |
  | Flop | 10 equity bins × 3 *potential* bins (potential = equity − current made-hand strength) |
  | Turn | 10 × 3, same as flop |
  | River | 30 equity quantiles |

  Features for 400k deals come from the same estimator and sample count used at play time (`offline/build_deals2.py`), so measurement noise matches.
- **Infoset key.** Street, group, position, pot at street start, and this street's action sequence. Stacks reset to 80 every hand, so the pot at street start determines both stacks. This is an imperfect-recall abstraction, as in Pluribus.
- **Solver (`offline/mccfr_jit.py`).** External-sampling MCCFR with Linear-CFR discounting, JIT-compiled with numba (about 15× the pure-Python solver). Independent chains run in parallel and their average strategies are merged. v2 used 8 chains × 1.5M = 12M iterations and covers 99.95% of infosets.
- **Evidence.** v2 beats the first blueprint (16 groups preflop, 12 per later street, 840k iterations):
  - by **+0.43 bb/deal** over 16,000 duplicate deals (4 wins / 12 chops / 0 losses);
  - and the first release by **+0.55**.
- **Experiment that did *not* help.** A finer v3 (60 / 12×4 / 12×4 / 40 groups, 47k infosets) scored −0.18 vs v2 over 16,000 deals. Groups narrower than the ±0.03 noise of play-time equity estimates (about 250 samples) only misclassify hands, so finer abstraction, whether solved on CPU or GPU, has hit diminishing returns.

## 2. Real-time equity

- **One Monte Carlo batch per (hand, street), cached.** It stores per-sample results plus the opponent's preflop strength and current made-hand strength. Later queries on the same street (for example after a re-raise changes the range estimate) are answered from the cache in microseconds.
- **Close-call refinement.** When a big-pot call is within 1.5 standard errors of the pot-odds line and we are ahead of time pace, a second batch is drawn.
- **Self-calibrating time manager.** It measures the cost per sample and the fixed overhead on the actual machine.

  | Setting | Value |
  |---|---|
  | Compute target | 2.5 s per half |
  | Panic mode | 3.2 s |
  | Hard stop (cheap fallback) | 4.0 s |

  The judge also charges HTTP overhead, measured at about +0.5 s on an idle machine.

## 3. Opponent model, showdown calibration, safe exploitation

- **Frequencies.** Preflop and postflop aggression, and fold-to-our-bets, each Bayesian-smoothed.
- **Showdown calibration.**
  - Whenever the opponent raised on a street and the hand reached showdown, we record the percentile of their actual hand on that street's board, against random hands.
  - Under a "top-k" range model, a mean percentile m implies the kept fraction k = 2(1 − m).
  - This is blended with the raise-frequency estimate through `sd_prior` pseudo-counts.
  - Measured means: nit 0.87, tag 0.79, equilibrium 0.67, barrel 0.57, maniac 0.50. Value-only bettors get a tight range; maniacs get none.
- **Range estimate.** When the opponent raises, sampled opponent hands are filtered to their calibrated top range. Each extra raise narrows it further.
- **Exploitative policy.** Range-adjusted equity vs pot odds, with value, semi-bluff and bluff thresholds that adapt to the opponent's fold rate.
- **Mixing (RNR-style).** The exploit share is `conf × clip((dev − dev0)/(dev1 − dev0), 0, 1)`:
  - `dev` is the opponent's distance from **equilibrium behaviour**, measured in typical-spread units over 5 statistics.
  - The reference is how our own blueprint looks through this same model: preflop raise 0.49, postflop aggression 0.415, fold-to-bet 0.40, showdown strength 0.73 / 0.665.
  - Typical measured deviations: balanced bots ≈ 0.7, heuristic bots ≈ 3.7, weak bots 5.5–7.
  - An earlier version compared against hand-picked norms. It mistook a balanced bot for an exploitable one and lost to it (−0.43 bb/deal). Measuring from equilibrium fixed this (+0.36).

## 4. Tuning (`offline/tune.py`, `offline/grid.py`)

- **Objective = expected tournament points** over a 10-bot practice pool, P(win) + 0.5·P(chop) per opponent:
  - P(win) per opponent is computed from the pooled mean and the per-match standard error of a 1000-deal match.
  - The pool includes strong opponents: the old blueprint and a frozen copy of the first release (`offline/release/`).
  - Common random numbers throughout: same cards, stable bot seeds.
- **(1+λ) evolution strategy:** 10 generations × 14 candidates over 25 parameters, with a winner's-curse guard (the best candidates must also win on fresh seeds).
  - **No candidate survived the fresh-seed check.** The hand-set thresholds are already near a local optimum at the resolution of 8,000 deals.
- **Targeted grid on the mixing thresholds** (8 seeds, then a confirmation on 12 fresh seeds).
  - Raising (dev0, dev1) from (1.5, 4) to **(2.5, 6)** keeps the blueprint longer against semi-competent bots.
  - Effect: +0.2 to +0.3 bb/deal against the strong opponents, a small cost against weak bots, and +0.1 expected points on both seed sets.
  - Shipped in `params_tuned.json`.

## Compute

- Laptop (i5, 4 cores / 8 threads): engine, tables, v2 training, evaluation.
- `swirl-beast-1` (i9-14900K, 30 threads): tuning, grids and long training runs. About 48 duplicate matches per 12 s.
- The RTX 5090 was freed but not needed. Tabular MCCFR is a serial, branchy tree walk that suits CPUs, and the v3 experiment shows a bigger abstraction (the thing a GPU solver would buy) no longer helps.

## Results (Nov 7, 2026, shipped configuration)

**Official HTTP judge vs the four baselines, seeds 101/202/303: 12/12 WIN, 0 errors.**

| Opponent | bb/deal (3 seeds) | First release (same seeds) |
|---|---|---|
| baseline_random | +7.2 / +7.6 / +7.9 | +6.3 / +7.2 / +7.2 |
| baseline_always_half | +12.0 / +13.3 / +12.5 | +11.8 / +13.4 / +12.2 |
| baseline_always_pot | +5.7 / +8.5 / +7.2 | +4.9 / +8.0 / +7.0 |
| baseline_calling_station | +8.5 / +7.8 / +8.3 | +8.8 / +8.2 / +8.7 |

**Practice pool, 12 fresh seeds × 1000 duplicate deals (mean bb/deal):**

| Opponent | Result |
|---|---|
| random | +7.33 |
| calling station | +8.42 |
| nit | +1.10 |
| tag | +2.73 |
| maniac | +13.41 |
| barrel | +3.95 |
| old blueprint (bp1) | **+0.42** |
| **first release** | **+0.28** |

On an 8-seed set: always half +12.21, always pot +6.81.

**Blueprint head-to-heads (pure blueprint vs pure blueprint):**

| Matchup | bb/deal | Deals |
|---|---|---|
| v2 vs v1 | +0.43 | 16,000 |
| v3 (finer) vs v2 | −0.18 | 16,000 |
| v2 at 84M iterations vs v2 at 12M | +0.01 | 32,000 |

The v2 blueprint is converged, and its granularity matches play-time measurement noise.

**Time bank.**
- Judge-charged time on an idle machine: 3.47 s per half.
- With two matches sharing a 4-core laptop: 2.9–6.7 s.
- Two guards fall back to zero-compute play:
  - our compute over 4.0 s;
  - estimated judge charge over 6.8 s (estimated from process CPU time, calibrated within 0.01 s of the judge).

**Docker (`tools/check_build.sh`): ALL CHECKS PASSED.** Startup 2.8 s, 1–12 ms per action, +1 MB image. The image contains only the shipped blueprint and the tuned parameters.

## Real competitors (the decisive test)

**Setup.**
- 10 public competitor bots for this exact game: forks of the official bundle, plus a GitHub search on Nov 7, 15:00 CST.
- Each one built and run with the tournament's Docker restrictions (`offline/rivals/`).
- None of them uses CFR. The strongest are Monte Carlo equity + opponent-model bots.

**First result.** The real field is much stronger than our practice pool.
- One rival is weak (+6.3 / +8.4 bb/deal).
- Against the other 9, the first v2 build won by only +0.1 to +1.0 bb/deal: mostly CHOPs, no losses.

**Variants, 9 rivals × 12 seeds = 108 tournament-faithful matches each:**

| Variant | Points (W=1, C=0.5) | W / C / L | Mean bb/deal |
|---|---|---|---|
| pure exploitative heuristic, 1500-sample cap | 78.0 | 51 / 54 / **3** | +0.68 |
| **mix (blueprint + exploitation), 1500-sample cap: shipped** | **77.5** | 47 / 61 / **0** | +0.60 |
| mix, 400-sample cap (the previous commit; 36 matches) | 69% of points | 14 / 22 / 0 | +0.55 |
| pure blueprint, 1500 (36 matches) | 69% of points | 14 / 22 / 0 | +0.53 |
| fixed 70% exploit, 1500 (72 matches) | 71% of points | 31 / 40 / 1 | +0.51 |

**Findings.**

1. **Equity precision matters against good bots.**
   - On fast CPUs the old 400-sample cap left most of the time bank unused (0.5 s of 8 s).
   - With a 1500 cap, the time manager still enforces the 2.5 s budget on slow machines.
2. **The best fixed policy differs by opponent, and our statistics cannot tell which.**
   - The heuristic crushes passive callers: shaswat +1.93, snr +2.11.
   - It loses to the most balanced adaptive bot: burnerdrew −0.44, 3 losses.
   - The blueprint is best against "trappy" bots (junehwang, shriharish) whose aggregate frequencies look like the passive callers'.
   - A rule fitted to 9 bots would overfit.
   - Switching strategies online within a 1000-hand half is too noisy: about 1.2σ for a 0.8 bb/hand difference.
3. **We ship the mix.** It ties on points and is the only variant that is positive against every rival, with 0 losses in 108 matches. The equilibrium blueprint is the insurance against adaptive bots.

### Range model v2 (shipped via `params_tuned.json`)

**Leak analysis.** We diffed the heuristic against the blueprint hand by hand on identical cards (`offline/rivals/`), and found that **defending against bets decides the matchups**:
- Against a value-only bettor (shaswat), the heuristic gained **+6.7k chips by folding**.
- Against an aggressive bettor (burnerdrew), it lost **−7.5k chips by calling and losing at showdown**.
- The old range model used one postflop aggression rate for all streets and sizes.

**v2 estimates the opponent's range from:**
- the raise rate on the street of their latest raise;
- a partial bet-size adjustment;
- per-street showdown calibration;
- earlier raises in the hand, each narrowing the range by the square root of its own keep fraction.

**Results vs 9 real rivals, same 8 seeds per variant (expected points from pooled mean/std):**

| Variant | E[pts] / 9 | W / C / L | Mean | Std |
|---|---|---|---|---|
| heuristic + range v2 | 6.62 | 33/39/0 | +0.76 | 8.97 |
| **mix + range v2 (shipped)** | **6.43** | 30/42/0 | +0.61 | 10.52 |
| mix, old ranges | 6.38 | 31/41/0 | +0.59 | 10.56 |
| heuristic, old ranges | 6.37 | 33/37/2 | +0.66 | 8.92 |

**Robustness check (16 seeds).** The pure heuristic + range v2 *loses* to balanced play:
- vs our CFR blueprint: −0.53 bb/deal (0/10/6);
- vs the old release: −0.18.

**Mix + range v2** is +0.04 vs the blueprint, +0.28 vs the release and +0.33 vs the heuristic, with 0 losses. None of the public rivals uses CFR, but the full field may include balanced bots, so the blueprint stays in as insurance.

## Research round 2 (4 research agents) and what came of it

**Winning the CI verdict.**
- P(win) ≈ Φ(μ·√1000/σ − 1.96). At μ = 0.6, σ = 11, cutting σ by 10% is worth about as much as +0.06 bb/deal of edge, so edge is the bigger lever.
- We now rank variants by expected points from pooled mean/std (`offline/rivals/summarize.py`). Raw W/C counts are too noisy (±0.14 per rival-variant).
- A mean-variance call penalty (`risk_lambda` = 0.0025) was implemented and tested against real rivals. It scored E[pts] 6.26 vs 6.48 and did not reduce variance, so it stays off.

**Equity precision.**
- *Shipped:* pair-decomposed river estimator (each opponent costs 10 lookups instead of 100) and river-stratified turn estimator. Both are unbiased (tested against brute force) and about 2.5× more efficient. Against real rivals: E[pts] 6.48 vs 6.43.
- *Not done:* a neural equity regressor. The literature shows about 3.4% MAE (Da Silva 2018), worse than our Monte Carlo.

**Real-time search.**
- Libratus-style safe river re-solving fits the time budget, but its published gains are smaller than our match noise.
- A gated river best-response search against the opponent model is the most promising search idea. The estimate is +0.1–0.3 bb/deal against passive callers, but it takes 1–2 days of work: future work.

**Online strategy selection (implicit modelling, Bard et al. 2013).**
- *Implemented and tested (`selector`, off by default):* importance-sampling evaluation of {mix, heuristic} on every hand, with a conservative ramp.
- *Result:* the estimates of the arm difference swung by ±2–7 bb/hand against a true difference of about ±0.3, and the selector lost 0.07–0.2 bb/deal against balanced bots.
- *Cause:* off-policy evaluation of a near-deterministic heuristic from a stochastic blueprint gives importance weights up to 1/w_min.
- *What it would need:* doubly-robust, AIVAT-style value estimates. Kept disabled.

**Adversarial exploiter search** (`offline/exploit_search.py`, league-training style).
- 96 heuristic variants were evolved over 25 parameters solely to beat the shipped bot.
- Best result: −0.008 bb/deal on the search seeds and **−0.225 on fresh seeds**. The shipped bot is not exploitable by the equity-vs-pot-odds family.

**LLM agents and online platforms.**
- **Frontier LLMs as live players: not viable.**
  - In the GTO Wizard benchmark, every LLM tested lost to a solver bot (best: −16 bb/100).
  - They take seconds to minutes per decision, against our 2–4 ms budget, and there is no network at match time.
- **LLMs as engineers: useful only inside an automated, low-noise evaluation loop** (AlphaEvolve / OpenEvolve style). That is effectively what this project did: agents analyse logs, propose changes, and the rival arena scores them. There is no evidence that mixing several LLM vendors helps.
- **No public platform (OpenSpiel, RLCard, PokerRL, Slumbot, arenas) supports this variant.** The Docker rival arena is the best available platform.

**Split offense/defense exploitation (rejected).**
- *Hypothesis.* A leak analysis against our own blueprint showed:
  - the heuristic's defense over-folds to balanced (polarized) betting, because its top-k range model treats bluffs as strong hands;
  - its offense beat the blueprint's.
- *Policy tested:* heuristic offense always, heuristic defense only against value-heavy bettors.
- *Result, on the robustness seeds:*
  - vs the old release: **−0.48 bb/deal with 8 losses**;
  - vs the blueprint: −0.14;
  - vs the heuristic: −0.20.
- *Lesson:* the "offense is better" conclusion came from one matchup and did not generalize. The robustness gate caught it, and the code was reverted.

## Anti-overfitting rules (every shipped change must pass all of them)

1. **Paired seeds, then fresh seeds.** Compare variants on identical cards. Accept a change only if it also wins on a second, unseen seed set (the winner's-curse guard).
2. **Robustness gate on opponents that were never tuned against:** our CFR blueprint (balanced play), the frozen first release, and the pure heuristic. A change must not lose to them. This gate rejected the pure heuristic, the IS selector and the split policy.
3. **Several opponent families.** A change must hold up against the synthetic pool (stations, maniacs, nits, barrel bots) *and* against the real rivals *and* against balanced bots. The 9 public rivals contain no CFR bot, so beating them alone is not enough.
4. **Prefer mechanism over fitting.**
   - Ship structural fixes with a measured cause, such as range v2 (fixed a diagnosed leak) and the fast estimators (unbiased, verified against brute force).
   - The 25-parameter tuner found nothing robust, so only two parameters (dev0, dev1) were changed, and both were confirmed on two seed sets.
5. **Rank with the right statistic:** expected points from pooled mean and std, not raw win/chop counts.
6. **For any future tuning on real rivals, hold out rivals.** Tune on some, validate on the others (leave-rivals-out), plus the robustness gate.

## How to keep improving

1. **Re-run the rival arena before the deadline** (`offline/rivals/`). Teams keep pushing, so test against their latest commits.
2. The biggest open lever is telling apart passive callers (exploit hard) from trappy bots (play the blueprint). An oracle choice would average about +0.96 bb/deal against these rivals, vs +0.60 now.
   - This needs features beyond aggregate frequencies: check-raise rate, calling-range strength at showdown, river bluff rate.
   - It also needs more rival bots, so a selection rule can be validated out of sample.
3. Blueprint work has hit diminishing returns: neither finer groups (v3) nor 7× more training helped.

## Deep RL track: a GPU-trained neural policy (`BOT_MODE=nn`, branch candidate-drl)

**Goal.** A neural policy trained by self-play on the GPU that is competitive with the CFR blueprint (mode `bp`) head-to-head, delivered as a runtime mode with a blueprint fallback. Everything lives in `offline/drl/` (training, GPU-only) plus `nn_policy.py` and `nn_policy.npz` (runtime, numpy only).

### Environment (`offline/drl/gpu_env.py`)

- **Exact transitions.** `offline/tree.py`'s 41,490-node betting tree (state = node index, `child_a[node, action]`); payoffs from `term_c0/term_c1/term_folder`.
- **Showdowns** from `plo_engine.T[triple, pair]` (the referee's scores): best of 10 board triples × 10 hole pairs by gather + max.
- **Card features**, once per deal, seat and street: preflop equity, Monte Carlo equity vs a random hand, current made-hand strength, potential (256 samples, the estimand of `plo_engine.sample_equity`), made-hand category, and the blueprint v2 bucket.
- **Opponents on the GPU:** the CFR blueprint v2 (bit-exact with `strategy._bp_probs`), the arena's station, nit, maniac, tag and barrel bots, and frozen networks.
- **Parity with `game.py`** (`offline/drl/parity.py`), over 12,000 random hands (89,243 decisions):
  - 0 mismatches in acting seat, legal actions, nets and showdown winners;
  - actor features identical to `nn_policy.obs_features` built from the judge's Observation (max difference 0);
  - blueprint probabilities equal to 1e-7;
  - GPU equity vs `plo_engine.sample_equity` at 4,096 samples: mean difference ≤ 0.0012, spread equal to the Monte Carlo noise.

### Policy (`nn_policy.py`)

- **Inputs (154 + 16):**
  - hole and board multi-hot, rank/suit counts, made-hand category;
  - [preflop equity, equity, made-hand strength, potential];
  - the exact betting state: pot, bets, stacks, raises, legal raise sizes, per-street raise/call counts of both players;
  - a learned embedding of the public betting state (street, position, pot at street start, street sequence).
- **Network:** 3 × 384 ReLU MLP → 4 logits, illegal actions masked. 1.5 MB per member; the numpy forward pass takes 75 µs per decision.
- **Runtime equity** comes from the same time-managed Monte Carlo as `bp`. Any problem (unknown state, exception, NaN) falls back to the blueprint.
- **Population models.** A model file can hold several members (`offline/drl/make_mixture.py`). The bot draws one member per hand, i.e. it plays the uniform mixed strategy (the PSRO meta-strategy).

### Training (`offline/drl/train.py`, RTX 5090, ~0.1–0.3 s per 32k-hand step)

1. **Behaviour cloning of the blueprint** (3,000 steps): KL 0.043 nats, 90% argmax agreement, **+0.004 ± 0.039 bb/deal vs bp** (parity).
2. **Regularised self-play policy optimisation**, Magnetic Mirror Descent style (Sokota et al. 2022), with R-NaD-style magnet resets:
   - objective: maximise Σₐ π(a|I)·A(h,a) − α·KL(π‖magnet), with α = 0.05; the magnet is reset to the current policy every 2,000 steps;
   - **all-actions advantages:** every legal action's child is scored exactly (terminal) or by a privileged critic that sees the whole deal. This is valid because chance outcomes do not depend on actions, and it removes the action-sampling variance;
   - league per hand: self-play 50%, blueprint 25%, simple bots 25%. Only learner decisions train the actor.
3. **From scratch**, the same algorithm without BC reaches blueprint parity after ~4,000 steps and +0.1–0.2 vs bp after ~8,000 (`scratch`, `scratch_b` runs).
4. **Robustness training** (see the exploitability probe below):
   - PSRO-style: best responses to frozen checkpoints join the league;
   - then a co-trained exploiter (AlphaStar-style "main exploiter"), snapshotted into the league and reset every 1,500 steps (`--exploiter`).

Learning curves (GPU duplicate evaluation on 65,536 fixed deals, s.e. ≈ 0.04): `~/poker-arena/gpu/drl/runs/curves.csv` on beast.

### Results: offline/arena.py (judge rules, 1000 duplicate deals per seed, budget scale 1.0)

Variants:
- **Shipped `nn_policy.npz`:** the uniform per-hand mixture of 9 checkpoints from four lines (see "Shipped model" below).
- **Single:** BC+RL at step 3,500, shipped as `nn_policy_single.npz`.
- The reference rows for `bp` and `mix` use the same seeds (101–404).

| Opponent | seeds (pop / single) | **nn population (shipped)** | nn single | bp (ref.) | mix (ref.) |
|---|---|---|---|---|---|
| **bp (CFR blueprint)** | 24 / 16 | **+0.113** [−0.017, +0.244], 2/22/0 | **+0.200** [+0.042, +0.358], 0/16/0 | — | — |
| **mix (shipped blend)** | 16 / 8 | **+0.118** [−0.042, +0.279], 0/16/0 | +0.003 [−0.222, +0.229], 0/8/0 | — | — |
| bp1 (blueprint v1) | 4 | +0.412 [+0.082, +0.741] | +0.410 [+0.057, +0.764] | | |
| heur (pure heuristic) | 4 | +1.091 [+0.799, +1.384] | +1.238 [+0.897, +1.580] | | |
| rel (first release) | 4 | +0.889 [+0.610, +1.168] | +1.065 [+0.734, +1.397] | | |
| station | 4 | +8.516 | +8.904 | +2.509 | +8.089 |
| nit | 4 | +1.840 | +1.932 | +0.810 | +1.209 |
| tag | 4 | +3.788 | +3.886 | +1.647 | +2.804 |
| maniac | 4 | +6.648 | +5.061 | +4.221 | **+12.774** |
| barrel | 4 | +5.286 | +4.346 | +3.174 | +4.396 |

Brackets are 95% CIs; W/C/L counts the per-match verdicts. Every simple-bot match is a WIN for every variant.

- **Other checkpoints vs bp:**
  - BC alone: −0.178 [−0.483, +0.128] (4 seeds, parity);
  - BC+RL at step 2,500: +0.470 [+0.252, +0.688] (8 seeds, run under CPU contention);
  - PSRO-1: +0.177 [−0.047, +0.401] (8 seeds);
  - from-scratch RL: +0.230 [−0.002, +0.462] (8 seeds);
  - a 4-member population: +0.086 [−0.077, +0.250] (16 seeds) and vs mix +0.125 [−0.106, +0.356] (8 seeds).
- **`nnmix`** (the mix machinery with the NN as the anchor): vs mix +0.118, vs bp +0.247 (4 seeds each), weaker head-to-head than pure `nn`.
- **Think time:** 2.50 s per half (our compute, idle core), the same time-managed Monte Carlo budget as `bp`, including vs the calling station (worst case: every hand reaches the river).
  - The NN costs ~150 µs per decision (features + forward pass) inside that budget.
  - Startup with `BOT_MODE=nn` takes 1.4 s; the 9-member file loads in 0.16 s.

### Exploitability probe (the decisive robustness test)

**Method.** A fresh best response (BR) is initialised from the BC policy and trained without regularisation for 2,000 steps × 32k hands against a frozen target (`--frozen <ckpts> --mix frozen:1.0 --alpha 0`). For a population, the BR plays a random member per hand, and the mixture value is the mean over members. Every target gets the same budget.

**The GPU numbers transfer to the CPU runtime.** In `offline/arena.py`, over 4 seeds:
- bp vs its BR: −0.473;
- the single NN vs its BR: −2.350 (4/4 LOSS);
- the 4-member population vs its BR: −1.590 (4/4 LOSS).

| Target | BR winnings after 2,000 steps (bb/deal) |
|---|---|
| **CFR blueprint (bp)** | **+0.42** |
| BC+RL step 3,500 (single) | +2.31 |
| PSRO iteration 1 | +3.61 |
| from-scratch RL (cumulative 9,000) | +3.94 |
| rob1 (co-trained exploiter) at cumulative 13,000 / 17,000 | +2.06 / +2.06 |
| population of 4 | +1.41 |
| population of 8 | +1.29 |
| **population of 9 (shipped)** | **+1.33** |
| population of 8 + blueprint on 30% of hands | +1.05 (≈ convexity bound) |

**What the exploiters do.** They pot-raise every hand preflop, call every raise, then barrel: the RL policies over-fold to relentless aggression.

**What was tried against it.**
- PSRO-style hardening (each BR joins the league) patches each hole within ~500 steps, but the next BR finds a new one (classic early-PSRO whack-a-mole).
- A co-trained exploiter, snapshotted into the league every 1,500 steps, steadily neutralises older exploiters, but slowly.
- Deploying the population mixture (the PSRO meta-strategy) is the cheapest robustness gain.
- None of these approaches the blueprint's +0.42 in the time available.

### Shipped model, how to enable, recommendation

**`nn_policy.npz`** (6.9 MB, float16): 9 members, one drawn per hand.

| Member | Checkpoint | Cumulative RL steps |
|---|---|---|
| m0 | BC+RL | 3,500 |
| m6 | BC+RL | 5,500 |
| m1 | PSRO-1 | 8,500 |
| m7 | from-scratch | 8,000 |
| m2 | from-scratch | 8,500 |
| m3 | rob1 | 11,500 |
| m4 | rob1 | 13,000 |
| m5 | rob1 | 15,000 |
| m8 | rob1 | 17,500 |

`nn_policy_single.npz` (1.5 MB) is the single best head-to-head policy (BC+RL step 3,500).

**How it is enabled.** `nn` is the default mode of the submitted bot.
- `BOT_MODE=nn` plays the deep-RL policy.
- `BOT_MODE=nnmix` plays the mix with the NN as the anchor.
- `BOT_NN_PATH=/path/to/file.npz` picks the model file.
- `make_mixture.py ... bp=<share>` gives the blueprint a share of the hands.
- For a submission, change the default of `MODE` in `strategy.py`; nothing else is needed, since the Docker image already contains both files.
- In the arena, use `nn`, `nn:<ckpt.npz>`, `nnmix` or `nnmix:<ckpt.npz>`.

**Verdict.**
- The deep-RL policy is competitive with the CFR blueprint head-to-head: the single policy is significantly ahead (+0.20 bb/deal over 16,000 deals); the population is ahead but not significantly (+0.11 over 24,000).
- It ties the shipped mix and beats the blueprint's margins against every simple bot. The mix keeps a large edge only against the maniac (+12.8 vs +6.6), thanks to its opponent model.
- **It is 3–5× more exploitable than the blueprint** (+1.33 to +2.31 vs +0.42 for a 2,000-step learned best response). The exploits are hyper-aggressive preflop raising and barrelling, the profile of the strongest field bots.
- `nn` became the submission after the rival arena (`offline/rivals/`) showed it ahead without losses; with ~1,000 deals per match, most strong-bot matches are CHOPs either way.
- Next steps for robustness:
  - more PSRO iterations or a longer co-trained-exploiter run;
  - a larger KL weight toward the BC policy;
  - an opponent-statistics input, so the policy can adapt within a half the way the mix does vs the maniac.
