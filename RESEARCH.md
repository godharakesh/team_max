# Heads-Up Short Deck PLO5 bot: prior-art research

Researched 2026-11-07, about 20 minutes. Sources are in `CITATIONS.md` as [C1], [C2], and so on. **[UNVERIFIED]** marks claims I could not confirm from a primary source. **[MEASURED]** marks numbers I measured myself with the kit's `evaluator.py` on this laptop (pure CPython, 1 core).

## 0. Facts measured with the kit (these matter most for design)

| Quantity | Value |
|---|---|
| `evaluator.warmup()` (builds the 377k-entry table) | 0.41 s [MEASURED]. Do it at import time, not inside a request. |
| `omaha_score(hole5, board5)` | ~18 µs per call (10 pairs x 10 triples of dict lookups) [MEASURED] |
| One Monte Carlo trial (sample opponent + runout, 2 scores) | ~53 µs [MEASURED], so about 19 trials per ms |
| Time budget | 8 s per 1000-hand half. Each request gets 1 ms free, the hard cap is 1 s, and 5 errors forfeit the half (`game.py`). At ~2-4 decisions per hand that is **~2-4 ms per decision, i.e. only ~40-80 naive MC trials**. |
| Preflop equity vs a random hand, heads-up (300 random hands x 600 trials) | min 0.24, p10 0.42, p25 0.46, **median 0.51**, p75 0.55, p90 0.59, max 0.65. SD 0.065 [MEASURED] |
| Worst hands | trips in hand with dead cards (`6s 7h Jc 6h 6d` = 0.24, `6h 9s 6d 6s Jc` = 0.30), plus disconnected rainbow junk (~0.36) |
| Best hands | AA or KK double-suited with broadway or connected side cards (~0.64-0.65) |
| MC standard error, 0.5/sqrt(n) worst case | n=64: 6.3%, n=100: 5%, n=256: 3.1%, n=400: 2.5%, n=1000: 1.6% |

What this implies:
1. Preflop equities are tightly bunched: 80% of hands fall between 42% and 59%.
2. Live MC preflop is a waste of the time bank. Use a precomputed table or formula.
3. Postflop MC has to be cheap and adaptive.
4. Memory is wiped and the process restarts every half (`agent.py` comments). Load all tables at module import. Do not load them in `new_half`, because that is a timed request (only startup, 15 s, is free).

## 1. Ranked finds

| # | Find | License | What it does | Reusable here | Must adapt |
|---|---|---|---|---|---|
| 1 | **The kit's own `evaluator.omaha_score`** (local) | the kit's | Exact short-deck PLO5 scoring through a 36-bit-mask dict | It is the referee's ruleset (flush > FH, trips > straight, A-6-7-8-9 wheel), so there is no risk of rule mismatch. | For speed: on the river, score hero once and only score the opponent in each trial. Precompute the board's 10 triple masks once per street. |
| 2 | **gtowizard-ai/mitpoker-2024** [C7] | MIT | Top-team bot for MIT Pokerbots 2024 (Auction Hold'em) under a very similar budget (1000 hands / 30 s, 1 GB) | The architecture: a `TimeManager` that splits the bank per street, a fast evaluator, precomputed equity matrices, and depth-limited solving with "all-in equity" leaf values. Its README says abstraction MCCFR "didn't scale" to their time limits. | Different game. Take the TimeManager idea and the precompute-everything pattern, not the code. |
| 3 | **pokerkit** (uoftcprg) [C1] | MIT | Pure-Python engine for many variants (NLHE, short-deck NLHE, PLO) with custom games | Offline sanity checks: cross-check `omaha_score` and PLO pot-limit raise sizes. | The README does not list 5-card or short-deck Omaha, so it would need a custom game definition. Too slow and heavy for match time [UNVERIFIED speed]. Needs Python ≥ 3.11. |
| 4 | **suryabhupa/MADbot**, MIT Pokerbots 2016 **HU PLO** entry [C9] | none stated (treat as all-rights-reserved) | Python equity-threshold fold/call bot. v2 swapped the evaluator for pretrained NN win-probability nets. | The closest prior game (HU PLO). Lessons from its README: v1 "probably too aggressive", one version had a bug where the bot never initiated bets, and thresholds were tuned on self-play logs. | Read only, do not copy (no license). |
| 5 | **Gongsta/Poker-AI** [C3] | MIT | HU NLHE: MC equity (EHS) + K-means card buckets + CFR blueprint, with pot-relative bet abstraction | The EHS-plus-k-means bucketing recipe if we ever try CFR. Has a pot-odds helper. | NLHE with 2 hole cards. The README is inconsistent about cluster counts and the CFR is marked WIP. |
| 6 | **brandinho/Poker-Probability-Approximation** [C5], paper arXiv 1808.07220 | **none stated** | An NN that approximates MC win probability, replacing lookup tables | The idea: fit a small regressor offline (numpy MLP or GBM on hand-shape features) to equity from millions of offline MC trials, then evaluate in microseconds at match time. | Hold'em features. Retrain on our game. Do not copy code (no license). |
| 7 | **Ponsen, de Jong, Lanctot, MC Restricted Nash Response**, arXiv 1401.4591 (JAIR 2011) [C6] | paper | MCCFR for equilibria, plus MCRNR for robust best responses to a fixed opponent model | The principle: exploit opponent models but limit how far you deviate from a sound baseline, so a wrong model does not get punished. | Full RNR requires solving the game. We approximate it with "exploit only after enough samples, with capped adjustment". |
| 8 | **Maîtrepierre, Mary, Munos, Adaptive play in Texas Hold'em** (ECAI 2008) [C10] | paper | Several base strategies, with a UCB bandit choosing between them per opponent | A cheap way to exploit: keep 3-4 parameter presets (tight-value, standard, hyper-aggressive bluffer, anti-maniac call-down) and pick between them per half using UCB on bb/hand. | Duplicate format and only ~1000 hands per half means few samples. Use UCB over blocks of ~50 hands, or simple stat thresholds. Their bot placed 8th of 17 in AAAI'07, so the method is not a proven winner. |
| 9 | **fedden/poker_ai** [C2] | **GPL-3.0** (copyleft) | MCCFR + clustering for short-deck Hold'em | Reference only for MCCFR structure. | Archived 2024. "Only supports a 20-card deck". GPL would infect our submission. **Do not copy.** |
| 10 | **ozzi7/Poker-MCCFRM** [C4] / **diditforlulz273/PokerRL-Omaha** [C8] | MIT / MIT | Pluribus-style MCCFR for NLHE (C++, wants >128 GB RAM) / Deep CFR for PLO (GPU + workers) | Shows how big full CFR is: 169-1000-1000-1000 buckets and tens of millions of iterations. | Both far beyond our time and compute. Read for ideas only. |

Not useful or not verified:
- **aipoker.bot** HU PLO leaderboard: I found only third-party directory blurbs about leaderboards with CIs, open hand histories and 55 AI opponents. I could not see bot strategies or rankings **[UNVERIFIED]**.
- **ACPC**: CFR equilibrium bots won almost every year [C11]. A 2010 analysis found ε-equilibrium bots generally beat opponent-modelling bots [C11]. That field was full of strong near-equilibrium bots, which is not our field.
- **arXiv 2512.04714** ("Playing the Player", heuristic exploitative AI) [C12]: abstract only, details not verified.

## 2. Approaches the strongest prior work uses

1. **Equilibrium (CFR/MCCFR) blueprint over card buckets plus an action abstraction**: ACPC winners, Libratus/Pluribus, the 2017 MIT winner Fro116 [C13]. It wins against strong fields, but needs heavy offline compute plus careful off-tree handling.
2. **Precompute everything, then do a cheap real-time decision under a strict time manager**: GTO Wizard's MIT 2024 bot [C7], the 2021 MIT writeups [C14]. Typical pieces are equity matrices or tables, a fast evaluator, and per-street time allocation. Some add depth-limited solving with equity leaves.
3. **Equity vs pot odds plus opponent-stat adaptation (aggression/looseness)**: J. Rubin's MIT 2nd-place team [C14], the 2011 MIT winner (opponent modelling, Huang thesis) [C15], MADbot [C9], Maîtrepierre's UCB [C10]. In student competitions against heterogeneous, mostly weak bots, this approach places well.

## 3. Pitfalls others hit

- **Time-bank exhaustion and timeouts.** The budgets in these competitions are milliseconds per action [C7]. A bot that does fixed-n MC every decision runs out late in the half. Then the kit's `default_action` (check/fold) plays for you, or errors pile up to a forfeit at 5. Fixes:
  - Track remaining bank in `memory` and budget per decision as `remaining / expected_remaining_decisions`.
  - Enforce a hard per-call wall-clock cutoff, well under 1 s.
  - Wrap `choose_action` in try/except that returns a legal fallback.
- **Rule and ruleset mismatches.** Short-deck rooms differ on flush vs full house and trips vs straight [C16]. Any third-party evaluator (pokerkit, treys, PokerRL) can be silently wrong for this game. Use only the kit's `omaha_score`.
- **Bugs that look like strategy problems.** MADbot shipped a version that "didn't initiate bets" [C9]. Write tests that assert the bot raises with the nuts and never folds when `to_call == 0` (folding for free is a pure leak).
- **Over-aggression without bankroll or opponent awareness** (MADbot v1 [C9]).
- **The opposite failure, over-folding against maniacs.** Against `always_pot`-style bots, PLO pot-limit sizing means equity of about 1/3 is enough to call a pot bet. Heads-up PLO5 hands rarely fall below 30% equity, so tight thresholds bleed chips to maniacs.
- **Bluffing calling stations.** Against bots with high call frequency, every bluff loses. Value-bet thin and bet big.
- **Exploitability of model-based bots.** Opponent-modelling bots lost to ε-equilibrium bots in ACPC 2010 [C11]. Only deviate from the baseline after enough evidence, the RNR idea [C6].
- **Measurement noise.** MC at n≈64 has ~6% SE. Near-threshold decisions are coin flips, so spend more samples only when |equity − required| is small.
- **Equity vs random overstates hands against strong ranges.** It understates nothing against random-ish bots, but against tight bots that only bet strong hands, discount equity when the opponent raises (shift toward a top range).
- **Duplicate variance and CI.** 1000 deals gives a wide CI. Hands lost by folding pots with 40-50% equity cannot be won back, so small systematic leaks decide matches.

## 4. Can CFR/MCCFR be precomputed here?

- **Betting tree: yes, it is tiny.** 4 actions, a 4-raise cap, and 40 bb pot-limit stacks mean money goes in after about 3 pot raises. Each street has a few dozen sequences.
- **Card abstraction: this is where it gets hard.**
  - C(36,5) = 376,992 hole hands, about 16k up to suit isomorphism (rough estimate).
  - Postflop, PLO5 needs bucketing by EHS and potential (for example 50-200 buckets per street).
  - A numpy MCCFR over such an abstraction is plausible offline in hours. But bucket quality for PLO5 draws is poor, and the payoff against a field of heuristic bots is smaller than straightforward exploitation [C11, C14].
- **Verdict: not worth it for this hackathon.** At most, run a small preflop-only solve or self-play threshold tuning.

## 5. Decision: reuse / adapt / build

**Build from scratch. The research supports the tentative plan**, with these changes:
1. **Preflop is a table, not MC.** Offline, compute equity vs random for canonical (suit-isomorphic) hands, or fit a small regression. Features: pairs/trips in hand (trips are bad), number of suits, double-suited, A-suited, connectedness or gaps for short-deck straights, high cards. Ship it as a compact file and load it at import. Mind `.dockerignore` and the image size limit.
2. **Postflop: adaptive MC driven by the time manager.**
   - Use ~40 samples, then extend in batches only when the decision is near the pot-odds threshold.
   - On the river, hero's score is fixed: compute it once and sample only opponent hands (~2x faster).
   - Cache hero-specific board masks per street. Reuse the street's equity across repeated decisions on the same street, adjusted for the opponent's action.
   - Optionally use range-weighted sampling: reject-sample opponent hands with low preflop-table equity after they pot-raise.
3. **Opponent model:** per-half counters for VPIP, preflop raise %, aggression factor, fold-to-bet %, and showdown hand strength from `opp_hole` (this field exists in `HandResult`). Map them to 3-4 presets (see the tips below). Switch only after ~30-50 hands and shrink toward the default (RNR-style). Optionally use UCB on blocks of hands [C10].
4. **Safety:** try/except fallback, a never-fold-when-free rule, wall-clock guards, and a test suite against all four baselines plus a self-copy.

Reuse: none of the third-party code (licenses, wrong variant, too slow). Adapt ideas from gtowizard mitpoker-2024 (time manager), brandinho (learned equity), and Maîtrepierre/Ponsen (safe adaptation). Use pokerkit only offline, if needed for cross-checks.

## 6. Concrete tuning tips

- **Almost never fold preflop heads-up.**
  - From the SB facing only the BB, a call needs 25% equity raw. Only ~1-2% of hands are below that (trips-in-hand junk).
  - Facing a pot raise to 6, you call 4 to win 12, which needs ~33%. Fold only about the bottom 3-5% vs random-range bots. Fold more vs a tight bot whose raises mean AA-type hands.
- **Raise preflop more, not less.** Open-raise roughly the top 40-60% by table equity from the button/SB and 3-bet the top ~15-20%. Since equities are close, the value of preflop aggression comes from fold equity and position, not raw equity. Against calling stations, raise only for value (top ~30%).
- **Position matters.** In position, realise about your equity or a bit more. Out of position, discount equity by roughly 3-8 points for draws when deciding to call multi-street (heuristic, **[UNVERIFIED]** magnitude; tune by self-play). GTO Wizard used an over-realisation multiplier and noted it should differ by position [C7].
- **Postflop thresholds:** call if equity > to_call / (pot + to_call) + margin. Margin is ~0 vs maniacs and +5-10% vs tight-aggressive bots. Bet for value when equity vs the calling range is > ~55-60%. Choose pot vs half-pot by opponent fold rate.
- **Stacks are short, so commit early.** At 40 bb pot-limit, a flop pot bet plus a raise is roughly all-in. With equity ≥ ~45% on the flop facing a big bet, prefer raising pot (getting it in) over flat-calling and facing a turn decision out of position.
- **Short-deck specifics:**
  - Straights are common and trips beat them.
  - Flushes are rarer (9 cards per suit) and beat full houses, so flush draws plus nut-flush blockers matter more than in normal Omaha.
  - The A-6-7-8-9 wheel and denser ranks make wraps strong.
  - Raw MC handles all of this automatically. For bluffs, use blocker heuristics: hold the nut-flush card on a monotone board.
- **Against a maniac (always pot or raise):** call down much wider, with no bluffs and no fancy plays. Raise for value with ≥ 55% equity and let them bet.
- **Against a calling station:** never bluff, bet pot with ≥ 55-60% equity, check back marginal hands, fold more to their rare raises.
- **Against a random bot:** value-bet wide and call with pot odds. It will be crushed by plain equity play.
- **Spend the time bank where pots are big.** Use few samples when the pot is ≤ 4 bb and more on turn and river all-in decisions.
- **Testing:** use duplicate self-play with the provided `judge_engine.py` against all baselines plus earlier versions. Report bb/deal with a CI and look for regressions, not single runs.

## 7. Addendum: what we built after this research (decision update)

The user asked for frontier techniques, so we went beyond the "build a heuristic" decision in §5. We added a scaled-down Libratus/Pluribus recipe and kept the heuristic as a measured benchmark and fallback:

- **Equilibrium blueprint.** External-sampling MCCFR with Linear-CFR discounting, over the *exact* betting tree. With only 4 actions and a 4-raise cap, no action abstraction is needed. Card abstraction is E[HS] percentile buckets: 16 preflop, 12 per later street. 840k iterations produced 12.4k infosets. This addresses the §4 concern about card abstraction: equity buckets computed with the same estimator as play-time keep the abstraction consistent and cheap.
- **Vectorised equity engine.** It replaces naive `omaha_score` Monte Carlo (§0: ~53 µs/trial) with numpy gathers over a (triple × pair) score table, making each sample about 20× cheaper. It is verified identical to the referee's evaluator. Monte Carlo runs at most once per street; later queries reuse the cached samples.
- **Safe exploitation (Restricted-Nash-Response spirit, Ponsen et al. / Johanson et al.).** The bot mixes blueprint and exploitative actions with a weight that grows with confidence in the opponent model and with how far that opponent is from balanced play.

Measured outcome (BOT_DESIGN.md, Results):
- The mix wins against all 8 sparring bots and all 4 baselines.
- It is head-to-head even with both its components.
- It matches the pure heuristic's margins against weak bots.

This is consistent with the ACPC finding [C11] that equilibrium bots are robust while exploitation earns more against weak fields. The mix is our attempt to get both.

### Addendum 2: second iteration (same day)

- Potential-aware abstraction (equity × draw-potential groups) solved with JIT-compiled MCCFR (numba, 12M iterations). It beats the first blueprint by +0.43 bb/deal over 16,000 duplicate deals.
- Finer groups (v3) and 7× more iterations both gave no further gain. The limit is the play-time equity noise.
- Showdown-calibrated opponent ranges.
- Exploitation weight now measured as deviation from *equilibrium* behaviour. This fixes a real failure: the old norms mistook a balanced bot for an exploitable one.
- Tuning objective = expected tournament points, using common random numbers and a fresh-seed winner's-curse guard. Details and evidence are in BOT_DESIGN.md.
