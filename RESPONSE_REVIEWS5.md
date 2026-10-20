# Response to the five independent reviews (paper at a19de16)

18 Oct 2026. Every finding was checked against the data before anything was changed. "Verified" means I
recomputed it with my own code; "accepted" means I traced it but did not recompute it, because it is wording or the
reviewer's evidence is direct. No result was deleted and no registered file was edited (`theory_v3.py`,
`theory_v34.py`, `localcheck.py`, `heldout*_v3.py`, the prediction files, `liu_mpga.py`). Script fixes were followed
by a full rebuild of the report on the Linux copy of `results_v2` ("all checks passed, all steps ran").

Outcome in one line: **no headline number changes**. Some wording was false and is fixed. Three numbers move: the
warning recall (0.55 to 0.54), Table XI's p90 column, and the conservation ratio, which is now tighter. One appendix
conclusion was wrong and is rewritten (R3 H-1). The registration and summary wording now says what the record supports.

## Still needs the author (not done)

| item | why | what I wrote meanwhile |
|---|---|---|
| R4 H1: Gao/Asghar/Liu/DRL registration records | not in the public repo | "fixed before any held-out run (the record is kept outside the public repository)"; "hashed" removed. If the BMP-repo records exist, commit them and the wording can be strengthened |
| R4 M6: DRL run logs | not released | the paper says they are not among the released files |
| R5 M5: `results_v2` | not public | App. C says it is attached to the tagged release, so the release must be made before submission |
| R4 H2 (optional run) | the registered downward fill was never run | counts and near-ties are disclosed in VII-A |
| R3 M-2 (optional run) | the 24 split-fleet sweeps were not extended | disclosed: reductions are lower bounds, location claims provisional |
| author block | four `\todo` | — |
| R5 L11 | `registrations/heldout4_manifest.sha256` says "registered 2026-10-13"; the push was 14 Oct 12:54 IST | left unedited (it is a registration record); the paper gives the GitHub push times |

## Review 1 (model and closed-form theory)

| finding | verdict | action |
|---|---|---|
| M1 interval coverage over the wrong population; width not reported | verified: 95/95 flagged; median width 38 core (16-74), 14 clustered (5-40); round Kc = K0 never | V-C and VII-K rewritten; "at least K0+1" dropped; ex-ante coverage 98/98 (verified) |
| M2 Prop. 1 inventory inconsistent for non-integer K_air | accepted (algebra) | statement now integer K_air with exact staggering; non-integer case needs a pool of ceil(K_air)U/K_air; third qualification corrected |
| M3 drift does not test (a) | accepted | rewritten: tests only the served-set part of (a); "supports" sentence removed |
| M4 criterion directions; calibration; 34/72 | verified 34/72 | three derived, two measured; radial scale kept because it flags correctly here, a calibration not a bound; 47% given beside 88/94% |
| L1 hover inconsistency | accepted | one sentence: excluding hover from both gives Psi' = Gamma exactly |
| L2 sensor-side transmit limit | accepted | "exactly" removed; equivalent only as an effective drop-head buffer |
| L3 Corollary 1 hypotheses | verified (share spent 0.99 / 0.99 / 0.93, p10 0.66 core) | both hypotheses added, with the numbers |
| L4 conservation uses nominal rate | verified: 0.95-1.01 with actual rates and the <5% empty filter; 0.95-1.04 without | `commute_exact.py` fixed; text gives both and states the filter |
| L5 boundary disagreement band | accepted (0 of 272 in either region) | sentence completed |
| L6 vacuous hypothesis; smaller fleets | accepted | hypothesis deleted; one line on why smaller fleets are dominated |
| L7 r_c(K) heuristic | accepted | "suggests ... heuristically" |
| L8 t_c two meanings | verified (0.7652 L/v = 8.0 min) | new symbol t_w; launch stagger stated |
| L9 notation | partly | Table I extended (t_w, tau_hop, t_d, T_b, T, H, Phi, Gamma, C, c_i, pi_i, Psi, sets, K_R, K_0, l, K_ref, kappa, eta, nu_i). Symbol overloads (R, C, A, B) left as they are; renaming them risks new errors this close to submission |
| L10 "smallest-K" inputs | verified (56 of 272 differ) | K_ref defined in VII-A; wording fixed in V-A and the Table XIII caption |
| L11 +66% / +7% | verified by my own Monte Carlo: +64.0% / +7.8% | changed to +64% / +8% |
| L12 eta Monte Carlo noise | accepted | SE and the 20k-field values given |
| L13 "reduces it to" | accepted | "replaces it with ... need not lie inside it" |
| L14 tau_hop "directly" | verified (theory_checks.py uses theta(Ts-tc)/n) | reworded, direct values cited |
| L15 battery-mass wording | accepted | "reach cap is exact"; "small on clustered fields" |
| L16 non sequitur | accepted | "the surrogate's minimum lies above K0" |
| L17 CV scales | accepted | both on the 1+CV^2 scale |
| L18 r_c includes empty sorties | verified: 51 instances, max 6.4% | one sentence in VII-A |
| out of scope: stale `report/` in data zip | accepted | remove it from the release archive |

## Review 2 (served-set theory)

| finding | verdict | action |
|---|---|---|
| H-1 Theorem 1 / Result 2 / "reverts to l" fail for v3.4 | accepted (mechanism is immediate; 24/160 ring fields) | Theorem 1 stated for a served-set-only hop; Result 2 defined as search over admissible K <= 200; v3.4 caveat and the 924/924 equivalence for v3/v3.3 added |
| H-2 Remark 5 disclosure covers test 3 only | **verified** with my own fixed-point code: 7/160 change in test 4; +-1 iteration 15; at 21 iterations the check fails L4 (21.5%, 18.6%) | Remark 5, VII-E local-check paragraph and Discussion rewritten; "two cell types" fixed; THEORY_V3 §14 |
| M-1 Corollary 1 "no prediction changes" | accepted | "one v3.4 near-tie in test 4, check unchanged" |
| M-2 "special cases" | accepted | F and Prop. 8 are a different surrogate to which Theorem 1's argument applies |
| M-3 gray evidence mixes cases | accepted | "5 of these 7 involved gray sensors; test-1 fields became design data" |
| lows 1-9 | accepted | d_S wording; level-bias conditions incl. 70/72; first-order "d_S held fixed"; Prop. 6 "estimated", cluster patrol 49/72, holds to +-1 for J; K >= 1 and no-c2 case; Algorithm 4 qualifiers; r_bar_S, H_S, nominal rates, K <= 200 |

## Review 3 (validation)

| finding | verdict | action |
|---|---|---|
| H-1 App. B "refutation" is a pooling artefact | **verified**: between-run beta_hop 1.76/1.54/1.23 (logs of means 1.67/1.49/1.26), within 0.60/0.19/-0.03 | `alpha_fit.py` now prints the decomposition; App. B per-sortie paragraphs rewritten; VII-C clause dropped; alpha kept as a diagnostic because it is fitted and varies with M |
| M-1 warning indicator still at smallest K | verified: 0.55 -> 0.54, 0.46 -> 0.45 | `strand_planners.py`, `paper_numbers.py` (3 reads), `analyze_depot.py` fixed; text updated |
| M-2 split-fleet sweeps not extended | accepted (edges.txt: 24/24) | disclosed; lower bounds, provisional location claims |
| M-3 intervals conditional on cells | accepted | stated, with Clopper-Pearson [71,82] and cell bootstrap [60,92] |
| L1 Table XI p90 interpolated | verified: 5.9 / 16.9 / 21.7 / 1.1 / 41.1 | nearest rank (`method='inverted_cdf'`) in every p90 script; table updated; other tables unchanged |
| L2 9.3-9.8% | accepted | "about 9-11% (8.7 / 10.6)" |
| L3 Table VI label | verified (seed 3 cap 5) | label and caption corrected |
| L4 "four disagree in 8 of 24" | accepted | definitions and the half-even note |
| L5 2% -> 3% | accepted | fixed |
| L6 relative-bias definitions | accepted | stated; "same jump" softened |
| L7 "within 10%" | accepted | "about 10%" |
| L8 depot fill rows lack coordinates | verified: edges check now 0 of 216 / 48 | assembly stamps the depot on fills; App. C says so |
| L9 analyze_depot at Ks[0] | accepted (no change in result; rebuild confirms) | fixed |
| L10 "82 held-out" | accepted | scoped |
| L11 "single pilot sortie" | accepted | "short pilot deployment" |
| L12 per-leg statistics | accepted | named |
| out of scope: hover-free variant uses nominal H | R1 found 7 with actual rates too | text says "nominal or actual" |
| out of scope: M=40 and mass sweeps fail 1.5x | verified (13/24, 59/72) | stated in the Table XIII caption and Remark 1 |

## Review 4 (planners, baselines, value of information)

| finding | verdict | action |
|---|---|---|
| H1 baseline registration records missing | cannot verify from here | wording reduced (see top table); **needs the author** |
| H2 registered downward rule unmet | **verified**: 97/125/85/155/170/166/167/108/164 and the six near-ties | VII-A states the rule applied, the counts and the near-ties; "no sweep needs a smaller K" deleted; C4 and Gao/Asghar paragraphs point to it |
| M1 VOI test choice and bound | **verified**: Wilcoxon .06/.23/.13 (kernel), .009/.03/.06 (prior-rates); bootstrap upper 4.3-5.6% | `singlecell_v2.py` prints the extra tests; caption, text, abstract-adjacent intro bullet say "one cell, at most about 5-6%" |
| M2 better neighbour | verified (K*-1: 26.2 / 26.8%) | "better of the two neighbouring sizes" in three places |
| M3 25% understates | verified (-25.2 to +64.4%) | per-deployment range given |
| M4 stability scoped to field-sharing planners | accepted | scoped in intro, abstract and VII-L |
| M5 registered vs rerun figures | accepted | registered figures (-21.0/-10.6/-6.6, 204/216; 34.0/61.6) stated |
| M6 DRL registered comparison | accepted | stated as registered and withheld; logs **needed** |
| M7 "published" too strong | accepted | "clustering stage of a published method"; "strongest of our ports"; Liu surrogate caveat |
| M8 CP-SAT | verified 97.6% (42,387/43,424) | VI-A "within one in 24, identical in 18"; proof share of runs used; integerisation note |
| L1-L16 | accepted | greedy seed and SA settings; ETA proxy; max(q,1e-3); tau_hop measure (33.1 vs 40.9); 0.5% / 3.7%; median -0.4%; five of six planners; Fig. 4 seed change; K_ref caption; Liu and SA cell losses; priorities for Gao/Asghar; Asghar CP-SAT orienteering; M=40 caption; latency provenance and machines; "Other designs"; Liu discard wording; scope of the M=40 statement; M <= 15 cells |

## Review 5 (held-out tests and summary claims)

| finding | verdict | action |
|---|---|---|
| H1 abstract 637 words | verified | rewritten to 250 words; C4, CP-SAT and VOI material stays in the intro |
| H2 test-1 registration overstated | accepted (THEORY_V3 §6 already said so) | VII-E uses the reviewer's registration text with GitHub push times; Table VIII "106 new"; verdict "2 of 3 on 16-20"; conclusion "three of them on entirely new fields" |
| H3 five failed verdicts, not two | verified | 38 of 43 in intro, VII-E, conclusion and abstract |
| M1 territorial wording | verified (C4 98.8->29.3, 93.0->24.3, 27.6->37.8) | "cut the tail in two tests, raised it in the third, never predicted the optimum better" |
| M2 three rule versions credited as one | verified (v3.3 alone 31.9/38.7/58.1; p90 17-19) | abstract says "registered versions"; VII-E gives v3.3 alone |
| M3 7/7 and 5/7 | verified: 6/6 p=.016, 4/6 p=.34 | paper and THEORY_V3 corrected |
| M4 Algorithm 4 unreachable branch | accepted | reordered: full sweep, local check, served-set rule, l with test, else interval + candidates |
| M5 App. C promises | accepted | `REPRODUCE.md` added; LF-normalisation stated; release **needed** |
| L1-L14 | accepted | 2.9%; seeds 21-28 (horizon arm 21-40); 6 of 8; test-1 one-sided sweeps; independent code only for tests 3-4; cluster patrol 6 clustered + 1 ring; intro run counts; v3.4 registered twice; "held out from C4's design"; Result 1 "exactly"; manifest date (left, see top); table labels in `reproduce_v2.py`; `.gitignore`; k-means wording removed |
| out of scope: IV-E "even" | verified via R2 M-3 numbers | "better in 3, worse in 4" |
| out of scope: VII-C peak at K=8 | accepted | "8 at M=50 and 100, 7 at M=200" |
