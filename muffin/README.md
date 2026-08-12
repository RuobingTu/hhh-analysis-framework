# MUFFIN for the v29pre 1tau0l jet→τ_h background

MUltivariate Fake-Factor INference — the binned fake factor replaced by a
continuous, multi-dimensional BDT reweighting giving a per-object weight
(I. Andreou, D. Colling, D. Winterbottom; LHCP poster, CMS-PAS-TAU-25-001):

```
w_MUFFIN(z) = [p_pass^data(z) − p_pass^sim(z)] / [p_fail^data(z) − p_fail^sim(z)]
```

This is the *same quantity* the binned map delivers as `FR/(1−FR)`: the 1tau0l
denominator is inclusive-loose (tight included), so
`FR/(1−FR) = N_pass/(N_loose−N_pass) = N_pass/N_fail`. MUFFIN only changes how
the ratio is estimated — continuously in seven variables instead of in
`jetPt × |η|` cells per prong.

## Result (MR → VR, v29pre 1tau0l)

Both predictions are built from the same VR fail events with the same signed
weights, so the only difference is the fake factor. Metric is the poster's
figure-2 quantity `sqrt((1−r)² + σ_r²)`, `r = pred/obs`, yield-weighted over the
bins of each variable — smaller is better.

| projection | MUFFIN | binned map |
|---|---|---|
| **τ_h pT** | **5.9%** | **13.8%** |
| seeding jet pT | 7.2% | 6.9% |
| number of jets | 4.5% | 4.4% |
| \|η(τ_h)\| | 6.2% | 5.3% |
| MET | 5.6% | 5.4% |
| seeding jet DeepFlavB | 17.5% | 12.3% |

Integral VR closure: MUFFIN 0.975, binned map 0.977 (both methods share the
same ~2.5% MR→VR non-closure, so it is an extrapolation effect, not a MUFFIN
one).

The gain is concentrated in τ_h pT, which the binned map has no axis for.
Feeding the BDT **only** the binned map's own variables reproduces the binned
map's performance (7.4% vs 7.6% averaged over pT/η/njet), so the improvement
comes from the added dimensions, not from replacing bins with trees.

Two caveats, both honest losses: the jet-DeepFlavB projection is *worse* than
the binned map (17.5% vs 12.3%) — neither method uses that axis any more, but
the binned map's `jetPt × |η|` cells apparently track the fake flavour
composition better than MUFFIN's variables do; and `|η|` and jet pT are each
about one point worse.

Uncertainty decomposition, inclusive, in the VR:

| component | how | value |
|---|---|---|
| component | how | MUFFIN | binned map |
|---|---|---|---|
| statistical | 20 Poisson-bootstrap retrainings / the map's bin errors treated as independent | 2.37% | 1.63% |
| statistical, coherent envelope | every replica / every cell moved the same way | 11.4% per bin | 15.28% |
| modelling | 7 hyper-parameter variations | 1.21% | not evaluated |
| background subtraction | ±10% on the subtracted genuine-τ simulation | 1.75% | not evaluated |
| non-closure + extrapolation | VR closure residual | 2.48% | 2.34% |

**MUFFIN does not reduce the uncertainty on the fake-τ normalisation** — its
statistical component is in fact slightly larger (2.37% vs 1.63%), which is what
one expects: the binned map has 54 cells to determine, MUFFIN a continuous
function in 7 variables, from the same 28k DR events. The gain is entirely in
the shape, and concentrated in τ_h pT (see the table above).

The coherent-envelope row is the convention the analysis actually applies
(`tauFR_weight_2d_up/_down` moves every cell the same way); it is conservative
for both methods. Per bin of τ_h pT, HT and jet1 pT the MUFFIN band is a flat
11.4% while the map's grows from 13.7% to 16.5% into the tails, where its cells
run out of events — `perbin_table.py` prints those tables.

Non-closure and extrapolation cannot be separated with a single validation
region: the VR differs from the MR by the very cut (`jet4DeepFlavB`) that is
deliberately not a feature.

### Dropping φ: tested, not adopted

φ carries no physics for a fake factor, so the φ-projection degradation looked
like it might be φ-as-an-input spending capacity on noise. Retrained without it
(`--features poster_nophi`, full chain), that is only a quarter true:

| | poster | no-φ | binned |
|---|---|---|---|
| `tau1Phi` | 6.74% | 5.20% | 4.57% |
| `tau1decayMode` | 2.10% | 2.49% | 8.11% |
| `tau1Mt` | 3.77% | 3.96% | 5.42% |
| mean over 234 variables | **5.56%** | 5.59% | 5.61% |
| mean over the 40 φ-type variables | 7.58% | 7.56% | 7.26% |

Dropping φ halves the `tau1Phi` degradation but pays it back on decay mode and
`tau1Mt`, leaves the φ-family average untouched, and is very slightly worse
overall. So φ is *not* the cause, and the poster list is kept verbatim. The
no-φ models and header are kept for reference.

## Conventions (identical to `measure_fr_flavour2d_v29pre.py`)

| | |
|---|---|
| base | `/eos/user/r/rtu/TurbOutputMC2017_v29pre_ak8_option92_2017` |
| pool | `kind_category_FR == 2 && trigSF_pfHT >= 300 && trigSF_caloHT >= 160` |
| DR (determination) | pool && `jet4DeepFlavB >= 0.1` — 27930 data events |
| AR (validation) | pool && `jet4DeepFlavB < 0.1` — 44158 data events |
| pass | `tau1idDeepTau2017v2p1VSjet >= 8` (analysis Loose WP) |
| fail | pool && `tau1idDeepTau2017v2p1VSjet < 8` |
| subtracted sim | `tau1genPartFlav == 5`, QCD excluded, ttbb overlap kill |
| weight | `W_HAD` with `triggerSF_perfilter_2nBtag_v24c × triggerLumiSF` |
| baseline map | `fr_flavour2d_1tau0l_all-mr-eta_nonj_2017.root` (x = `tau1jetPt`, y = `abs(tau1Eta)`, njet integrated) |

## Input features — the poster's list, verbatim

| poster | branch |
|---|---|
| 1. decay mode of τ_h | `tau1decayMode` |
| 2. ratio of seeding jet to τ_h pT | `tau1jetPt / tau1Pt` |
| 3. τ_h pT | `tau1Pt` |
| 4. number of jets | `nsmalljets` |
| 5. number of b-tagged jets | `nbtags` |
| 6. η, φ of τ_h | `tau1Eta`, `tau1Phi` |
| 7. era label | dropped — 2017 only |

`tau1jetPt` enters only through the ratio. The mother-jet DeepFlavB and QGL
axes are **not** used: the v29pre FR method dropped them.

Deliberately excluded: the τ ID discriminant itself (it defines pass/fail) and
`jet4DeepFlavB` (it defines DR vs AR — the DR has no support at AR values).

Measured ranking on the DR (total gain): `ptratio` ≫ `tau1decayMode` >
`tau1Eta` > `tau1Pt` > `tau1Phi` > `nsmalljets` > `nbtags`. Close to the
poster's, with the top two swapped.

Diagnostic-only sets (`--features`): `poster_abseta` (folded η — indistinguishable,
5.2% vs 5.1%), `poster_nophi` (φ dropped — also indistinguishable),
`poster_nobtag` (probes the DR→AR extrapolation), `binlike` (the baseline map's
own three variables).

## How the estimator works

A single binary classifier is trained on pass vs fail with **signed** weights
(+1 data, −w_analysis for genuine-τ simulation). The per-`z` stationary point of
that weighted log-loss is `s = A/(A+B)` with `A`, `B` the *net* densities, so

```
w_MUFFIN(z) = s/(1−s) = exp(raw margin)
```

exactly, with no normalisation constant needed — the signed weights already
carry the absolute yields.

Signed weights are essential rather than convenient: 96642 of the 153059 DR
training entries carry a negative generator weight (amcatnlo /
powheg-openloops), so the genuine-τ subtraction is a cancellation between two
large numbers. Splitting it into a separate density-ratio "subtraction step"
(the two-step scheme of arXiv:2511.06972) was tried first and is numerically
unstable at this sample size — it gave a 6% out-of-fold non-closure against
sub-percent here.

XGBoost refuses negative `DMatrix` weights, so they are applied in a custom
objective instead: exact gradient `w(s−y)`, positive surrogate Hessian
`|w|s(1−s)`. Damping the Hessian only rescales the Newton step — the stationary
point, where the signed gradients cancel, is unchanged.

The initial margin is `log(inclusive F_F)`, so the estimator falls back to the
inclusive fake factor where statistics are thin rather than to 1. One
calibration constant (`norm`, 1.007 for the nominal model) restores the DR
normalisation, which is the number the DR exists to fix.

### Choosing the capacity

`scan_muffin.py` judges configurations out-of-fold on the DR. Two numbers must
be read together: the integral closure alone is useless — a constant fake factor
scores exactly 1.000 on it while sitting at 10.8% on the differential metric.

Over `max_depth` 2→4 and 300→600 rounds the differential metric is flat at
5.2–5.5% while the integral out-of-fold closure drifts from 1.007 to 1.017, so
the nominal is the most conservative point of that plateau
(`max_depth=2, 300 rounds, min_child_weight=300`, out-of-fold 1.007). The spread
across the plateau is what the modelling variations quantify.

Capacity above the plateau overfits in a way that is easy to miss: an
overfit model drives `exp(margin) → 0` on the fail events, so the *in-sample*
closure falls below 1 and the normalisation constant then inflates every
out-of-fold prediction. Watch `raw` and `norm` in the scan, not just `oof`.

## Running

Everything needs the LCG view that provides xgboost + uproot + ROOT; `run.sh`
sources it:

```bash
bash muffin/run.sh prefetch.py DR AR          # cache the flat arrays (~2 min/region)
bash muffin/run.sh scan_muffin.py             # capacity scan, out-of-fold on the DR
bash muffin/run.sh compare_features.py        # feature sets vs the binned map
bash muffin/run.sh train_muffin.py --features poster --bootstrap 20 --variations --bkgsub 0.10
bash muffin/run.sh closure_muffin.py --features poster    # MR→VR closure + plots
bash muffin/run.sh export_muffin_cpp.py --tag poster --validate
```

`train_muffin.py --resume` skips the self-closure and any model already on disk,
so a long systematics run can be continued. `out/cache/` is gitignored and
stale after a re-production — delete it.

## What to look at, in order

1. **Out-of-fold DR self-closure** (`train_muffin.py`). Every event predicted by
   a model that never saw it, on the full DR statistics. This isolates the
   estimator from the MR→VR extrapolation: if it is not flat, the model is at
   fault, not the method. Currently 0.997–1.024 per njet × prong class, 1.007
   in total.
2. **MR→VR closure vs the binned map** (`closure_muffin.py`), the table above.
3. **Uncertainty decomposition**, printed by the same script.

## Full-variable closure plots, in the analysis' own format

`../closure_v29pre_muffin_1tau0l_20bin_split.py` is a copy of
`closure_v29pre_in_v29pre_option92_1tau0l_NHiggs_20bin_split.py` with one
addition: `USE_MUFFIN=1` builds the stacked fake-τ template from the MUFFIN
weight instead of the binned map, and the ratio pad then carries **both**
Data/Pred curves — MUFFIN as the black points, the binned map as a red line.
Same events, same MC-prompt stack, same anti-ID selection and prompt
subtraction, so the gap between the two curves is the difference between the
two fake factors and nothing else. Regions, samples, weights, variable list,
binning and styling are untouched.

```bash
bash muffin/run_closure_muffin.sh vr      # validation region, jet4DeepFlavB < 0.1
bash muffin/run_closure_muffin.sh incl    # inclusive (contains the training region)
bash muffin/run_closure_muffin.sh vr --variable ht        # one variable
bash muffin/run_closure_muffin.sh vr --chunk 0/8          # condor-style chunking
```

Output: `out/closure_muffin_{vr,incl}/`, 280 variables × {log, lin} × {png, pdf}
plus the histograms in a ROOT file (`*_ratio` = MUFFIN, `*_ratio_binned` = the
binned map, `*_faketau` / `*_faketau_binned` = the two templates).

The wrapper pins the settings that make the comparison meaningful:
`TAU_TIGHT_WP=loose` (the WP MUFFIN is trained at, anti-ID window [2, 8)) and
the current baseline map with `FR2D_YVAR=abseta`. The band on the MUFFIN
prediction is its own bootstrap spread (`muffin_weight_rms` in the exported
header), not the binned map's up/down.

Note the inclusive region contains the determination region, so it is a
consistency check rather than a test; the VR is the test.

`summarise_closure.py` ranks the variables by how much MUFFIN moves the closure,
reading both curves out of the ROOT file (bins with <25 data events are dropped —
they carry no information about the fake factor and otherwise dominate any
average):

```bash
bash muffin/run.sh summarise_closure.py out/closure_muffin_vr
```

Result over the 234 variables that survive the cut, yield-weighted
mean |1 − Data/Pred|:

| | VR | inclusive |
|---|---|---|
| MUFFIN | 5.56% | 4.07% |
| binned map | 5.61% | 4.19% |
| MUFFIN better in | 46% of variables | 50% |

So **on the flat average over all variables the two methods are equivalent** —
most of that list is jet-pair kinematics only weakly coupled to the fake factor,
where the differences are noise. The gains and losses are concentrated:

| improves (VR) | | degrades (VR) | |
|---|---|---|---|
| `tau1jetQGL` | +11.9% | `tau1jetDeepFlavB` | −6.3% |
| `tau1Pt` | +8.0% | `tau1Phi` | −2.2% |
| `tau1decayMode` | +6.0% | `massjet7jet8` | −1.9% |
| `ProbHHH4b2tau_*` (all trainings) | +1.4…+2.7% | φ-type jet-pair vars | −1% each |
| `tau1Mt` | +1.7% | | |

The improvement lands on the τ variables and — the part that matters for the
final fit — on every SPANet ProbHHH4b2tau training. The two losses are worth
knowing:

- **`tau1jetDeepFlavB` (−6.3%)** is the mother-jet flavour axis that the FR
  method dropped. Neither fake factor uses it, but the binned map's
  `jetPt × |η|` cells track it better than MUFFIN's variables do.
- **`tau1Phi` (−2.2%) and the φ-type jet-pair variables (−1% each)** point at
  the one poster feature that carries no physics for a fake factor. With 28k DR
  events φ is capacity spent on noise, and the DR scan found `poster_nophi`
  indistinguishable from `poster`. Retraining without φ is the obvious next
  step if these projections matter.

## Cross-channel: applying the 1tau0l fake factor to 2tau0l

`../closure_v29pre_muffin_2tau0l.py` + `run_closure_muffin_2tau0l.sh` do the same
thing for 2tau0l. MUFFIN is a per-object weight, so it drops straight into the
existing two-tau inclusion-exclusion template — evaluate it on τ₁ and τ₂ in turn
and keep

```
FAKE = Σ[tight₁ & anti₂]·FF₂ + Σ[anti₁ & tight₂]·FF₁ − Σ[anti₁ & anti₂]·FF₁·FF₂
```

unchanged. Both fake factors were measured in 1tau0l and are applied here as they
are, so this is a **cross-channel transfer test on equal terms**.

2tau0l holds ~110 tight-tight data events, so `|1 − Data/Pred|` mostly measures
the data Poisson noise; `CHI2=1 summarise_closure.py` reports χ²/ndf against the
data instead (same Poisson error for both methods, hence comparable):

| subset | n vars | χ²/ndf MUFFIN | χ²/ndf binned | D/P MUFFIN | D/P binned |
|---|---|---|---|---|---|
| all | 241 | 1.90 | **1.67** | 1.285 | **1.234** |
| well populated (≥8 filled bins) | 112 | 1.79 | 1.71 | **1.172** | 1.190 |
| sparse / zero-filtered | 129 | 2.00 | **1.63** | 1.382 | **1.273** |
| τ-related | 16 | **1.93** | 1.98 | **1.164** | 1.182 |

Per variable, on the τ sector MUFFIN transfers **better** — `tau1Pt` 0.94 vs
1.15, `tau2Pt` 1.03 vs 1.12, `tau1Eta` 1.28 vs 1.55, `tau2Eta` 1.69 vs 2.22,
`tau1decayMode` 1.12 vs 1.38 — and it is ~2% closer on the normalisation
everywhere. On non-τ variables it transfers worse: `ht` 3.72 vs 2.69, `jet1Pt`
0.66 vs 0.46, `higgs3_mass_manu` 1.70 vs 1.38.

**The global verdict (MUFFIN worse, better in only 31% of variables) is driven
entirely by the sparse subset** — jet7/jet8 and fatjet variables that exist only
in high-multiplicity events, where MUFFIN under-predicts by 38% against the map's
27%. That is exactly the failure mode to expect: MUFFIN takes `nsmalljets` and
`nbtags` as inputs and the map does not, so a topology with a second τ and a
different jet-multiplicity spectrum pushes it further from its training
distribution. Both methods under-predict 2tau0l by ~17% overall, which is a
property of the cross-channel transfer itself, not of either fake factor.

So: the multivariate gain survives the transfer **in the sector it was built for**
(the τ kinematics) and is paid for in the jet-multiplicity tails. If MUFFIN is to
be used in 2tau0l, the obvious thing to test first is retraining without
`nsmalljets`/`nbtags`, or measuring it in-channel.

## Using it in the analysis

`export_muffin_cpp.py` writes a dependency-free header with a `muffin_weight()`
function, so the existing RDataFrame chain uses it the way it uses
`declare_fr2d()` — no friend trees, no python in the event loop. The exported
header is validated against the python model at export time (currently agreeing
to 3.4e-6 relative, i.e. float32 accumulation noise).

```python
ROOT.gInterpreter.Declare(open('muffin/out/muffin_poster.h').read())
df = df.Define('muffin_weight',
               'muffin_weight(tau1decayMode, (tau1jetPt/std::max(tau1Pt,1.e-6f)), '
               'tau1Pt, nsmalljets, nbtags, tau1Eta, tau1Phi)')
```

`muffin_weight` replaces `tauFR_weight_2d` one-for-one: it is already
`FR/(1−FR)`, not `FR`.

## Files

| | |
|---|---|
| `muffin_common.py` | conventions, feature sets, region loading + cache |
| `muffin_model.py` | the estimator (signed-weight custom objective) |
| `train_muffin.py` | training + bootstrap / modelling / background-subtraction replicas |
| `closure_muffin.py` | MR→VR closure, plots, uncertainty decomposition |
| `compare_features.py` | feature-set comparison against the binned map |
| `scan_muffin.py` | capacity scan |
| `export_muffin_cpp.py` | standalone C++ header + validation |
| `prefetch.py`, `run.sh` | cache filling, LCG wrapper |
| `check_negw.py`, `debug_export.py` | the two investigations above, kept reproducible |
