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
| statistical | 20 Poisson-bootstrap replicas of the training set | 2.37% |
| modelling | 7 hyper-parameter variations | 1.21% |
| background subtraction | ±10% on the subtracted genuine-τ simulation | 1.75% |
| non-closure + extrapolation | VR closure residual | 2.48% |
| **total** | quadrature | **4.03%** |

Non-closure and extrapolation cannot be separated with a single validation
region: the VR differs from the MR by the very cut (`jet4DeepFlavB`) that is
deliberately not a feature.

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
