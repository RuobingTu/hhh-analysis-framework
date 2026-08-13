# MUFFIN — the jet→τ_h fake-tau estimate

Reference for the fake-tau method used in HHH→4b2τ. MUFFIN (MUltivariate
Fake-Factor INference; I. Andreou, D. Colling, D. Winterbottom, LHCP poster,
CMS-PAS-TAU-25-001) replaces the binned fake factor with a continuous,
multi-dimensional per-object weight:

```
w_MUFFIN(z) = [p_pass^data(z) − p_pass^sim(z)] / [p_fail^data(z) − p_fail^sim(z)]
```

**This is the same quantity the binned map delivers as `FR/(1−FR)`.** The
denominator is inclusive-loose (tight included), so
`FR/(1−FR) = N_pass/(N_loose−N_pass) = N_pass/N_fail`. Only the estimation
changes: continuous in seven variables instead of `p_T × |η|` cells per prong.
`muffin_weight` therefore replaces `tauFR_weight_2d` one-for-one — it is already
`FR/(1−FR)`, not `FR`.

Status: adopted as the fake-tau method as of v29pre (2017). Validated in
1τ_h0l, 1τ_h1l and 2τ_h0l; see [Validation](#validation).

---

## 1. How the estimator works

A single binary classifier is trained on pass vs fail with **signed** weights
(+1 data, −w_analysis for genuine-τ simulation). The per-`z` stationary point of
that weighted log-loss is `s = A/(A+B)` with `A`, `B` the *net* densities, so

```
w_MUFFIN(z) = s/(1−s) = exp(raw margin)
```

exactly — no normalisation constant, because the signed weights already carry
the absolute yields. The raw margin is used rather than `s/(1−s)`: for a
logistic objective the margin *is* the log density ratio, and it is safer in the
saturated tails.

**Signed weights are essential, not a convenience.** 96642 of the 153059 DR
training entries carry a negative generator weight (amcatnlo / powheg-openloops),
so the genuine-τ subtraction is a cancellation between two large numbers.
Splitting it into a separate density-ratio "subtraction step" (the two-step
scheme of arXiv:2511.06972) was tried first and is numerically unstable at this
sample size — 6% out-of-fold non-closure against sub-percent here.

XGBoost refuses negative `DMatrix` weights, so they enter through a custom
objective: exact gradient `w(s−y)`, positive surrogate Hessian `|w|s(1−s)`.
Damping the Hessian only rescales the Newton step — the stationary point, where
the signed gradients cancel, is unchanged, and `min_child_weight` keeps acting
on a positive quantity.

The initial margin is `log(inclusive F_F)`, so the estimator falls back to the
inclusive fake factor where statistics are thin rather than to 1. One
calibration constant (`norm`, 1.007 for the nominal 1τ_h0l model) restores the
DR normalisation: a converged model closes by construction, but a regularised
model lives in log space and `exp()` of a shrunk margin is biased low by Jensen.

### Choosing the capacity

`scan_muffin.py` judges configurations out-of-fold. **The integral closure alone
is useless** — a constant fake factor scores exactly 1.000 on it while sitting at
10.8% on the differential metric. Over `max_depth` 2→4 and 300→600 rounds the
differential metric is flat at 5.2–5.5% while the integral out-of-fold closure
drifts from 1.007 to 1.017, so the nominal is the most conservative point of
that plateau: `max_depth=2, 300 rounds, min_child_weight=300`.

Capacity above the plateau overfits in a way that is easy to miss: an overfit
model drives `exp(margin) → 0` on the fail events, so the *in-sample* closure
falls below 1 and the normalisation constant then inflates every out-of-fold
prediction. **Watch `raw` and `norm` in the scan, not just `oof`.**

---

## 2. Input features — the poster's list, verbatim

| poster | branch |
|---|---|
| 1. decay mode of τ_h | `tau1decayMode` |
| 2. ratio of seeding jet to τ_h pT | `tau1jetPt / tau1Pt` |
| 3. τ_h pT | `tau1Pt` |
| 4. number of jets | `nsmalljets` |
| 5. number of b-tagged jets | `nbtags` |
| 6. η, φ of τ_h | `tau1Eta`, `tau1Phi` |
| 7. era label | dropped — one year at a time |

`tau1jetPt` enters only through the ratio. The mother-jet DeepFlavB and QGL axes
are **not** used: the v29pre FR method dropped them.

Deliberately excluded: the τ ID discriminant itself (it defines pass/fail) and
the DR/AR-defining b-tag variable (the DR has no support at AR values).

Measured ranking (total gain), consistent across 1τ_h0l and 1τ_h1l:
`ptratio` ≫ `tau1decayMode` > `tau1Eta` > `tau1Pt` > `tau1Phi` > `nsmalljets` >
`nbtags`.

---

## 3. Per-channel recipes

### 1τ_h0l — measured in its own MR

| | |
|---|---|
| pool | `kind_category_FR == 2 && trigSF_pfHT >= 300 && trigSF_caloHT >= 160` |
| DR (measure) | pool && `jet4DeepFlavB >= 0.1` — 27930 data events |
| AR (validate) | pool && `jet4DeepFlavB < 0.1` — 44158 data events |
| pass / fail | `tau1idDeepTau2017v2p1VSjet >= 8` / pool && `< 8` |
| subtracted sim | `tau1genPartFlav == 5`, QCD excluded, ttbb overlap kill |
| weight | `W_HAD` with `triggerSF_perfilter_2nBtag_v24c × triggerLumiSF` |

```bash
bash muffin/run.sh train_muffin.py --features poster --bootstrap 20 --variations --bkgsub 0.10
```

### 1τ_h1l — measured in-channel

Following `taufr_1tau1l_measure.py`: pool `kind_category_FR == 1` with a tight
lepton, SingleMuon and SingleElectron each restricted to their own flavour,
genuine taus subtracted with the 1τ_h1l weight chain
(`triggerSF_1tau1l_v1 × Muon1IdSF × Ele1IdSF`).

The DR is the channel's own **mr60** (`jet4DeepFlavB < 0.035`, the balanced
60/40 split of the fake FO pool), *not* the whole pool. This matters: an
inclusive in-channel fit reproduces the tight region by construction and the
agreement plot then shows nothing. 27060 data events in the MR.

```bash
bash muffin/run.sh train_muffin_1tau1l.py --bootstrap 20
bash muffin/run.sh export_muffin_cpp.py --tag 1tau1l --bootstrap 20 --validate
bash muffin/run_agreement_muffin_1tau1l.sh          # MUFFIN_HEADER=out/muffin_1tau1l.h
```

**Measure in-channel rather than carrying the 1τ_h0l one over**: the inclusive
fake factor is 0.3357 (MR) against 0.3393 (VR), 1.1% apart, while the
1τ_h0l → 1τ_h1l cross-channel difference is 4.5%.

### 2τ_h0l — the 1τ_h0l fake factor, applied per object

MUFFIN is a per-object weight, so it drops into the existing two-tau
inclusion-exclusion template unchanged — evaluate it on τ₁ and τ₂ in turn:

```
FAKE = Σ[tight₁ & anti₂]·FF₂ + Σ[anti₁ & tight₂]·FF₁ − Σ[anti₁ & anti₂]·FF₁·FF₂
```

**Structural fact worth knowing:** the two taus are ordered by their raw DeepTau
VSjet score (verified: `tau1 > tau2` in 100% of the pool), so `(anti₁, tight₂)`
is empty by construction and the B term vanishes. The formula stays correct —
the A term absorbs both TL and LT — but the "two-tau" template is effectively
single-legged.

```bash
bash muffin/run_closure_muffin_2tau0l.sh              # (tight,tight)
SIDEBAND=1 bash muffin/run_closure_muffin_2tau0l.sh   # single-tight sideband
```

---

## 4. Validation

### 4.1 Against the binned map, 1τ_h0l MR→VR

Poster figure-2 metric `sqrt((1−r)² + σ_r²)`, yield-weighted:

| projection | MUFFIN | binned map |
|---|---|---|
| **τ_h pT** | **5.9%** | **13.8%** |
| seeding jet pT | 7.2% | 6.9% |
| number of jets | 4.5% | 4.4% |
| \|η(τ_h)\| | 6.2% | 5.3% |
| MET | 5.6% | 5.4% |
| seeding jet DeepFlavB | 17.5% | 12.3% |

Integral VR closure: MUFFIN 0.975, binned 0.977 — the ~2.5% non-closure is
shared, so it belongs to the MR→VR extrapolation, not to either fake factor.

Feeding the BDT **only** the binned map's own variables reproduces the binned
map (7.4% vs 7.6%), so the gain comes from the added dimensions, not from
replacing bins with trees.

### 4.2 Over all 234 analysis variables

`summarise_closure.py`, yield-weighted mean |1 − Data/Pred|, bins with <25 data
events dropped:

| | VR | inclusive |
|---|---|---|
| MUFFIN | 5.56% | 4.07% |
| binned map | 5.61% | 4.19% |

**On the flat average the two are equivalent** — most of that list is jet-pair
kinematics only weakly coupled to the fake factor. The differences concentrate:

| improves | | degrades | |
|---|---|---|---|
| `tau1jetQGL` | +11.9% | `tau1jetDeepFlavB` | −6.3% |
| `tau1Pt` | +8.0% | `tau1Phi` | −2.2% |
| `tau1decayMode` | +6.0% | φ-type jet-pair vars | −1% each |
| `ProbHHH4b2tau_*` (every training) | +1.4…+2.7% | | |

The improvement lands on the τ variables and on every SPANet discriminant —
which is what matters for the final fit.

### 4.3 Total yields

Both fake factors normalise to the same DR, so the totals are locked together
by construction and only the shape can differ (`total_yields.py`):

| | data | MUFFIN | binned | D/P MUFFIN | D/P binned |
|---|---|---|---|---|---|
| 1τ_h0l VR | 10471 | 10285.1 | 10283.9 | 1.018 | 1.018 |
| 1τ_h0l inclusive | 17042 | 16879.6 | 16866.1 | 1.010 | 1.010 |
| 2τ_h0l | 111 | 95.5 | 94.1 | 1.163 ± 0.039 | 1.180 ± 0.054 |

### 4.4 Uncertainty

| component | how | MUFFIN | binned map |
|---|---|---|---|
| statistical | 20 bootstrap retrainings / cells treated as independent | 2.37% | 1.63% |
| statistical, coherent envelope | every replica / cell moved together | 11.4% per bin | 15.28% |
| modelling | 7 hyper-parameter variations | 1.21% | not evaluated |
| background subtraction | ±10% on the subtracted genuine-τ MC | 1.75% | not evaluated |
| non-closure + extrapolation | VR closure residual | 2.48% | 2.34% |

**MUFFIN does not reduce the uncertainty on the fake-τ normalisation** — its
statistical component is slightly larger, which is expected: the map has 54
cells to determine, MUFFIN a continuous function in seven variables, from the
same 28k DR events. The gain is in the shape.

Per bin of τ_h pT / HT / jet1 pT the MUFFIN band is a flat **11.4%** while the
map's grows 13.7% → **16.5%** into the tails, where its cells run out of events
(`perbin_table.py`).

### 4.5 Cross-channel transfer, validated on ttbar MC

`ttbar_transfer.py` — pure MC truth (keep `tau1genPartFlav != 5`), so no
data-driven subtraction; 724k / 270k fake τ_h.

| | integral | τ pT | jet pT | \|η\| | njet | MET |
|---|---|---|---|---|---|---|
| **1τ_h1l → 1τ_h0l** | | | | | | |
| MUFFIN | **1.018** | **1.8%** | 3.3% | **2.6%** | **1.9%** | **3.5%** |
| binned map | 1.031 | 14.8% | 3.0% | 3.4% | 4.1% | 5.0% |
| constant F_F | 1.046 | 8.8% | 16.9% | 4.9% | 4.6% | 6.4% |
| **1τ_h0l → 1τ_h0l** (out-of-fold reference) | | | | | | |
| MUFFIN | 1.000 | 0.5% | 4.1% | 5.2% | 0.6% | 2.4% |
| binned map | 1.000 | 16.2% | 0.5% | 1.7% | 3.2% | 2.6% |

The inclusive ttbar fake factor differs by 4.5% between the channels
(0.3355 vs 0.3209) and a differential fake factor absorbs most of it.
**MUFFIN transfers better than the binned map**, because the channels differ
largely in njet/pT and it is differential in those.

### 4.6 2τ_h0l — the single-tight sideband

The `(tight,tight)` region holds 111 data events (9.5% Poisson), which cannot
separate 1.00 from 1.17. The single-tight sideband holds 526 and the fake factor
predicts it just as directly: under score-ordering, promoting either tau of an
`(anti,anti)` event gives an exactly-one-tight event, so

```
pred N(1T) = Σ over (anti,anti) of [F(τ₁) + F(τ₂)]
```

which is exact for independent per-object pass probabilities.

| test | MUFFIN | binned map | events |
|---|---|---|---|
| single-tight sideband | **0.998 ± 0.056** | 0.980 ± 0.055 | 526 |
| (tight,tight) | 0.842 | 0.828 | 111 |

**The per-object fake factor transfers into 2τ_h0l correctly.** The
`(tight,tight)` shortfall is Poisson p = 0.061, i.e. **1.5σ — not established**.
Both tests share the fake factor, the events and the `(anti,anti)` input, so if
that shortfall were real it could not come from the fake factor.

---

## 5. Studies done and their outcomes

Recorded so they are not repeated. **All five attempts to explain the 2τ_h0l
`(tight,tight)` shortfall failed**, which is itself the answer to give a
reviewer: it is not that we did not look.

| study | script | outcome |
|---|---|---|
| two-step subtraction (arXiv:2511.06972) | — | **rejected**: unstable, 6% out-of-fold non-closure |
| drop φ from the inputs | `--features poster_nophi` | **not adopted**: halves the `tau1Phi` degradation but pays it back elsewhere; overall 5.56% → 5.59% |
| ttbar/QCD source split | `source_split.py` | **mechanism real, insufficient**: ttbar fraction 13.2% → 28.8%, F_ttbar 24% above F_QCD, explains 3.9 of the 7.2% channel difference |
| composition-induced correlation | analytic | **~1%**, and the wrong sign |
| score-ordering anti-correlation | `order_bias_mc.py` | **not supported** by ttbar MC |
| jet bookkeeping (FR_AST) in 2τ_h0l | `njet_veto_check.py` | **excluded at 13σ**: Δ⟨njet⟩ = +0.165 ± 0.138 against +2 expected — the tau veto removes both channel taus regardless of ID |
| njet correction from the sideband | `njet_correction.py` | **does not fix it**: k = 1.50 ± 0.19 at njet=4 only, and applying it moves (tight,tight) 1.166 → 1.136 |

Settling the 2τ_h0l question needs **statistics, not another mechanism**: full
Run 2 would take 111 events to ~370, i.e. 9.5% → 5.2%.

---

## 6. Applying this to a new year

Everything below is year-dependent and must be redone per year:

1. **Trigger SF** — the weight chain (`triggerSF_perfilter_*`, `triggerLumiSF`,
   `triggerSF_1tau1l_*`) enters both the subtraction and the application. Update
   `mc_weight()` in `muffin_common.py` and the per-channel loaders.
2. **`BASE`** in `muffin_common.py` → that year's production.
3. **`LUMI`** in `muffin_common.py`.
4. **Retrain from scratch** — the fake factor is not transferable between years:
   ```bash
   rm -rf muffin/out/cache                     # the flat arrays are year-specific
   bash muffin/run.sh train_muffin.py --features poster --bootstrap 20 --variations --bkgsub 0.10
   bash muffin/run.sh closure_muffin.py --features poster
   bash muffin/run.sh export_muffin_cpp.py --tag poster --bootstrap 20 --validate
   ```
5. **Re-check the capacity** with `scan_muffin.py` if the DR statistics change
   materially — the nominal was chosen for ~28k DR events.
6. **Re-derive the 1τ_h1l in-channel model** (`train_muffin_1tau1l.py`) and
   confirm the mr60 split is still balanced for that year.
7. **Re-run the validations**: MR→VR closure, the full-variable comparison, and
   the 2τ_h0l sideband.

The DR/AR cut values (`J4CUT = 0.1` for 1τ_h0l, `0.035` for 1τ_h1l) were tuned
on 2017 and should be re-checked, not assumed.

---

## 7. Running

Everything needs the LCG view that provides xgboost + uproot + ROOT; `run.sh`
sources it.

```bash
bash muffin/run.sh prefetch.py DR AR                # cache the flat arrays (~2 min/region)
bash muffin/run.sh scan_muffin.py                   # capacity scan, out-of-fold
bash muffin/run.sh compare_features.py              # feature sets vs the binned map
bash muffin/run.sh train_muffin.py --features poster --bootstrap 20 --variations --bkgsub 0.10
bash muffin/run.sh closure_muffin.py --features poster
bash muffin/run.sh export_muffin_cpp.py --tag poster --bootstrap 20 --validate

# full-variable closure plots, in the analysis' own format, both ratios
bash muffin/run_closure_muffin.sh vr | incl
SIDEBAND=1 bash muffin/run_closure_muffin_2tau0l.sh
bash muffin/run_agreement_muffin_1tau1l.sh

# summaries
bash muffin/run.sh summarise_closure.py out/closure_muffin_vr
CHI2=1 bash muffin/run.sh summarise_closure.py out/closure_muffin_2tau0l
bash muffin/run.sh total_yields.py
bash muffin/run.sh perbin_table.py out/perbin_vr tau1Pt ht jet1Pt
bash muffin/run.sh convener_checks_2tau0l.py
```

`train_muffin.py --resume` skips the self-closure and any model already on disk.
`out/cache/` is gitignored and **stale after a re-production — delete it**.

### Using it in the analysis

`export_muffin_cpp.py` writes a dependency-free header with a `muffin_weight()`
function, validated against the python model at export time (3e-6), so the
RDataFrame chain uses it the way it uses `declare_fr2d()`:

```python
ROOT.gInterpreter.Declare(open('muffin/out/muffin_poster.h').read())
df = df.Define('muffin_weight',
               'muffin_weight(tau1decayMode, (tau1jetPt/std::max(tau1Pt,1.e-6f)), '
               'tau1Pt, nsmalljets, nbtags, tau1Eta, tau1Phi)')
```

The header also carries the 20 bootstrap replicas and `muffin_weight_rms()`,
which returns their per-event RMS — that is the band on the closure plots.

Two threshold pitfalls, already fixed in the exporter but worth knowing if it is
ever rewritten: split values must come from the JSON dump, not
`trees_to_dataframe()` (which rounds them, and the hist builder puts thresholds
exactly on integer feature values); and C++ float literals need a decimal point
before the `f` suffix.

---

## 8. Files

| | |
|---|---|
| `muffin_common.py` | conventions, feature sets, region loading + cache |
| `muffin_model.py` | the estimator (signed-weight custom objective) |
| `train_muffin.py` | 1τ_h0l training + bootstrap / modelling / bkg-subtraction replicas |
| `train_muffin_1tau1l.py` | 1τ_h1l in-channel training (mr60) |
| `closure_muffin.py` | MR→VR closure, plots, uncertainty decomposition |
| `compare_features.py`, `scan_muffin.py` | feature-set and capacity scans |
| `export_muffin_cpp.py` | standalone C++ header + validation |
| `summarise_closure.py`, `total_yields.py`, `perbin_table.py` | summaries |
| `ttbar_transfer.py` | cross-channel transfer on ttbar MC |
| `source_split.py` | ttbar vs QCD fake factors and compositions |
| `sideband_2tau0l.py`, `convener_checks_2tau0l.py` | 2τ_h0l validation |
| `njet_veto_check.py`, `njet_correction.py`, `order_bias_mc.py` | the 2τ_h0l investigations |
| `../closure_v29pre_muffin_1tau0l_20bin_split.py` | 1τ_h0l plotter, `USE_MUFFIN=1` |
| `../closure_v29pre_muffin_2tau0l.py` | 2τ_h0l plotter, `USE_MUFFIN=1`, `SIDEBAND=1` |
| `../agreement_v29pre_muffin_1tau1l.py` | 1τ_h1l plotter, `USE_MUFFIN=1` |
| `beamer/` | the group-meeting talk (tex + figures) |

### Plot directories (main checkout)

| | |
|---|---|
| `plots_v29pre_1tau0l_muffin_inclusive/` | 1τ_h0l, 279 variables, both ratios |
| `plots_v29pre_1tau0l_muffin_in_VR/` | the same in the VR |
| `plots_v29pre_2tau0l_muffin_vs_map/` | 2τ_h0l (tight,tight) |
| `plots_v29pre_2tau0l_muffin_sideband/` | 2τ_h0l single-tight sideband |
| `plots_v29pre_2tau0l_convener_checks/` | the validation package |
| `plots_v29pre_1tau0l_muffin_nophi_in_VR/` | the no-φ test |
