# hier_hmm

A two-level hidden Markov model that segments **single molecules** of methylation-footprinting
data (Fiber-seq / SMAC / dSMF style m6A calls) into chromatin states, one molecule at a time.

* **Level 1** — open / linker / nucleosome, over the whole molecule.
* **Level 2** — short protein footprints, searched only *inside* Level-1 open runs, at base-pair
  resolution.

Two things make it different from a stock HMM:

1. **Durations are priors, not geometric accidents.** Every state is a phase-type (Erlang) chain,
   so a nucleosome has a mean length of 129 bp with a hard 60 bp floor and an SD of ~21 bp,
   instead of the "most likely length is 1 bp" that a plain HMM self-loop gives you. Open and
   linker are separated *by their duration priors and the grammar*, not by their signal — they
   share one emission rate by default.
2. **The emission is a masked binomial on counts,** not a Gaussian on a smoothed rate. A bin with
   one callable adenine and no methylation is weak evidence; a bin with five is five times
   stronger; a bin with none is simply uninformative and the duration prior carries it. Callable
   density therefore sets how *confident* a bin may be without biasing *which* state it prefers.

Every prior, threshold and bin size lives in one commented YAML file. The code holds no second
copy of any of them — if a number is not in the config, the model does not use it.

## Install

```bash
pip install -e .            # numpy, pyyaml, matplotlib
pip install -e ".[test]"    # + pytest
```

## Input

One `(n_fiber, n_pos)` float array, `m6a`:

| value | meaning |
| --- | --- |
| `NaN` | not callable here — the read does not cover this position, or the base is not assayable |
| `0` | callable, unmethylated |
| `1` | callable, methylated |

Optionally, in the same `.npz`:

* `coverage` `(n_fiber, n_pos)` bool — read coverage as distinct from callability. Only used to
  decide where each molecule's usable span begins and ends. Without it the span is inferred from
  the first and last callable position, which is fine when callable sites are dense.
* `positions` `(n_pos,)` and `center` — genomic coordinates for the plot axis.
* **any other length-`n_fiber` array** — group labels, cluster ids, read names. These ride
  through to the output and can be used to split the plot (`--group-by`) or order its rows
  (`plot.order_by`).

## Use

```bash
hier-hmm init-config my.yaml           # a commented copy of every knob
hier-hmm check -c my.yaml              # what duration priors does this config actually imply?
hier-hmm run data.npz -c my.yaml -o out/ --group-by celltype
hier-hmm run data.npz --set level1.tau=0.7 --set binning.level1_bp=10   # one-off overrides
hier-hmm simulate synthetic.npz        # a dataset from the model's own priors, to try it on
```

`run` writes `annotation.npz` (labels, per-bp footprint posterior, per-fiber metadata),
`annotation.calibration.json` (the fitted rates and every refinement step),
`annotation.config.yaml` (the fully resolved config — the run is reproducible from its own
output) and `annotation.png`.

As a library:

```python
from hier_hmm import load_config, prepare, run

cfg = load_config("my.yaml")
res = run(prepare(m6a_array, cfg), cfg)
res.labels          # (n_fiber, n_bp) int8; -1 = no data
res.label_map       # {'open': 0, 'linker': 1, 'nucleosome': 2, 'footprint': 3}
res.rates           # calibrated methylation rate per emission class
frac, cov = res.fraction("open", "footprint")   # aggregate accessibility profile
```

Scoring the Level-2 calls against a permutation null (see *Calibration* below):

```python
from hier_hmm import HierHMM
from hier_hmm.calibrate import calibrate
from hier_hmm.metrics import concordance

model = HierHMM(cfg)
rates, _ = calibrate(model, ds)
concordance(model, ds, rates, n_perm=4, group_key="celltype")
# {'excess_coh': 0.054, 'excess_top': 0.079, 'raw_coh': 0.063, 'n_obs': 4.4, ...}
```

## The config

`hier_hmm/default_config.yaml` is the whole model. Its four sections:

* **`binning`** — bp per Level-1 bin (5) and per Level-2 bin (1). Level 2 runs finer because a
  25 bp footprint quantised to 5 bp carries 20% error at each edge, while a 147 bp nucleosome
  does not care. **A bin-size change is a model change**: `k` is fixed while the per-stage
  advance probability scales with the bin, so dwell *means* are invariant but dwell *variances*
  are not (nucleosome SD 27 bp at 1 bp bins → 21 bp at 5 bp → 10 bp at 10 bp). `hier-hmm check`
  prints what you actually got.
* **`level1` / `level2`** — per state a mean length, an Erlang `k`, an optional hard minimum, and
  an emission class; plus the grammar (which transitions are allowed and with what weight — the
  zeros are hard), the start distribution, and the decoder.
* **`emission`** — one methylation rate per class, and which decoded states each class is
  re-estimated from.
* **`data` / `output` / `plot`** — acceptance thresholds, what to write, colours and ordering.

Any key you pass that does not already exist in the defaults is an **error**, so a misspelled
setting fails loudly instead of being silently ignored.

### Decoding and `tau`

Both levels default to forward-backward posterior decoding: a position is called by the
threshold state (`open` at Level 1, `footprint` at Level 2) when its posterior clears `tau`, and
otherwise takes the argmax of the remaining states. `tau` is the operating point — sweeping it
turns a single fixed call into a precision/recall curve. `decode: viterbi` gives the single best
path instead, which discards that confidence and, in our footprint tuning, scored *worse* on how
positionally concentrated the calls were.

### Calibration, and how many refinements

Rates are not fitted jointly with the state path. They are initialised by a two-component
binomial mixture over all pooled bins, then re-estimated from the positions the current
segmentation assigns to each class — annotate, re-estimate, repeat (`refine_iters`, default 2).

This loop is self-referential — each pass re-estimates a rate from the calls the previous rates
produced — so `refine_iters` was set by measuring the calls, not by taste. `hier_hmm.metrics`
scores Level-2 calls as **excess over a permutation null**: Level 1 is frozen, each molecule's
methylated positions are re-placed at random within its own open runs (count preserved), and the
null is called with the same rates. Raw coherence is not a valid target — a shared positional
prior raises it by construction — and the two excess metrics can disagree, in which case
`excess_top` (share of call mass in the 2% most recurrent positions) is the one to trust because
it is density-invariant while coherence rises mechanically with the number of calls.

On three T-cell loci × two cell types, 4 nulls (`dhs_analysis/tcell_refine_concordance.py` in
the parent study repo):

| refinement pass `k` | 0 | 1 | 2 | 3 | 4 | 5 |
| --- | --- | --- | --- | --- | --- | --- |
| `excess_top` | **−0.0174** | +0.0551 | **+0.0792** | +0.0639 | +0.0600 | +0.0613 |
| `excess_coh` | +0.0585 | +0.0542 | +0.0538 | +0.0511 | +0.0502 | +0.0505 |
| calls / molecule (obs / null) | 2.46 / 0.92 | 3.74 / 1.99 | 4.44 / 2.70 | 4.77 / 2.98 | 4.81 / 2.97 | 4.81 / 2.97 |

With no refinement at all the footprint calls are **less** positionally concentrated than the
null: the mixture rates alone do not produce footprints that agree across molecules. Two passes
is the peak, and it decays slowly after. `excess_coh` is monotone decreasing and misleading here
— at `k=0` the null makes 2.7× fewer calls than the observation, so it simply has less
opportunity to accumulate coherence.

Isolating the footprint rate (Level-1 classes held at `k=2`, footprint rate taken from pass
`k`) shows the decay after `k=2` is **not** the footprint rate's doing:

| footprint rate from pass | 0 (0.041) | 1 (0.067) | 2 (0.083) | 3 (0.087) | 4 (0.085) | 5 (0.082) |
| --- | --- | --- | --- | --- | --- | --- |
| `excess_top` | +0.0611 | +0.0774 | +0.0792 | +0.0801 | +0.0794 | +0.0791 |

It settles by pass 2 and is flat to ±0.001 thereafter. What keeps sliding is `accessible`
(0.629 → 0.578) and `protected` (0.038 → 0.026), and concordance decays with them. So the
footprint rate is the *stable* estimate here, not the runaway one, and capping it early costs a
little (`+0.0774` at one pass against `+0.0792` at two).

Each emission class nevertheless accepts `max_refine: N` to freeze it after `N` passes, which is
the knob to reach for if you raise `refine_iters` and want the Level-1 rates pinned. Note that
the direction of the footprint rate's drift is dataset-dependent: on the synthetic data in
`examples/`, where footprints are sparse and cleanly separated, it drifts *down* instead
(0.036 → 0.016 → 0.006 against a true 0.020). Measure it on your data before trusting the
calibrated footprint rate as an estimate of anything physical.

## Where the defaults come from

The numbers in `default_config.yaml` are the settings from a Fiber-seq study of mouse T-cell
loci, tuned there and carried over verbatim. Three that are worth knowing about:

* **`level1.states.linker.emission_class: accessible`** — open and linker share one rate. Giving
  them their own (measured at ~0.73 vs ~0.57) looks justified and is strictly dominated: the
  lower linker rate is an artefact of linkers being *short* (edge blur), not of being less
  accessible — trimming 20 bp off each run collapses the gap to 0.018 — and untying feeds that
  artefact back through calibration in a self-confirming loop.
* **`level2.min_call_bp: 8`** — the state's `min_bp` constrains the hidden dwell, but
  thresholding the posterior clips the tapering edges of a call, so decoded runs come out shorter
  than the state that generated them (~8-10% landed under 8 bp with the dwell floor alone). The
  extra floor costs some cross-fiber coherence but nearly doubles positional concentration, and
  concentration is the density-invariant metric of the two.
* **`level1.start: uniform`** — "enter at any dwell stage of any state". Starting the chain in
  `open` is only right if every window is centred on something open by construction; on
  read-start data it puts a spurious open/footprint stripe at the beginning of every molecule.

These were tuned on one dataset. Re-tune them on yours; that is what the config is for.

## Tests

```bash
pytest                       # or: python tests/test_hmm.py   (each file runs standalone)
```

They cover the parts where a silent error would be invisible in the output: dwell means and
variances against what the config asked for, Viterbi and forward-backward against brute-force
enumeration of every state path on small chains, config validation, and end-to-end recovery of a
path simulated from the model's own priors (the sampler walks the same chains the annotator
builds, so a disagreement is a real bug).

## Limitations

* Molecules are treated as **independent**. There is no coupling across molecules and no shared
  positional field, so nothing pools evidence across the ensemble at a locus.
* It is a **segmenter, not a generative model of an ensemble**. The states are phenomenological:
  a "nucleosome" is a protected run of about the right length, not a rod with an excluded volume.
* It is **sequence-blind**, and its states are a renewal process — one state's position carries
  no information about the next beyond the grammar and the durations.
* Calibration is EM-like but not a likelihood ascent with a convergence guarantee; it is two
  passes of re-estimation from the current segmentation.

Treat its calls as good proposals, not ground truth.

## License

MIT.
