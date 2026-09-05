# Stage 2 (EXAMPLE) – export to a spec

> **This is an example.** It shows the shape of an export stage, not a
> methodology to adopt.

Stage 2 is a serialization step. **It introduces no new figures.** If something
is missing, the fix is in Stage 1, not here.

## 1. The shape

```yaml
schema_version: "1.0"
name: "<YYYY-MM-DD>-<TICKER>"
valuation: <adapter>
price: <Stage 1 price>
seed: 0
n: 50000

factors: [rate, demand]

drivers:
  some_driver:
    marginal: {family: normal, p10: 1.0, p50: 3.0, p90: 5.0}
    loadings: {rate: -0.35, demand: 0.70}

constants:
  some_constant: 42.0
```

Sort each Stage 1 triple ascending into `p10` / `p50` / `p90`. Base lands at
`p50`. If it does not, that is a Stage 1 defect - fix it there, do not reorder
here.

## 2. Any key may be a driver or a constant

`hydrate()` merges drivers and constants into one flat dict, so the adapter
cannot tell which block a name arrived from. That is a feature, and it is the
decision this stage exists to make:

> **A constant is a claim that you know the number.** If Stage 1 had to range it,
> freezing it does not remove the uncertainty - it hides it, and the tornado
> cannot show a reader what it never sampled.

**Measure, do not judge.** Run it both ways and compare `Model.driver_swings()`:

```python
from valdist import load_spec, hydrate

model = hydrate(load_spec("path/to/spec.yaml"))
for row in sorted(model.driver_swings(), key=lambda d: -abs(d["swing"])):
    print(f"{row['name']:20} {row['swing']:8.2f}")
```

If freezing it would move the blend more than your weakest sampled driver does,
promote it. Estimating that swing by eye is how it goes wrong: this repo has a
recorded case where a frozen constant was reasoned to be worth about a fortieth
of the weakest driver and measured at more than ten times that.

## 3. Choosing a marginal family

Pick the family that reproduces the triple you typed. `valdist validate` tells
you when it does not:

```text
NOTICE - [quantile_mismatch] driver 'x': lognormal fit's p50 (3) is far from the
geometric mean of p10/p90 (1/5) - the realized p10/p90 are ~1.342/~6.708
```

That spec claims a p90 of 5 and samples one of 6.7. Work it in this order:

1. **Roughly symmetric** triple (p50 near the arithmetic midpoint)? Use `normal`.
2. **Log-symmetric** (p50 near the geometric mean)? Use `lognormal`.
3. Genuinely skewed and all three quantiles must be exact? Try `lognormal3`.
4. Otherwise the triple is the problem. Go back to Stage 1.

**Do not reach for `triangular` or `pert` to silence the notice.** They will
silence it, and they redefine `p10`/`p90` as hard minimum and maximum rather than
percentiles - a materially narrower belief than the one you wrote, with zero
probability outside the band. Choose them only if you positively mean that.

## 4. Factors and loadings

Two or three factors. Define in words what **high** means for each, in the
header comment, *before* signing any loading - that definition is the contract
every sign has to obey.

Then sign each loading from the mechanism, not from a correlation lookup.
Fitting loadings to observed co-movement is exactly the overfitting to avoid; the
question is "if this factor moved, how would this driver move, and why", and the
answer is a sentence in the report, not a regression.

`Σ(loading²) ≤ 1` per driver - the residual is that driver's idiosyncratic
variance. A driver with genuinely no factor exposure gets zeros, and that is an
honest row rather than a missing one.

**Check the signs after running.** A driver that is *subtracted* by the adapter
has an inverted tornado bar; if a factor that should help the company appears to
hurt it, the sign is backwards.

## 5. Header comment

The spec has to carry its own provenance, because the numbers outlive the
conversation that produced them. Record: the Stage 1 report it came from, the
basis for each judgment-heavy input, what each factor's "high" means, any
constant deliberately left frozen with its range and effect, and the real output
of the validate and run below.

## 6. Before delivering

- [ ] Every `p10/p50/p90` traces to a Stage 1 triple.
- [ ] Every constant traces to a Stage 1 figure.
- [ ] Every constant hiding a range has been promoted, or its range and measured
      effect are stated in the header.
- [ ] Loadings justified against the report; `Σλ² ≤ 1`; blend weights sum to 100.
- [ ] `valdist validate <spec>` run, output pasted, **no NOTICE** left unresolved.
- [ ] `valdist run <spec>` run, output pasted, **no WARNINGS block**.

A `WARNINGS` block means the engine could not compute what you asked. Floored
draws are not a pessimistic scenario - they are an arithmetic placeholder that
can be thousands of times too large, and they bias the result upward. A
`p_undervalued` of exactly 0 or 1 is not a result either: the `±0.0000` beside it
means no spread, not no doubt. Fix the bands; do not caveat the number.
