# Stage 1 (EXAMPLE) – research

> **This is an example.** It shows the shape of a research stage, not a
> methodology to adopt. The real instruction sets are private; see
> [../README.md](../README.md) for why.

Stage 1 produces *reasoning and ranges*. It produces no spec. Its only obligation
to Stage 2 is that every number Stage 2 will need already exists here, with a
source and a date beside it.

## 1. Data, and the hard stop

Every figure must be retrieved from a current primary source and carry that
source and an as-of date. Numbers recalled from memory are stale by definition.

**State up front whether you have a working retrieval tool.** If you do not,
stop and say the valuation cannot be completed. A hard stop is a correct output;
a plausible-looking report built from remembered figures is not, and it is worse
than nothing because it is indistinguishable from a real one.

## 2. Choose the adapter, then read what it requires

The adapter decides which drivers exist. Ask the registry rather than guessing:

```python
from valdist.adapters.registry import required_inputs, known_valuations

sorted(known_valuations())  # what is available
sorted(required_inputs("reit_v2"))  # what that one needs
```

Every name it returns must end up in the spec, as a sampled driver or a constant.

## 3. Give every driver a worst / base / best

This is the whole job. For each required input, write three numbers and the
reason for each:

- **base** – your central estimate, and how you got there;
- **worst** and **best** – not the extremes you can imagine, but the ends of the
  range you would actually defend.

Two rules that save trouble later:

- **Base must sit between worst and best.** If it does not, the belief is
  incoherent and Stage 2 will only hide that.
- **Ask whether the range is two-sided.** A driver you have only argued downside
  for usually means you have not looked for the upside, not that there is none.

## 4. Say which numbers you do not know

A constant is a claim that you know the figure. Most inputs are: a share count, a
declared dividend. Some are not - anything you had to *estimate* is a judgment
wearing a constant's clothes.

For each, write the range down here even if you expect to freeze it. Stage 2
decides whether it becomes a driver, and it cannot make that decision from a
number with no range attached.

## 5. Report format

Deliver in this order:

1. **Preflight line** - retrieval tool, named, or the hard stop.
2. **Verdict** - one or two sentences: the conclusion and your confidence in it.
3. **Snapshot table** - ticker, price with its as-of date, and the headline
   figures a reader needs to follow the rest.
4. **The methods** - one section each, showing the inputs, the arithmetic, and a
   sensitivity over whichever input the result is most exposed to.
5. **Cross-check** - where two methods disagree, say so and say why. Averaging a
   disagreement away is the one thing not to do: the gap is information.
6. **Risks** - what would have to be true for this to be wrong.
7. **Assumptions and sources** - every figure, its source, its as-of date, and
   the **worst / base / best triple for every driver**. A report without this
   block cannot be exported.
8. **Confidence** - high / medium / low, tied to input quality and to whether the
   methods agreed.
