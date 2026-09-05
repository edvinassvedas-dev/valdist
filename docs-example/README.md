# docs-example/ – worked examples of a valdist instruction set

**Everything in this folder is an example.** It is here to show the *shape* of an
instruction set that drives valdist, so you can write your own. It is not the
instruction set this repository's own analyses were produced with, and it is not
a valuation methodology you should adopt as-is.

## Why the real ones are not here

The instruction sets actually used to produce this project's own work are not
included in this repository: the engine is a general-purpose tool and is
public, but the judgment behind any particular valuation is not.

A second reason those files could not simply be published, stated plainly
because it constrains what this folder can be: parts of that methodology are
**distilled from a copyrighted book**, and are therefore derivative work.
Nothing in `docs-example/` derives from it. These files were written from
scratch against valdist's own public spec format, documented in
[README.md](../README.md).

## What an instruction set is for

valdist consumes a spec: driver marginals, factor loadings, constants, weights.
Somebody has to decide what those numbers are, and the engine deliberately does
not help - fitting them to data defeats the point. An instruction set is how you make
that judgment repeatable: a research stage that produces reasoning and ranges,
and an export stage that turns those ranges into a spec **without introducing a
single new number**.

Splitting it in two is the part worth copying. It keeps "what do I believe"
separate from "how is that written down", so a spec review can ask *where did
this triple come from* and get an answer.

## The example

[two-stage-pipeline/](two-stage-pipeline/) has the three files that shape does
need:

| file | job |
|---|---|
| [`prompt.txt`](two-stage-pipeline/prompt.txt) | what you hand the model, with the two stages attached |
| [`stage1-research.md`](two-stage-pipeline/stage1-research.md) | produce reasoning and a worst/base/best triple per driver |
| [`stage2-spec-export.md`](two-stage-pipeline/stage2-spec-export.md) | serialize those triples into a runnable spec |

They are deliberately thin. A real instruction set for a domain you care about
will be much longer, and most of that length will be the domain's own traps -
which is exactly the part nobody can write for you.

## Trying it

The synthetic specs under [`fixtures/analyses/`](../fixtures/analyses/) are what
a spec produced this way looks like. They are invented end to end; the tickers
are not real listings.

```bash
valdist validate fixtures/analyses/2026-01-05-ACME/2026-01-05-ACME.yaml
valdist run      fixtures/analyses/2026-01-05-ACME/2026-01-05-ACME.yaml
```
