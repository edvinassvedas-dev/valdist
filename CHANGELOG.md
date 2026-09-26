# Changelog

Test suite: 704 -> 747

## 2026-09-26

### Added

- `clinical_v1` adapter for pre-revenue drug developers: net cash, less the discounted
  burn, plus a probability-weighted asset, with an explicit raise, a fractional
  `years_to_approval` and an optional `years_to_readout` that stops the burn at
  a failed readout. `equity_v2` discounts a negative cash flow forever. A
  negative `asset_value` draw is floored at zero and flagged
  (`clinical_asset_value_floored`): a `lognormal3` fit put 2.79% of one spec's
  asset draws below zero, silently subtracting value.
- `valdist run --swings`: each driver's own signed effect, p10 to p90, others at
  median. The tornado bar can borrow its sign from a shared factor (rate -0.093
  against a positive derivative on a clinical spec).
- `valdist solve`: the deterministic value at every driver's typed p50, and each
  driver's implied input at the price, with `--set NAME=VALUE` for branches.
  Implied inputs and branch values had been computed in ad-hoc scripts.

### Fixed

- Viewer: **All analyses** gains a currency column.

## 2026-09-20

### Changed

- Viewer: **All analyses** now fills in batches of 8 rather than in one blocking
  request.

## 2026-09-13

### Added

- Viewer: checkboxes in the analysis list, and **Run selected** on **All analyses**.

### Changed

- Viewer: opening **All analyses** shows cached results instead of running every
  spec; **Run all** runs the rest, and a run from an analysis page fills its row.

## 2026-09-10

### Added

- Viewer: multi-select filter on the **All analyses** table.
