"""valdist CLI — run | validate | worlds | calibrate."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import typer

app = typer.Typer(
    name="valdist",
    help="Probabilistic intrinsic-value engine.",
    add_completion=False,
)


def _checked_sampling(sampling: str) -> Literal["mc", "lhs"]:
    """Validate --sampling once, for every command that runs a model."""
    if sampling not in ("mc", "lhs"):
        typer.echo(f"--sampling must be 'mc' or 'lhs', got {sampling!r}", err=True)
        raise typer.Exit(code=1)
    return sampling  # type: ignore[return-value]


def _load_and_hydrate(spec_path: Path):
    """Load a spec and build its Model, turning a rejected spec into a clean
    CLI failure (the digest on stderr, exit 1) rather than a raw traceback.
    """
    from valdist.spec.loader import hydrate, load_spec
    from valdist.spec.validate import SpecError, check_marginal_fit

    model_spec = load_spec(spec_path)
    try:
        model = hydrate(model_spec)
    except SpecError as exc:
        typer.echo(f"INVALID – '{spec_path}' has {len(exc.errors)} error(s):", err=True)
        for e in exc.errors:
            typer.echo(f"  {e}", err=True)
        raise typer.Exit(code=1) from None

    # Non-fatal quality notices never block the run, but must not be invisible.
    for n in check_marginal_fit(model_spec):
        typer.echo(f"NOTICE – {n}", err=True)

    return model_spec, model


@app.command()
def run(
    spec: Path = typer.Argument(..., help="Path to YAML/JSON spec file."),
    sampling: str = typer.Option(
        "mc", "--sampling", help="'mc' (default) or 'lhs' (Latin Hypercube Sampling)."
    ),
) -> None:
    """Run a valuation spec and print results."""
    from valdist import report

    sampling_literal = _checked_sampling(sampling)
    model_spec, model = _load_and_hydrate(spec)
    result = model.run(
        n=model_spec.n,
        price=model_spec.price,
        seed=model_spec.seed,
        sampling=sampling_literal,
        nu=model_spec.nu,
    )
    typer.echo(report.to_text(result))


@app.command()
def validate(
    spec: Path = typer.Argument(..., help="Path to YAML/JSON spec file."),
) -> None:
    """Validate a spec file and report all errors, plus non-fatal quality notices."""
    from valdist.spec.loader import load_spec
    from valdist.spec.validate import check_marginal_fit
    from valdist.spec.validate import validate as _validate

    model_spec = load_spec(spec)
    errors = _validate(model_spec)
    notices = check_marginal_fit(model_spec)

    if not errors:
        typer.echo(f"OK – '{spec}' is valid.")
    else:
        typer.echo(f"INVALID – '{spec}' has {len(errors)} error(s):", err=True)
        for e in errors:
            typer.echo(f"  {e}", err=True)

    if notices:
        typer.echo(f"NOTICE – {len(notices)} non-fatal quality notice(s):")
        for n in notices:
            typer.echo(f"  {n}")

    if errors:
        raise typer.Exit(code=1)


@app.command()
def worlds(
    spec: Path = typer.Argument(..., help="Path to YAML/JSON spec file."),
    sampling: str = typer.Option(
        "mc", "--sampling", help="'mc' (default) or 'lhs' (Latin Hypercube Sampling)."
    ),
) -> None:
    """Extract percentile worlds from a spec run."""
    from valdist import report

    sampling_literal = _checked_sampling(sampling)
    model_spec, model = _load_and_hydrate(spec)
    result = model.run(
        n=model_spec.n,
        price=model_spec.price,
        seed=model_spec.seed,
        sampling=sampling_literal,
        nu=model_spec.nu,
    )
    typer.echo(report.print_worlds(result))


@app.command()
def plot(
    spec: Path = typer.Argument(..., help="Path to YAML/JSON spec file."),
    kind: str = typer.Option("tornado", "--kind", help="'tornado' or 'distribution'."),
    out: Path = typer.Option(..., "--out", help="Output PNG path."),
    theme: str = typer.Option("light", "--theme", help="'light' or 'dark'."),
    sampling: str = typer.Option(
        "mc", "--sampling", help="'mc' (default) or 'lhs' (Latin Hypercube Sampling)."
    ),
) -> None:
    """Render a chart from a spec run (requires `pip install valdist[plot]`)."""
    if kind not in ("tornado", "distribution"):
        typer.echo(f"--kind must be 'tornado' or 'distribution', got {kind!r}", err=True)
        raise typer.Exit(code=1)
    if theme not in ("light", "dark"):
        typer.echo(f"--theme must be 'light' or 'dark', got {theme!r}", err=True)
        raise typer.Exit(code=1)
    # Without this, `valdist plot` after `valdist run --sampling lhs` charted a
    # DIFFERENT set of draws than the report the analyst had just read.
    sampling_literal = _checked_sampling(sampling)

    try:
        from valdist import plot as plot_module
    except ImportError:
        typer.echo(
            "matplotlib is not installed – run `pip install valdist[plot]` to use this command.",
            err=True,
        )
        raise typer.Exit(code=1) from None

    model_spec, model = _load_and_hydrate(spec)
    result = model.run(
        n=model_spec.n,
        price=model_spec.price,
        seed=model_spec.seed,
        sampling=sampling_literal,
        nu=model_spec.nu,
    )

    fig = (
        plot_module.plot_tornado(result, theme=theme)
        if kind == "tornado"
        else plot_module.plot_value_distribution(result, theme=theme)
    )
    try:
        fig.savefig(out, dpi=150)
    finally:
        # matplotlib keeps every un-closed figure alive in its global registry,
        # so a programmatic caller looping over specs leaks one per call.
        import matplotlib.pyplot as plt

        plt.close(fig)
    typer.echo(f"Wrote {out}")


@app.command()
def calibrate(
    folder: Path = typer.Argument(..., help="Folder of (spec, realised) pairs."),
    lo: float = typer.Option(0.10, "--lo", help="Lower quantile of coverage band."),
    hi: float = typer.Option(0.90, "--hi", help="Upper quantile of coverage band."),
    n: int = typer.Option(50_000, "--n", help="MC draws per spec."),
    seed: int = typer.Option(0, "--seed", help="RNG seed per spec."),
    sampling: str = typer.Option(
        "mc", "--sampling", help="'mc' (default) or 'lhs' (Latin Hypercube Sampling)."
    ),
) -> None:
    """Check calibration: fraction of realised outcomes inside the P10-P90 band."""
    from valdist.calibrate.coverage import check_coverage

    sampling_literal = _checked_sampling(sampling)
    result = check_coverage(folder, band=(lo, hi), n=n, seed=seed, sampling=sampling_literal)

    if result.n_pairs == 0 and not result.errors:
        typer.echo("No (spec, realised) pairs found in folder.", err=True)
        raise typer.Exit(code=1)

    if result.n_pairs:
        lo_pct = int(round(lo * 100))
        hi_pct = int(round(hi * 100))
        typer.echo(f"Coverage check – {result.n_pairs} pair(s)")
        typer.echo(f"  Band:          P{lo_pct}–P{hi_pct}")
        typer.echo(f"  Expected rate: {result.expected_rate:.0%}")
        typer.echo(f"  Observed rate: {result.hit_rate:.1%}  ({result.n_hits}/{result.n_pairs})")

    # A partial backtest is still worth printing, but the rate was computed
    # over a subset of the folder - say so loudly rather than letting it read
    # as a clean result over every pair.
    if result.errors:
        typer.echo(
            f"\nSKIPPED – {len(result.errors)} pair(s) could not be scored; "
            "the rate above covers only the rest:",
            err=True,
        )
        for e in result.errors:
            typer.echo(f"  {e}", err=True)
        raise typer.Exit(code=1)
