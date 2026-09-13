"""One click command per table, plus `build` and `migrate`.

    uv run pipelines --help
    uv run pipelines option-greeks --start 2024-01-01 --end 2024-12-31
    uv run pipelines build

Every command takes at most `--start` and `--end`, and both default to the
window the store is meant to hold, so a bare command reproduces the same
data every time. Run one command at a time: ThetaData issues one session
per account and two processes fight over it.
"""

import datetime as dt
import time
from collections.abc import Callable

import click

from pipelines import (
    calendar,
    corporate_actions,
    earnings,
    factor_model,
    forecast,
    index_greeks,
    index_repair,
    indices,
    migrate,
    open_interest,
    option_greeks,
    rates,
    realized_vol,
    reference_returns,
    sectors,
    signals,
    stock_features,
    surface,
    symbology,
    underlying,
    universe,
    yields,
)

# Command name -> module. The order is `build`'s order: raw pulls first,
# in dependency order, then the derived tables.
WINDOWED = {
    "calendar": calendar,
    "universe": universe,
    "indices": indices,
    "yields": yields,
    "rates": rates,
    "underlying": underlying,
    "option-greeks": option_greeks,
    "open-interest": open_interest,
    "index-greeks": index_greeks,
    "index-repair": index_repair,
    "symbology": symbology,
    "reference-returns": reference_returns,
    "factor-model": factor_model,
    "surface": surface,
    "realized-vol": realized_vol,
    "forecast": forecast,
    "stock-features": stock_features,
    "signals": signals,
}
BUILD_ORDER = [
    "calendar",
    "universe",
    "sectors",
    "indices",
    "yields",
    "rates",
    "corporate-actions",
    "earnings",
    "underlying",
    "option-greeks",
    "open-interest",
    "index-greeks",
    "index-repair",
    "symbology",
    "reference-returns",
    "factor-model",
    "surface",
    "realized-vol",
    "forecast",
    "stock-features",
    "signals",
]


def parse_date(value: str) -> dt.date:
    return dt.date.fromisoformat(value)


@click.group(help=__doc__)
def cli() -> None:
    pass


def register_windowed(name: str, module) -> None:
    summary = module.__doc__.strip().splitlines()[0]

    @cli.command(name=name, help=summary)
    @click.option("--start", type=parse_date, default=module.START.isoformat(), show_default=True)
    @click.option("--end", type=parse_date, default=module.END.isoformat(), show_default=True)
    def command(start: dt.date, end: dt.date) -> None:
        module.run(start, end)


for command_name, command_module in WINDOWED.items():
    register_windowed(command_name, command_module)


@cli.command(name="sectors", help=sectors.__doc__.strip().splitlines()[0])
def sectors_command() -> None:
    sectors.run()


@cli.command(name="earnings", help=earnings.__doc__.strip().splitlines()[0])
def earnings_command() -> None:
    earnings.run()


@cli.command(name="corporate-actions", help=corporate_actions.__doc__.strip().splitlines()[0])
@click.option(
    "--start", type=parse_date, default=corporate_actions.START.isoformat(), show_default=True
)
def corporate_actions_command(start: dt.date) -> None:
    corporate_actions.run(start)


@cli.command(name="migrate", help=migrate.__doc__.strip().splitlines()[0])
@click.option("--source", type=click.Path(exists=True, file_okay=False), required=True)
def migrate_command(source: str) -> None:
    migrate.run(source)


@cli.command(name="build", help="Run every step in order with its defaults.")
@click.option("--skip-to", default=None, help="resume from this step")
@click.option("--dry-run", is_flag=True, help="print the order and stop")
def build_command(skip_to: str | None, dry_run: bool) -> None:
    steps = BUILD_ORDER
    if skip_to is not None:
        if skip_to not in steps:
            raise click.BadParameter(f"unknown step {skip_to!r}")
        steps = steps[steps.index(skip_to) :]
    if dry_run:
        click.echo("\n".join(steps))
        return
    started = time.perf_counter()
    for number, name in enumerate(steps, start=1):
        click.echo(f"\n=== [{number}/{len(steps)}] {name} ===")
        step_started = time.perf_counter()
        run_step(name)()
        click.echo(f"--- {name}: {(time.perf_counter() - step_started) / 60:.1f} min")
    click.echo(f"\nstore built in {(time.perf_counter() - started) / 3600:.1f} hr")


def run_step(name: str) -> Callable[[], None]:
    if name in WINDOWED:
        module = WINDOWED[name]
        return lambda: module.run(module.START, module.END)
    return {
        "sectors": sectors.run,
        "earnings": earnings.run,
        "corporate-actions": lambda: corporate_actions.run(corporate_actions.START),
    }[name]


if __name__ == "__main__":
    cli()
