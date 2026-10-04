"""EGridCast: VIC1 hourly demand forecasting."""

import os
import sys

# PyTorch and XGBoost ship distinct OpenMP runtimes on macOS.
# Bound parallelism before either runtime loads; do not alter other platforms.
if sys.platform == "darwin":
    os.environ["OMP_NUM_THREADS"] = "1"


def main() -> None:
    from egridcast.training import main as cli

    cli()
