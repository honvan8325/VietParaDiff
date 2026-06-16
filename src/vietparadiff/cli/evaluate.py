"""CLI for text metric utilities.  Full FID/KID/HWD evaluation requires trained external models."""

from __future__ import annotations

import argparse
from vietparadiff.utils.metrics import cer, decomposed_error_rates


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute Vietnamese text error rates")
    parser.add_argument("--pred", required=True)
    parser.add_argument("--target", required=True)
    args = parser.parse_args()
    print({"cer": cer(args.pred, args.target), **decomposed_error_rates(args.pred, args.target)})


if __name__ == "__main__":
    main()
