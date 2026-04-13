"""CLI entry point for bh-sentinel tools."""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="bh-sentinel",
        description="bh-sentinel clinical safety signal detection tools",
    )
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("validate-config", help="Validate config file consistency")
    subparsers.add_parser("test-patterns", help="Run pattern test fixtures")

    args = parser.parse_args()

    if args.command == "validate-config":
        from bh_sentinel.cli.validate_config import run_validate

        sys.exit(run_validate())
    elif args.command == "test-patterns":
        from bh_sentinel.cli.test_patterns import run_test_patterns

        sys.exit(run_test_patterns())
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
