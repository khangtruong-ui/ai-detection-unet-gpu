"""
CLI entrypoint and utilities for duplicating and modifying SID-UNet configuration files (sid-copy-config).
Allows developers and researchers to quickly duplicate baseline and experiment YAML configs,
apply arbitrary CLI dot-notation overrides, and save or inspect modified configurations.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Optional
import yaml

from sid_unet.utils.config import copy_config
from sid_unet.utils.logger import setup_logger


def parse_args(args: Optional[list] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Duplicate and customize SID-UNet configuration files (sid-copy-config)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "source",
        nargs="?",
        default=None,
        help="Path to source YAML configuration file (positional)",
    )
    parser.add_argument(
        "dest",
        nargs="?",
        default=None,
        help="Path to destination YAML configuration file (positional)",
    )
    parser.add_argument(
        "--src",
        "--source",
        "--config",
        dest="source_flag",
        default=None,
        help="Path to source YAML configuration file",
    )
    parser.add_argument(
        "--dst",
        "--dest",
        "--output",
        "-o",
        dest="dest_flag",
        default=None,
        help="Path to destination YAML configuration file",
    )
    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        help="Configuration overrides in key.nested=value format (e.g. --override data.image_size=[512,512])",
    )
    parser.add_argument(
        "--merge-defaults",
        "--with-defaults",
        action="store_true",
        default=False,
        help="Merge with project DEFAULT_CONFIG before applying overrides (default: false, preserving source keys only)",
    )
    parser.add_argument(
        "--print",
        "-p",
        action="store_true",
        default=False,
        help="Print modified configuration YAML to stdout",
    )
    return parser.parse_args(args)


def cli_main(args: Optional[list] = None) -> None:
    from sid_unet.utils.signals import shield_process_signals
    shield_process_signals()

    parsed = parse_args(args)
    logger = setup_logger(name="sid_copy_config")

    source = parsed.source or parsed.source_flag
    dest = parsed.dest or parsed.dest_flag

    if not source:
        logger.error(
            "Source configuration file must be specified.\n"
            "Usage: sid-copy-config <source.yaml> <dest.yaml> [--override key=value]"
        )
        sys.exit(1)

    if not os.path.exists(source):
        logger.error(f"Source configuration file not found: {source}")
        sys.exit(1)

    if not dest and not parsed.print:
        logger.error(
            "Destination configuration path or --print must be specified.\n"
            "Usage: sid-copy-config <source.yaml> <dest.yaml> [--override key=value]"
        )
        sys.exit(1)

    overrides = list(parsed.override or [])
    try:
        config_dict = copy_config(
            source_path=source,
            dest_path=dest,
            overrides=overrides,
            merge_defaults=parsed.merge_defaults,
        )
    except Exception as e:
        logger.error(f"Failed to copy and override configuration: {e}")
        sys.exit(1)

    if dest and dest != "-":
        resolved_dest = os.path.join(dest, os.path.basename(source)) if os.path.isdir(dest) else dest
        logger.info(
            f"✅ Configuration successfully copied from '{source}' to '{resolved_dest}' "
            f"with {len(overrides)} override(s) applied."
        )

    if parsed.print or dest == "-":
        yaml_str = yaml.safe_dump(config_dict, default_flow_style=False, sort_keys=False)
        print(yaml_str)


if __name__ == "__main__":
    cli_main()
