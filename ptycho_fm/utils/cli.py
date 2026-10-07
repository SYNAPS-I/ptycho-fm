"""Shared argparse type converters for command-line entry points."""

import argparse
import math


def positive_int(value: str) -> int:
    """Parse an integer greater than zero."""
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def nonnegative_float(value: str) -> float:
    """Parse a finite float greater than or equal to zero."""
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("must be a finite, nonnegative number")
    return number


def positive_float(value: str) -> float:
    """Parse a finite float greater than zero."""
    number = nonnegative_float(value)
    if number == 0:
        raise argparse.ArgumentTypeError("must be a positive number")
    return number


def parse_boolean(value: str) -> bool:
    """Parse a case-insensitive explicit true/false CLI value."""
    normalized = value.lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise argparse.ArgumentTypeError("expected true or false")
