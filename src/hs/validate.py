import math
import numbers

def positive_int(name, v):
    """Real integers only: rejects bool, floats, strings, NaN. Raises ValueError up front."""
    if isinstance(v, bool) or not isinstance(v, numbers.Integral) or v <= 0:
        raise ValueError(f"{name} must be a positive integer, got {v!r}")

def finite_number(name, v):
    if isinstance(v, bool) or not isinstance(v, numbers.Real) or not math.isfinite(v):
        raise ValueError(f"{name} must be a finite number, got {v!r}")
