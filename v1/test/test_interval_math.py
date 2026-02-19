"""
test_interval_math.py — Unit tests for interval_math.py

Covers:
    - Interval arithmetic (add, sub, mul, neg, radius)
    - Interval trigonometric functions (I_sin, I_cos)
    - AffineForm arithmetic (add, sub, mul, neg, to_interval)
    - smart_sin / smart_cos on AffineForm
    - Edge cases (zero-width interval, degenerate inputs)
"""

import math
import numpy as np
import pytest

from box_aabb.interval_math import (
    Interval,
    AffineForm,
    I_sin,
    I_cos,
    smart_sin,
    smart_cos,
    reset_affine_noise_counter,
)


# Helper: Interval has no `contains` or `width`, so we write thin wrappers.
def _width(iv: Interval) -> float:
    return iv.max - iv.min


def _contains(iv: Interval, x: float) -> bool:
    return iv.min <= x <= iv.max


# =========================================================================
# Interval
# =========================================================================

class TestInterval:

    def test_creation(self):
        iv = Interval(1, 3)
        assert iv.min == 1.0
        assert iv.max == 3.0

    def test_auto_swap(self):
        """Interval(hi, lo) should auto-swap."""
        iv = Interval(5, 2)
        assert iv.min == 2.0 and iv.max == 5.0

    def test_center(self):
        assert Interval(-1, 3).center == 1.0

    def test_radius(self):
        iv = Interval(2, 5)
        assert iv.radius == pytest.approx(1.5)
        assert _width(iv) == pytest.approx(3.0)

    def test_containment(self):
        iv = Interval(-1, 1)
        assert _contains(iv, 0)
        assert _contains(iv, -1)
        assert _contains(iv, 1)
        assert not _contains(iv, 2)

    def test_add(self):
        a = Interval(1, 2)
        b = Interval(3, 5)
        c = a + b
        assert c.min == pytest.approx(4)
        assert c.max == pytest.approx(7)

    def test_sub(self):
        a = Interval(1, 4)
        b = Interval(1, 2)
        c = a - b
        assert c.min == pytest.approx(-1)
        assert c.max == pytest.approx(3)

    def test_mul_positive(self):
        a = Interval(2, 3)
        b = Interval(4, 5)
        c = a * b
        assert c.min == pytest.approx(8)
        assert c.max == pytest.approx(15)

    def test_mul_mixed_sign(self):
        a = Interval(-1, 2)
        b = Interval(-3, 4)
        c = a * b
        assert c.min == pytest.approx(-6)
        assert c.max == pytest.approx(8)

    def test_neg(self):
        iv = Interval(2, 5)
        neg = -iv
        assert neg.min == pytest.approx(-5)
        assert neg.max == pytest.approx(-2)

    def test_scalar_mul(self):
        iv = Interval(1, 3)
        c = iv * 2
        assert c.min == pytest.approx(2)
        assert c.max == pytest.approx(6)

    def test_zero_width(self):
        iv = Interval(3, 3)
        assert _width(iv) == 0
        assert iv.center == 3.0


class TestIntervalTrig:

    def test_I_sin_narrow(self):
        """sin on [0, 0.1] should tightly contain sampled values."""
        result = I_sin(Interval(0, 0.1))
        for x in [0, 0.05, 0.1]:
            assert _contains(result, math.sin(x))

    def test_I_cos_narrow(self):
        result = I_cos(Interval(0, 0.1))
        for x in [0, 0.05, 0.1]:
            assert _contains(result, math.cos(x))

    def test_I_sin_full_range(self):
        """sin on [-pi, pi] must contain [-1, 1]."""
        result = I_sin(Interval(-math.pi, math.pi))
        assert result.min <= -1.0 + 1e-10
        assert result.max >= 1.0 - 1e-10

    def test_I_cos_full_range(self):
        result = I_cos(Interval(-math.pi, math.pi))
        assert result.min <= -1.0 + 1e-10
        assert result.max >= 1.0 - 1e-10

    def test_I_sin_contains_samples(self):
        """Samples within the interval must be contained."""
        iv = Interval(-1.0, 0.5)
        result = I_sin(iv)
        for x in np.linspace(iv.min, iv.max, 50):
            assert _contains(result, math.sin(x)), f"sin({x}) not in {result}"

    def test_I_cos_contains_samples(self):
        iv = Interval(0.5, 2.0)
        result = I_cos(iv)
        for x in np.linspace(iv.min, iv.max, 50):
            assert _contains(result, math.cos(x)), f"cos({x}) not in {result}"


# =========================================================================
# AffineForm
# =========================================================================

class TestAffineForm:

    def setup_method(self):
        reset_affine_noise_counter()

    def test_from_interval(self):
        af = AffineForm.from_interval(1, 3)
        iv = af.to_interval()
        assert iv.min <= 1.0
        assert iv.max >= 3.0

    def test_constant(self):
        af = AffineForm(5.0)
        iv = af.to_interval()
        assert iv.min == pytest.approx(5.0)
        assert iv.max == pytest.approx(5.0)

    def test_add(self):
        a = AffineForm.from_interval(1, 3)
        b = AffineForm.from_interval(2, 4)
        c = a + b
        iv = c.to_interval()
        assert iv.min <= 3.0  # 1+2
        assert iv.max >= 7.0  # 3+4

    def test_sub(self):
        a = AffineForm.from_interval(1, 3)
        b = AffineForm(1.0)  # constant 1
        c = a - b
        iv = c.to_interval()
        assert iv.min <= 0.0
        assert iv.max >= 2.0

    def test_neg(self):
        a = AffineForm.from_interval(2, 5)
        b = -a
        iv = b.to_interval()
        assert iv.min <= -5.0
        assert iv.max >= -2.0

    def test_mul_scalar(self):
        a = AffineForm.from_interval(1, 3)
        b = a * 2
        iv = b.to_interval()
        assert iv.min <= 2.0
        assert iv.max >= 6.0

    def test_smart_sin_contains_samples(self):
        af = AffineForm.from_interval(-0.5, 0.5)
        result = smart_sin(af)
        iv = result.to_interval()
        for x in np.linspace(-0.5, 0.5, 20):
            assert _contains(iv, math.sin(x)), (
                f"smart_sin failed to contain sin({x})={math.sin(x)}, "
                f"interval=[{iv.min}, {iv.max}]")

    def test_smart_cos_contains_samples(self):
        af = AffineForm.from_interval(-0.5, 0.5)
        result = smart_cos(af)
        iv = result.to_interval()
        for x in np.linspace(-0.5, 0.5, 20):
            assert _contains(iv, math.cos(x)), (
                f"smart_cos failed to contain cos({x})={math.cos(x)}, "
                f"interval=[{iv.min}, {iv.max}]")
