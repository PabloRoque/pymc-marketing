#   Copyright 2022 - 2025 The PyMC Labs Developers
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
"""Tests for standardized serialization of Transformation subclasses.

This module tests the wrapped format serialization for all Transformation
subclasses (Adstock, Saturation, Basis classes) to ensure:
- to_dict() returns wrapped format: {"class": "ClassName", "version": 1, "data": {...}}
- from_dict() handles both wrapped and flat formats (backward compatibility)
- JSON round-trip serialization works
- Factory functions work with both formats
- Prior deserialization works correctly
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from pymc_marketing.mmm.components.adstock import (
    AdstockTransformation,
    BinomialAdstock,
    DelayedAdstock,
    GeometricAdstock,
    NoAdstock,
    WeibullCDFAdstock,
    WeibullPDFAdstock,
    adstock_from_dict,
)
from pymc_marketing.mmm.components.saturation import (
    LogisticSaturation,
)
from pymc_marketing.mmm.events import (
    AsymmetricGaussianBasis,
    Basis,
    GaussianBasis,
    HalfGaussianBasis,
)


class TestAdstockWrappedFormat:
    """Test wrapped format serialization for Adstock classes."""

    @pytest.mark.parametrize(
        "adstock_cls,kwargs",
        [
            (BinomialAdstock, {"l_max": 5}),
            (GeometricAdstock, {"l_max": 5}),
            (DelayedAdstock, {"l_max": 5}),
            (WeibullPDFAdstock, {"l_max": 5}),
            (WeibullCDFAdstock, {"l_max": 5}),
            (NoAdstock, {"l_max": 5}),
        ],
    )
    def test_adstock_to_dict_wrapped_format(
        self, adstock_cls: type[AdstockTransformation], kwargs: dict[str, Any]
    ) -> None:
        """Test that to_dict() returns wrapped format for Adstock classes."""
        adstock = adstock_cls(**kwargs)
        data = adstock.to_dict()

        # Check wrapped format structure
        assert isinstance(data, dict)
        assert "class" in data, (
            "to_dict() should return wrapped format with 'class' key"
        )
        assert data["class"] == adstock_cls.__name__
        assert "version" in data
        assert data["version"] == 1
        assert "data" in data
        assert isinstance(data["data"], dict)

        # Check that data contains expected fields
        assert "lookup_name" in data["data"]
        assert "prefix" in data["data"]
        assert "l_max" in data["data"]

    @pytest.mark.parametrize(
        "adstock_cls,kwargs",
        [
            (BinomialAdstock, {"l_max": 7}),
            (GeometricAdstock, {"l_max": 10}),
        ],
    )
    def test_adstock_from_dict_wrapped_format(
        self, adstock_cls: type[AdstockTransformation], kwargs: dict[str, Any]
    ) -> None:
        """Test that from_dict() can load wrapped format."""
        adstock = adstock_cls(**kwargs)
        data = adstock.to_dict()

        # Load from wrapped format
        restored = adstock_cls.from_dict(data)
        assert isinstance(restored, adstock_cls)
        assert restored.l_max == adstock.l_max
        assert restored.lookup_name == adstock.lookup_name

    def test_adstock_from_dict_flat_backward_compat(self) -> None:
        """Test that from_dict() handles old flat format."""
        # Old flat format
        old_data = {
            "lookup_name": "geometric",
            "prefix": "adstock",
            "priors": {},
            "l_max": 5,
        }

        # Should still load (backward compatibility)
        adstock = GeometricAdstock.from_dict(old_data)
        assert isinstance(adstock, GeometricAdstock)
        assert adstock.l_max == 5

    @pytest.mark.parametrize(
        "adstock_cls,kwargs",
        [
            (BinomialAdstock, {"l_max": 6}),
            (GeometricAdstock, {"l_max": 8}),
        ],
    )
    def test_adstock_json_roundtrip(
        self, adstock_cls: type[AdstockTransformation], kwargs: dict[str, Any]
    ) -> None:
        """Test JSON serialization round-trip."""
        adstock = adstock_cls(**kwargs)
        data = adstock.to_dict()

        # Serialize to JSON
        json_str = json.dumps(data)
        restored_data = json.loads(json_str)

        # Deserialize
        restored = adstock_cls.from_dict(restored_data)
        assert restored.l_max == adstock.l_max
        assert restored.lookup_name == adstock.lookup_name

    @pytest.mark.parametrize(
        "adstock_cls,lookup_name",
        [
            (BinomialAdstock, "binomial"),
            (GeometricAdstock, "geometric"),
            (DelayedAdstock, "delayed"),
        ],
    )
    def test_adstock_factory_wrapped_format(
        self, adstock_cls: type[AdstockTransformation], lookup_name: str
    ) -> None:
        """Test that factory function works with wrapped format."""
        adstock = adstock_cls(l_max=5)
        data = adstock.to_dict()

        # Factory should handle wrapped format
        restored = adstock_from_dict(data)
        assert isinstance(restored, adstock_cls)
        assert restored.lookup_name == lookup_name

    def test_adstock_factory_flat_backward_compat(self) -> None:
        """Test that factory function handles old flat format."""
        old_data = {
            "lookup_name": "geometric",
            "prefix": "adstock",
            "priors": {},
            "l_max": 5,
        }

        # Factory should still work with flat format
        adstock = adstock_from_dict(old_data)
        assert isinstance(adstock, GeometricAdstock)


class TestSaturationWrappedFormat:
    """Test wrapped format serialization for Saturation classes."""

    def test_saturation_to_dict_wrapped_format(self) -> None:
        """Test that to_dict() returns wrapped format."""
        saturation = LogisticSaturation()
        data = saturation.to_dict()

        assert isinstance(data, dict)
        assert "class" in data
        assert data["class"] == "LogisticSaturation"
        assert "version" in data
        assert data["version"] == 1
        assert "data" in data

    def test_saturation_from_dict_wrapped_format(self) -> None:
        """Test that from_dict() can load wrapped format."""
        saturation = LogisticSaturation()
        data = saturation.to_dict()

        restored = LogisticSaturation.from_dict(data)
        assert isinstance(restored, LogisticSaturation)

    def test_saturation_from_dict_flat_backward_compat(self) -> None:
        """Test backward compatibility with flat format."""
        old_data = {
            "lookup_name": "logistic",
            "prefix": "saturation",
            "priors": {},
        }

        saturation = LogisticSaturation.from_dict(old_data)
        assert isinstance(saturation, LogisticSaturation)


class TestBasisWrappedFormat:
    """Test wrapped format serialization for Basis classes."""

    @pytest.mark.parametrize(
        "basis_cls",
        [
            GaussianBasis,
            HalfGaussianBasis,
            AsymmetricGaussianBasis,
        ],
    )
    def test_basis_to_dict_wrapped_format(self, basis_cls: type[Basis]) -> None:
        """Test that to_dict() returns wrapped format."""
        basis = basis_cls()
        data = basis.to_dict()

        assert isinstance(data, dict)
        assert "class" in data
        assert data["class"] == basis_cls.__name__
        assert "version" in data
        assert data["version"] == 1
        assert "data" in data

    @pytest.mark.parametrize(
        "basis_cls",
        [
            GaussianBasis,
            HalfGaussianBasis,
            AsymmetricGaussianBasis,
        ],
    )
    def test_basis_from_dict_wrapped_format(self, basis_cls: type[Basis]) -> None:
        """Test that from_dict() can load wrapped format."""
        basis = basis_cls()
        data = basis.to_dict()

        restored = basis_cls.from_dict(data)
        assert isinstance(restored, basis_cls)

    def test_basis_from_dict_flat_backward_compat(self) -> None:
        """Test backward compatibility with flat format."""
        old_data = {
            "lookup_name": "gaussian",
            "prefix": "basis",
            "priors": {},
        }

        basis = GaussianBasis.from_dict(old_data)
        assert isinstance(basis, GaussianBasis)
