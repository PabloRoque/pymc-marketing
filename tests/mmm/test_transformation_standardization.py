#   Copyright 2022 - 2026 The PyMC Labs Developers
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
- to_dict() returns wrapped format: {"class": "ClassName", "data": {...}}
- from_dict() handles wrapped format
- JSON round-trip serialization works
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
    HillSaturation,
    HillSaturationSigmoid,
    InverseScaledLogisticSaturation,
    LogisticSaturation,
    MichaelisMentenSaturation,
    NoSaturation,
    RootSaturation,
    SaturationTransformation,
    TanhSaturation,
    TanhSaturationBaselined,
    saturation_from_dict,
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

    @pytest.mark.parametrize(
        "saturation_cls,lookup_name",
        [
            (LogisticSaturation, "logistic"),
            (InverseScaledLogisticSaturation, "inverse_scaled_logistic"),
            (TanhSaturation, "tanh"),
            (TanhSaturationBaselined, "tanh_baselined"),
            (MichaelisMentenSaturation, "michaelis_menten"),
            (HillSaturation, "hill"),
            (HillSaturationSigmoid, "hill_sigmoid"),
            (RootSaturation, "root"),
            (NoSaturation, "no_saturation"),
        ],
    )
    def test_saturation_to_dict_wrapped_format(
        self, saturation_cls: type[SaturationTransformation], lookup_name: str
    ) -> None:
        """Test that to_dict() returns wrapped format for all saturation classes."""
        saturation = saturation_cls()
        data = saturation.to_dict()

        # Check wrapped format structure
        assert isinstance(data, dict)
        assert "class" in data, (
            "to_dict() should return wrapped format with 'class' key"
        )
        assert data["class"] == saturation_cls.__name__
        assert "data" in data
        assert isinstance(data["data"], dict)

        # Check that data contains expected fields
        assert "lookup_name" in data["data"]
        assert data["data"]["lookup_name"] == lookup_name
        assert "prefix" in data["data"]
        assert data["data"]["prefix"] == "saturation"

    @pytest.mark.parametrize(
        "saturation_cls",
        [
            LogisticSaturation,
            InverseScaledLogisticSaturation,
            TanhSaturation,
            TanhSaturationBaselined,
            MichaelisMentenSaturation,
            HillSaturation,
            HillSaturationSigmoid,
            RootSaturation,
            NoSaturation,
        ],
    )
    def test_saturation_from_dict_wrapped_format(
        self, saturation_cls: type[SaturationTransformation]
    ) -> None:
        """Test that from_dict() can load wrapped format."""
        saturation = saturation_cls()
        data = saturation.to_dict()

        # Load from wrapped format
        restored = saturation_cls.from_dict(data)
        assert isinstance(restored, saturation_cls)
        assert restored.lookup_name == saturation.lookup_name

    @pytest.mark.parametrize(
        "saturation_cls,lookup_name",
        [
            (LogisticSaturation, "logistic"),
            (InverseScaledLogisticSaturation, "inverse_scaled_logistic"),
            (TanhSaturation, "tanh"),
            (TanhSaturationBaselined, "tanh_baselined"),
            (MichaelisMentenSaturation, "michaelis_menten"),
            (HillSaturation, "hill"),
            (HillSaturationSigmoid, "hill_sigmoid"),
            (RootSaturation, "root"),
            (NoSaturation, "no_saturation"),
        ],
    )
    def test_saturation_from_dict_flat_backward_compat(
        self, saturation_cls: type[SaturationTransformation], lookup_name: str
    ) -> None:
        """Test that from_dict() handles old flat format for backward compatibility."""
        # Old flat format
        old_data = {
            "lookup_name": lookup_name,
            "prefix": "saturation",
            "priors": {},
        }

        # Should still load (backward compatibility)
        saturation = saturation_cls.from_dict(old_data)
        assert isinstance(saturation, saturation_cls)
        assert saturation.lookup_name == lookup_name

    @pytest.mark.parametrize(
        "saturation_cls",
        [
            LogisticSaturation,
            InverseScaledLogisticSaturation,
            TanhSaturation,
            TanhSaturationBaselined,
            MichaelisMentenSaturation,
            HillSaturation,
            HillSaturationSigmoid,
            RootSaturation,
            NoSaturation,
        ],
    )
    def test_saturation_json_roundtrip(
        self, saturation_cls: type[SaturationTransformation]
    ) -> None:
        """Test JSON serialization round-trip for saturation classes."""
        saturation = saturation_cls()
        data = saturation.to_dict()

        # Serialize to JSON
        json_str = json.dumps(data)
        restored_data = json.loads(json_str)

        # Deserialize
        restored = saturation_cls.from_dict(restored_data)
        assert isinstance(restored, saturation_cls)
        assert restored.lookup_name == saturation.lookup_name

    @pytest.mark.parametrize(
        "saturation_cls,lookup_name",
        [
            (LogisticSaturation, "logistic"),
            (InverseScaledLogisticSaturation, "inverse_scaled_logistic"),
            (TanhSaturation, "tanh"),
            (TanhSaturationBaselined, "tanh_baselined"),
            (MichaelisMentenSaturation, "michaelis_menten"),
            (HillSaturation, "hill"),
            (HillSaturationSigmoid, "hill_sigmoid"),
            (RootSaturation, "root"),
            (NoSaturation, "no_saturation"),
        ],
    )
    def test_saturation_factory_wrapped_format(
        self, saturation_cls: type[SaturationTransformation], lookup_name: str
    ) -> None:
        """Test that factory function works with wrapped format."""
        saturation = saturation_cls()
        data = saturation.to_dict()

        # Factory should handle wrapped format
        restored = saturation_from_dict(data)
        assert isinstance(restored, saturation_cls)
        assert restored.lookup_name == lookup_name

    @pytest.mark.parametrize(
        "lookup_name",
        [
            "logistic",
            "inverse_scaled_logistic",
            "tanh",
            "tanh_baselined",
            "michaelis_menten",
            "hill",
            "hill_sigmoid",
            "root",
            "no_saturation",
        ],
    )
    def test_saturation_factory_flat_backward_compat(self, lookup_name: str) -> None:
        """Test that factory function handles old flat format."""
        old_data = {
            "lookup_name": lookup_name,
            "prefix": "saturation",
            "priors": {},
        }

        # Factory should still work with flat format
        saturation = saturation_from_dict(old_data)
        assert isinstance(saturation, SaturationTransformation)
        assert saturation.lookup_name == lookup_name


class TestBasisWrappedFormat:
    """Test wrapped format serialization for Basis classes."""

    @pytest.mark.parametrize(
        "basis_cls,lookup_name",
        [
            (GaussianBasis, "gaussian"),
            (HalfGaussianBasis, "half_gaussian"),
            (AsymmetricGaussianBasis, "asymmetric_gaussian"),
        ],
    )
    def test_basis_to_dict_wrapped_format(
        self, basis_cls: type[Basis], lookup_name: str
    ) -> None:
        """Test that to_dict() returns wrapped format for all basis classes."""
        basis = basis_cls()
        data = basis.to_dict()

        # Check wrapped format structure
        assert isinstance(data, dict)
        assert "class" in data
        assert data["class"] == basis_cls.__name__
        assert "data" in data
        assert isinstance(data["data"], dict)

        # Check that data contains expected fields
        assert "lookup_name" in data["data"]
        assert data["data"]["lookup_name"] == lookup_name
        assert "prefix" in data["data"]
        assert data["data"]["prefix"] == "basis"

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

        # Load from wrapped format
        restored = basis_cls.from_dict(data)
        assert isinstance(restored, basis_cls)
        assert restored.lookup_name == basis.lookup_name

    @pytest.mark.parametrize(
        "basis_cls,lookup_name",
        [
            (GaussianBasis, "gaussian"),
            (HalfGaussianBasis, "half_gaussian"),
            (AsymmetricGaussianBasis, "asymmetric_gaussian"),
        ],
    )
    def test_basis_from_dict_flat_backward_compat(
        self, basis_cls: type[Basis], lookup_name: str
    ) -> None:
        """Test backward compatibility with flat format."""
        old_data = {
            "lookup_name": lookup_name,
            "prefix": "basis",
            "priors": {},
        }

        # Should still load (backward compatibility)
        basis = basis_cls.from_dict(old_data)
        assert isinstance(basis, basis_cls)
        assert basis.lookup_name == lookup_name

    @pytest.mark.parametrize(
        "basis_cls",
        [
            GaussianBasis,
            HalfGaussianBasis,
            AsymmetricGaussianBasis,
        ],
    )
    def test_basis_json_roundtrip(self, basis_cls: type[Basis]) -> None:
        """Test JSON serialization round-trip for basis classes."""
        basis = basis_cls()
        data = basis.to_dict()

        # Serialize to JSON
        json_str = json.dumps(data)
        restored_data = json.loads(json_str)

        # Deserialize
        restored = basis_cls.from_dict(restored_data)
        assert isinstance(restored, basis_cls)
        assert restored.lookup_name == basis.lookup_name

    @pytest.mark.parametrize(
        "basis_cls,lookup_name",
        [
            (GaussianBasis, "gaussian"),
            (HalfGaussianBasis, "half_gaussian"),
            (AsymmetricGaussianBasis, "asymmetric_gaussian"),
        ],
    )
    def test_basis_factory_wrapped_format(
        self, basis_cls: type[Basis], lookup_name: str
    ) -> None:
        """Test that factory function works with wrapped format."""
        from pymc_marketing.mmm.events import basis_from_dict

        basis = basis_cls()
        data = basis.to_dict()

        # Factory should handle wrapped format
        restored = basis_from_dict(data)
        assert isinstance(restored, basis_cls)
        assert restored.lookup_name == lookup_name

    @pytest.mark.parametrize(
        "lookup_name",
        [
            "gaussian",
            "half_gaussian",
            "asymmetric_gaussian",
        ],
    )
    def test_basis_factory_flat_backward_compat(self, lookup_name: str) -> None:
        """Test that factory function handles old flat format."""
        from pymc_marketing.mmm.events import basis_from_dict

        old_data = {
            "lookup_name": lookup_name,
            "prefix": "basis",
            "priors": {},
        }

        # Factory should still work with flat format
        basis = basis_from_dict(old_data)
        assert isinstance(basis, Basis)
        assert basis.lookup_name == lookup_name
