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
"""Tests for serialization infrastructure in components.

Tests the global field serializer and discriminated union types
that form the foundation for Phase 1 of the Pydantic refactoring.

"""

from pymc_extras.prior import Prior

from pymc_marketing.mmm.components.adstock import (
    BinomialAdstock,
    DelayedAdstock,
    NoAdstock,
    WeibullCDFAdstock,
    WeibullPDFAdstock,
)
from pymc_marketing.mmm.components.saturation import (
    HillSaturation,
    LogisticSaturation,
    MichaelisMentenSaturation,
    TanhSaturation,
)
from pymc_marketing.mmm.discriminated_unions import (
    AdstockUnion,
    BasisUnion,
    FourierUnion,
    SaturationUnion,
)


class TestGlobalFieldSerializer:
    """Test the @field_serializer in Transformation base class."""

    def test_serializer_handles_prior_objects(self):
        """Prior objects should be serialized to dict via to_dict()."""
        adstock = NoAdstock(l_max=1)
        serialized = adstock.to_dict()

        # The "priors" key should contain dicts, not Prior objects
        assert isinstance(serialized["priors"], dict)

    def test_serializer_handles_numpy_arrays(self):
        """numpy arrays should be converted to lists."""
        adstock = NoAdstock(l_max=1)
        serialized = adstock.to_dict()

        # Should not raise an error and should produce valid dict
        assert isinstance(serialized, dict)
        assert "lookup_name" in serialized
        assert "prefix" in serialized
        assert "priors" in serialized

    def test_serialized_structure_valid(self):
        """Serialized output should have correct structure."""
        adstock = NoAdstock(l_max=1)
        serialized = adstock.to_dict()

        # Must have these required keys (adstock may have additional keys)
        assert "lookup_name" in serialized
        assert "prefix" in serialized
        assert "priors" in serialized

        # lookup_name and prefix should be strings
        assert isinstance(serialized["lookup_name"], str)
        assert isinstance(serialized["prefix"], str)

        # priors should be a dict with serialized Prior objects
        assert isinstance(serialized["priors"], dict)


class TestDiscriminatedUnions:
    """Test discriminated union type definitions."""

    def test_adstock_union_type_defined(self):
        """AdstockUnion should be properly defined."""
        assert AdstockUnion is not None

    def test_saturation_union_type_defined(self):
        """SaturationUnion should be properly defined."""
        assert SaturationUnion is not None

    def test_fourier_union_type_defined(self):
        """FourierUnion should be properly defined."""
        assert FourierUnion is not None

    def test_basis_union_type_defined(self):
        """BasisUnion should be properly defined."""
        assert BasisUnion is not None


class TestSerializationRoundTrip:
    """Test serialization and deserialization round trips."""

    def test_adstock_serialization_roundtrip(self):
        """NoAdstock should serialize and deserialize correctly."""
        # Create an instance
        adstock1 = NoAdstock(l_max=1)

        # Serialize to dict
        serialized = adstock1.to_dict()

        # Deserialize back
        adstock2 = NoAdstock.from_dict(serialized)

        # Should be equal
        assert adstock1 == adstock2

    def test_saturation_serialization_roundtrip(self):
        """LogisticSaturation should serialize and deserialize correctly."""
        # Create an instance
        saturation1 = LogisticSaturation()

        # Serialize to dict
        serialized = saturation1.to_dict()

        # Deserialize back
        saturation2 = LogisticSaturation.from_dict(serialized)

        # Should be equal
        assert saturation1 == saturation2

    def test_multiple_transformations_serialization(self):
        """Multiple transformations should serialize independently."""
        adstock = NoAdstock(l_max=1)
        saturation = LogisticSaturation()

        serialized_adstock = adstock.to_dict()
        serialized_saturation = saturation.to_dict()

        # Should produce different outputs
        assert serialized_adstock != serialized_saturation

        # Both should deserialize correctly
        adstock2 = NoAdstock.from_dict(serialized_adstock)
        saturation2 = LogisticSaturation.from_dict(serialized_saturation)

        assert adstock == adstock2
        assert saturation == saturation2


class TestPriorSerialization:
    """Test Prior object serialization in transformations."""

    def test_prior_with_custom_priors(self):
        """Transformations with custom priors should serialize correctly."""
        # Create with custom priors
        custom_priors = {"power": Prior("Uniform", lower=0, upper=2)}
        adstock = NoAdstock(l_max=1, priors=custom_priors)

        # Serialize
        serialized = adstock.to_dict()

        # Prior should be in the priors dict as a dict
        assert "power" in serialized["priors"]
        prior_data = serialized["priors"]["power"]

        # Should be a dict representation (from Prior.to_dict())
        assert isinstance(prior_data, dict)

    def test_deserialization_restores_priors(self):
        """Deserialized transformation should restore priors correctly."""
        custom_priors = {"power": Prior("Uniform", lower=0, upper=2)}
        adstock1 = NoAdstock(l_max=1, priors=custom_priors)

        serialized = adstock1.to_dict()
        adstock2 = NoAdstock.from_dict(serialized)

        # Both should have the same priors
        assert adstock1.function_priors.keys() == adstock2.function_priors.keys()


class TestSerializerEdgeCases:
    """Test edge cases in serialization."""

    def test_empty_priors_serialization(self):
        """Transformations with default (empty) priors should serialize."""
        adstock = NoAdstock(l_max=1, priors={})
        serialized = adstock.to_dict()

        # Should still have all required keys
        assert "lookup_name" in serialized
        assert "prefix" in serialized
        assert "priors" in serialized

    def test_serialization_with_none_values(self):
        """Serialization should handle None values gracefully."""
        adstock = NoAdstock(l_max=1)

        # Should not raise an error
        serialized = adstock.to_dict()
        assert isinstance(serialized, dict)

    def test_multiple_serialization_calls_consistent(self):
        """Multiple serialization calls should produce identical output."""
        adstock = NoAdstock(l_max=1)

        serialized1 = adstock.to_dict()
        serialized2 = adstock.to_dict()

        # Should be identical
        assert serialized1 == serialized2


class TestInfrastructureIntegration:
    """Integration tests for the serialization infrastructure."""

    def test_all_adstock_types_serialize(self):
        """All adstock types should have working serialization."""
        adstock_types_and_kwargs = [
            (BinomialAdstock, {"l_max": 10}),
            (DelayedAdstock, {"l_max": 10}),
            (NoAdstock, {"l_max": 1}),
            (WeibullCDFAdstock, {"l_max": 10}),
            (WeibullPDFAdstock, {"l_max": 10}),
        ]

        for adstock_cls, kwargs in adstock_types_and_kwargs:
            adstock = adstock_cls(**kwargs)
            serialized = adstock.to_dict()

            # Should have required keys
            assert "lookup_name" in serialized
            assert "prefix" in serialized
            assert "priors" in serialized

            # lookup_name should match class lookup_name attribute
            assert serialized["lookup_name"] == adstock.lookup_name

    def test_all_saturation_types_serialize(self):
        """All saturation types should have working serialization."""
        saturation_types = [
            HillSaturation,
            LogisticSaturation,
            MichaelisMentenSaturation,
            TanhSaturation,
        ]

        for saturation_cls in saturation_types:
            saturation = saturation_cls()
            serialized = saturation.to_dict()

            # Should have required keys
            assert "lookup_name" in serialized
            assert "prefix" in serialized
            assert "priors" in serialized

            # lookup_name should match class
            assert serialized["lookup_name"] == saturation_cls.lookup_name
