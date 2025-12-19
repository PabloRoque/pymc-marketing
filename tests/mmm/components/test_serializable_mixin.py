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
"""Tests for SerializableMixin class."""

import pytest
from pydantic import BaseModel, InstanceOf
from pymc_extras.prior import Prior

from pymc_marketing.mmm.components.base import SerializableMixin
from pymc_marketing.mmm.fourier import YearlyFourier


class SimpleSerializable(BaseModel, SerializableMixin):
    """Simple test class using SerializableMixin."""

    name: str
    value: int


class SerializableWithPrior(BaseModel, SerializableMixin):
    """Test class with Prior field."""

    name: str
    prior: InstanceOf[Prior]


class TestSerializableMixinToDict:
    """Tests for SerializableMixin.to_dict() method."""

    def test_to_dict_format_with_class_and_data(self) -> None:
        """Test that to_dict returns wrapped format with class name and data."""
        obj = SimpleSerializable(name="test", value=42)
        result = obj.to_dict()

        assert "class" in result
        assert "data" in result
        assert result["class"] == "SimpleSerializable"
        assert result["data"]["name"] == "test"
        assert result["data"]["value"] == 42

    def test_to_dict_preserves_all_fields(self) -> None:
        """Test that to_dict includes all fields in data section."""
        obj = SimpleSerializable(name="example", value=100)
        result = obj.to_dict()

        assert set(result["data"].keys()) == {"name", "value"}

    def test_to_dict_with_multiple_fields(self) -> None:
        """Test to_dict with multiple fields of different types."""
        obj = SimpleSerializable(name="multi", value=999)
        result = obj.to_dict()

        assert result["data"]["name"] == "multi"
        assert result["data"]["value"] == 999
        assert isinstance(result["data"], dict)


class TestSerializableMixinFromDict:
    """Tests for SerializableMixin.from_dict() method."""

    def test_from_dict_with_wrapped_format(self) -> None:
        """Test from_dict handles wrapped format with class and data."""
        wrapped_data = {
            "class": "SimpleSerializable",
            "data": {"name": "test", "value": 42},
        }
        obj = SimpleSerializable.from_dict(wrapped_data)

        assert obj.name == "test"
        assert obj.value == 42

    def test_from_dict_rejects_flat_format(self) -> None:
        """Test from_dict raises ValueError for flat format."""
        flat_data = {"name": "test", "value": 42}
        with pytest.raises(ValueError, match="Invalid serialization format"):
            SimpleSerializable.from_dict(flat_data)

    def test_from_dict_raises_on_missing_class_key(self) -> None:
        """Test from_dict raises ValueError when 'class' key is missing."""
        data = {"data": {"name": "test", "value": 42}}
        with pytest.raises(ValueError, match="Invalid serialization format"):
            SimpleSerializable.from_dict(data)

    def test_from_dict_raises_on_missing_data_key(self) -> None:
        """Test from_dict raises ValueError when 'data' key is missing."""
        data = {"class": "SimpleSerializable", "name": "test", "value": 42}
        with pytest.raises(ValueError, match="Invalid serialization format"):
            SimpleSerializable.from_dict(data)

    def test_from_dict_extracts_data_from_wrapped(self) -> None:
        """Test from_dict correctly extracts data from wrapped format."""
        wrapped = {
            "class": "SimpleSerializable",
            "data": {"name": "wrapped", "value": 123},
        }
        obj = SimpleSerializable.from_dict(wrapped)

        assert obj.name == "wrapped"
        assert obj.value == 123


class TestSerializableMixinRoundtrip:
    """Tests for complete serialization roundtrips."""

    def test_to_dict_from_dict_roundtrip(self) -> None:
        """Test full roundtrip: object -> to_dict -> from_dict -> object."""
        original = SimpleSerializable(name="roundtrip", value=777)
        serialized = original.to_dict()
        restored = SimpleSerializable.from_dict(serialized)

        assert restored.name == original.name
        assert restored.value == original.value

    def test_roundtrip_preserves_all_data(self) -> None:
        """Test roundtrip preserves all field values."""
        original = SimpleSerializable(name="preserve", value=555)
        serialized = original.to_dict()
        restored = SimpleSerializable.from_dict(serialized)

        assert original == restored

    def test_multiple_roundtrips_consistent(self) -> None:
        """Test multiple roundtrips produce consistent results."""
        original = SimpleSerializable(name="multi", value=111)

        # First roundtrip
        dict1 = original.to_dict()
        obj1 = SimpleSerializable.from_dict(dict1)

        # Second roundtrip
        dict2 = obj1.to_dict()
        obj2 = SimpleSerializable.from_dict(dict2)

        # Third roundtrip
        dict3 = obj2.to_dict()
        obj3 = SimpleSerializable.from_dict(dict3)

        # All should be equal
        assert obj1 == obj2 == obj3 == original


class TestSerializableMixinWithFourier:
    """Tests for SerializableMixin with FourierBase (practical example)."""

    def test_fourier_inherits_to_dict_from_mixin(self) -> None:
        """Test FourierBase uses SerializableMixin.to_dict()."""
        fourier = YearlyFourier(n_order=3)
        result = fourier.to_dict()

        assert isinstance(result, dict)
        assert "class" in result
        assert "data" in result
        assert result["class"] == "YearlyFourier"

    def test_fourier_inherits_from_dict_from_mixin(self) -> None:
        """Test FourierBase uses SerializableMixin.from_dict()."""
        original = YearlyFourier(n_order=3)
        serialized = original.to_dict()
        restored = YearlyFourier.from_dict(serialized)

        assert restored.n_order == original.n_order
        assert restored.prefix == original.prefix

    def test_fourier_roundtrip_with_custom_prefix(self) -> None:
        """Test Fourier roundtrip preserves custom prefix."""
        original = YearlyFourier(n_order=2, prefix="custom_fourier")
        serialized = original.to_dict()
        restored = YearlyFourier.from_dict(serialized)

        assert restored.prefix == "custom_fourier"

    def test_fourier_roundtrip_with_custom_prior(self) -> None:
        """Test Fourier roundtrip preserves custom prior."""
        prior = Prior("Normal", mu=0, sigma=1)
        original = YearlyFourier(n_order=2, prior=prior)
        serialized = original.to_dict()
        restored = YearlyFourier.from_dict(serialized)

        # Prior should be restored (field_validator handles this)
        assert restored.prior is not None


class TestSerializableMixinSerializePrior:
    """Tests for SerializableMixin.serialize_prior() helper."""

    def test_serialize_prior_with_prior_object(self) -> None:
        """Test serialize_prior with Prior object that has to_dict."""
        prior = Prior("Beta")
        result = SerializableMixin.serialize_prior(prior)

        assert isinstance(result, dict)
        # Prior.to_dict() returns {"dist": "...", ...}
        assert "dist" in result or "class" in result

    def test_serialize_prior_with_non_prior_value(self) -> None:
        """Test serialize_prior with non-Prior value returns as-is."""
        value = 42
        result = SerializableMixin.serialize_prior(value)

        assert result == 42

    def test_serialize_prior_with_string(self) -> None:
        """Test serialize_prior with string value."""
        value = "test_string"
        result = SerializableMixin.serialize_prior(value)

        assert result == "test_string"

    def test_serialize_prior_with_dict_without_to_dict(self) -> None:
        """Test serialize_prior with dict that has no to_dict method."""
        value = {"key": "value"}
        result = SerializableMixin.serialize_prior(value)

        assert result == {"key": "value"}


class TestSerializableMixinEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_to_dict_then_from_dict_creates_new_instance(self) -> None:
        """Test that from_dict creates a new instance, not same object."""
        original = SimpleSerializable(name="test", value=42)
        serialized = original.to_dict()
        restored = SimpleSerializable.from_dict(serialized)

        assert restored == original
        assert restored is not original

    def test_wrapped_format_with_extra_keys_ignored(self) -> None:
        """Test from_dict ignores extra keys in wrapped format."""
        data = {
            "class": "SimpleSerializable",
            "data": {"name": "test", "value": 42},
            "extra_key": "should_be_ignored",
        }
        obj = SimpleSerializable.from_dict(data)

        assert obj.name == "test"
        assert obj.value == 42

    def test_mixin_works_with_inheritance(self) -> None:
        """Test SerializableMixin works correctly with class inheritance."""

        class DerivedSerializable(SimpleSerializable):
            """Subclass that inherits from SimpleSerializable."""

            extra_field: str = "default"

        obj = DerivedSerializable(name="derived", value=10, extra_field="custom")
        serialized = obj.to_dict()

        assert serialized["class"] == "DerivedSerializable"
        assert serialized["data"]["extra_field"] == "custom"

        restored = DerivedSerializable.from_dict(serialized)
        assert restored.extra_field == "custom"

    def test_strict_parameter_enforces_wrapped_format(self) -> None:
        """Test that strict=True enforces wrapped format (default behavior)."""
        wrapped = {
            "class": "SimpleSerializable",
            "data": {"name": "test", "value": 42},
        }
        obj = SimpleSerializable.from_dict(wrapped, strict=True)
        assert obj.name == "test"

    def test_error_message_shows_expected_format(self) -> None:
        """Test that error message shows the expected wrapped format."""
        flat_data = {"name": "test", "value": 42}
        try:
            SimpleSerializable.from_dict(flat_data)
            pytest.fail("Should have raised ValueError")
        except ValueError as e:
            assert "SimpleSerializable" in str(e)
            assert "class" in str(e)
            assert "data" in str(e)


class TestSerializableMixinConsistency:
    """Tests for consistency between different serialization methods."""

    def test_to_dict_data_section_matches_model_dump(self) -> None:
        """Test to_dict data section matches model_dump(mode='json')."""
        obj = SimpleSerializable(name="test", value=99)
        to_dict_result = obj.to_dict()
        model_dump_result = obj.model_dump(mode="json")

        # Data section should match model_dump
        assert to_dict_result["data"] == model_dump_result

    def test_wrapped_format_roundtrip_consistent(self) -> None:
        """Test wrapped format roundtrips produce consistent class names."""
        original = SimpleSerializable(name="test", value=42)

        # First roundtrip
        dict1 = original.to_dict()
        obj1 = SimpleSerializable.from_dict(dict1)
        dict2 = obj1.to_dict()

        # Class names should be consistent
        assert dict1["class"] == dict2["class"]
        assert dict1["class"] == "SimpleSerializable"
        assert dict2["class"] == "SimpleSerializable"
