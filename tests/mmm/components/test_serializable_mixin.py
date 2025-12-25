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

from typing import Any

import numpy as np
import pytest
import xarray as xr
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


class TestSerializeXarray:
    """Tests for SerializableMixin.serialize_xarray() helper."""

    def test_serialize_xarray_1d_array(self) -> None:
        """Test serialize_xarray with 1D xarray.DataArray."""
        arr = xr.DataArray([1, 2, 3], dims=["x"])
        result = SerializableMixin.serialize_xarray(arr)

        assert isinstance(result, list)
        assert result == [1, 2, 3]

    def test_serialize_xarray_2d_array(self) -> None:
        """Test serialize_xarray with 2D xarray.DataArray."""
        arr = xr.DataArray([[1, 2], [3, 4]], dims=["x", "y"])
        result = SerializableMixin.serialize_xarray(arr)

        assert isinstance(result, list)
        assert result == [[1, 2], [3, 4]]

    def test_serialize_xarray_with_named_dimensions(self) -> None:
        """Test serialize_xarray preserves values despite named dimensions."""
        arr = xr.DataArray(
            [1.5, 2.5, 3.5],
            dims=["time"],
            coords={"time": [0, 1, 2]},
        )
        result = SerializableMixin.serialize_xarray(arr)

        assert isinstance(result, list)
        assert result == [1.5, 2.5, 3.5]

    def test_serialize_xarray_with_float_values(self) -> None:
        """Test serialize_xarray with float values."""
        arr = xr.DataArray([1.1, 2.2, 3.3], dims=["x"])
        result = SerializableMixin.serialize_xarray(arr)

        assert result == [1.1, 2.2, 3.3]

    def test_serialize_xarray_fallback_to_numpy(self) -> None:
        """Test serialize_xarray fallback for numpy arrays."""
        numpy_arr = np.array([1, 2, 3])
        result = SerializableMixin.serialize_xarray(numpy_arr)

        assert isinstance(result, list)
        assert result == [1, 2, 3]

    def test_serialize_xarray_empty_array(self) -> None:
        """Test serialize_xarray with empty array."""
        arr = xr.DataArray([], dims=["x"])
        result = SerializableMixin.serialize_xarray(arr)

        assert result == []

    def test_serialize_xarray_single_value(self) -> None:
        """Test serialize_xarray with single value."""
        arr = xr.DataArray([42], dims=["x"])
        result = SerializableMixin.serialize_xarray(arr)

        assert result == [42]


class TestSerializeNdarray:
    """Tests for SerializableMixin.serialize_ndarray() helper."""

    def test_serialize_ndarray_1d(self) -> None:
        """Test serialize_ndarray with 1D numpy array."""
        arr = np.array([1, 2, 3])
        result = SerializableMixin.serialize_ndarray(arr)

        assert isinstance(result, list)
        assert result == [1, 2, 3]

    def test_serialize_ndarray_2d(self) -> None:
        """Test serialize_ndarray with 2D numpy array."""
        arr = np.array([[1, 2], [3, 4]])
        result = SerializableMixin.serialize_ndarray(arr)

        assert isinstance(result, list)
        assert result == [[1, 2], [3, 4]]

    def test_serialize_ndarray_3d(self) -> None:
        """Test serialize_ndarray with 3D numpy array."""
        arr = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        result = SerializableMixin.serialize_ndarray(arr)

        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0] == [[1, 2], [3, 4]]
        assert result[1] == [[5, 6], [7, 8]]

    def test_serialize_ndarray_float_dtype(self) -> None:
        """Test serialize_ndarray with float dtype."""
        arr = np.array([1.1, 2.2, 3.3])
        result = SerializableMixin.serialize_ndarray(arr)

        assert result == [1.1, 2.2, 3.3]

    def test_serialize_ndarray_int_dtype(self) -> None:
        """Test serialize_ndarray with int dtype."""
        arr = np.array([1, 2, 3], dtype=np.int32)
        result = SerializableMixin.serialize_ndarray(arr)

        assert result == [1, 2, 3]

    def test_serialize_ndarray_empty(self) -> None:
        """Test serialize_ndarray with empty array."""
        arr = np.array([])
        result = SerializableMixin.serialize_ndarray(arr)

        assert result == []

    def test_serialize_ndarray_single_value(self) -> None:
        """Test serialize_ndarray with single value array."""
        arr = np.array([42])
        result = SerializableMixin.serialize_ndarray(arr)

        assert result == [42]

    def test_serialize_ndarray_bool_dtype(self) -> None:
        """Test serialize_ndarray with boolean dtype."""
        arr = np.array([True, False, True])
        result = SerializableMixin.serialize_ndarray(arr)

        assert result == [True, False, True]


class TestSerializeDictRecursive:
    """Tests for SerializableMixin.serialize_dict_recursive() helper."""

    def test_serialize_dict_recursive_with_integers(self) -> None:
        """Test serialize_dict_recursive with integer values."""
        d = {"a": 1, "b": 2, "c": 3}
        result = SerializableMixin.serialize_dict_recursive(
            d,
            lambda x: x * 2,
        )

        assert result == {"a": 2, "b": 4, "c": 6}

    def test_serialize_dict_recursive_with_strings(self) -> None:
        """Test serialize_dict_recursive with string values."""
        d = {"a": "hello", "b": "world"}
        result = SerializableMixin.serialize_dict_recursive(
            d,
            lambda x: x.upper(),
        )

        assert result == {"a": "HELLO", "b": "WORLD"}

    def test_serialize_dict_recursive_preserves_keys(self) -> None:
        """Test that serialize_dict_recursive preserves all keys."""
        d = {"key1": 1, "key2": 2, "key3": 3}
        result = SerializableMixin.serialize_dict_recursive(
            d,
            lambda x: x + 10,
        )

        assert set(result.keys()) == {"key1", "key2", "key3"}

    def test_serialize_dict_recursive_empty_dict(self) -> None:
        """Test serialize_dict_recursive with empty dict."""
        d = {}
        result = SerializableMixin.serialize_dict_recursive(
            d,
            lambda x: x,
        )

        assert result == {}

    def test_serialize_dict_recursive_with_prior_serializer(self) -> None:
        """Test serialize_dict_recursive with Prior serializer."""
        priors = {
            "mu": Prior("Normal", mu=0, sigma=1),
            "sigma": Prior("HalfNormal", sigma=1),
        }
        result = SerializableMixin.serialize_dict_recursive(
            priors,
            SerializableMixin.serialize_prior,
        )

        # Should have same keys
        assert set(result.keys()) == {"mu", "sigma"}
        # Values should be dicts (serialized Priors)
        assert isinstance(result["mu"], dict)
        assert isinstance(result["sigma"], dict)

    def test_serialize_dict_recursive_single_entry(self) -> None:
        """Test serialize_dict_recursive with single entry."""
        d = {"only_key": 42}
        result = SerializableMixin.serialize_dict_recursive(
            d,
            lambda x: x * 10,
        )

        assert result == {"only_key": 420}

    def test_serialize_dict_recursive_does_not_modify_original(self) -> None:
        """Test that serialize_dict_recursive doesn't modify original dict."""
        d = {"a": 1, "b": 2}
        original_copy = d.copy()
        SerializableMixin.serialize_dict_recursive(
            d,
            lambda x: x * 100,
        )

        # Original should be unchanged
        assert d == original_copy


class TestGetFieldSerializersHook:
    """Tests for SerializableMixin._get_field_serializers() hook."""

    def test_default_get_field_serializers_returns_empty_dict(self) -> None:
        """Test that default _get_field_serializers returns empty dict."""
        serializers = SimpleSerializable._get_field_serializers()

        assert isinstance(serializers, dict)
        assert len(serializers) == 0

    def test_override_get_field_serializers(self) -> None:
        """Test that subclass can override _get_field_serializers."""

        class CustomSerializable(BaseModel, SerializableMixin):
            """Test class with custom field serializers."""

            value: int

            @classmethod
            def _get_field_serializers(cls) -> dict:
                return {"value": lambda x: x * 2}

        serializers = CustomSerializable._get_field_serializers()

        assert "value" in serializers
        assert callable(serializers["value"])

    def test_field_serializers_applied_in_to_dict(self) -> None:
        """Test that field serializers are applied in to_dict()."""

        class CustomSerializable(BaseModel, SerializableMixin):
            """Test class with custom field serializers."""

            value: int

            @classmethod
            def _get_field_serializers(cls) -> dict:
                return {"value": lambda x: x * 2}

        obj = CustomSerializable(value=21)
        result = obj.to_dict()

        # Value should be doubled by the serializer
        assert result["data"]["value"] == 42

    def test_multiple_field_serializers(self) -> None:
        """Test class with multiple field serializers."""

        class MultiFieldSerializable(BaseModel, SerializableMixin):
            """Test class with multiple custom field serializers."""

            a: int
            b: int
            c: int

            @classmethod
            def _get_field_serializers(cls) -> dict:
                return {
                    "a": lambda x: x * 2,
                    "b": lambda x: x * 3,
                    "c": lambda x: x * 4,
                }

        obj = MultiFieldSerializable(a=1, b=1, c=1)
        result = obj.to_dict()

        assert result["data"]["a"] == 2
        assert result["data"]["b"] == 3
        assert result["data"]["c"] == 4

    def test_field_serializer_with_missing_field(self) -> None:
        """Test that serializer for non-existent field is safely ignored."""

        class CustomSerializable(BaseModel, SerializableMixin):
            """Test class with serializer for non-existent field."""

            value: int

            @classmethod
            def _get_field_serializers(cls) -> dict:
                return {
                    "value": lambda x: x * 2,
                    "nonexistent": lambda x: x,  # This field doesn't exist
                }

        obj = CustomSerializable(value=10)
        result = obj.to_dict()

        # Should not raise an error, serializer is just skipped
        assert result["data"]["value"] == 20


class TestToDictWithFieldSerializers:
    """Tests for to_dict with custom field serializers."""

    def test_to_dict_applies_serializers_to_original_values(self) -> None:
        """Test that to_dict applies serializers to original field values."""

        class SerializableWithArray(BaseModel, SerializableMixin):
            """Test class with numpy array field."""

            data: Any

            model_config = {"arbitrary_types_allowed": True}

            @classmethod
            def _get_field_serializers(cls) -> dict:
                return {"data": SerializableMixin.serialize_ndarray}

        arr = np.array([1, 2, 3])
        obj = SerializableWithArray(data=arr)
        result = obj.to_dict()

        # Data should be serialized to list
        assert result["data"]["data"] == [1, 2, 3]
        assert isinstance(result["data"]["data"], list)

    def test_to_dict_wraps_with_class_name(self) -> None:
        """Test that to_dict includes class name in output."""

        class CustomClass(BaseModel, SerializableMixin):
            """Test class."""

            value: int

        obj = CustomClass(value=42)
        result = obj.to_dict()

        assert result["class"] == "CustomClass"

    def test_to_dict_structure_has_class_and_data_keys(self) -> None:
        """Test that to_dict output has required keys."""

        class CustomClass(BaseModel, SerializableMixin):
            """Test class."""

            value: int

        obj = CustomClass(value=42)
        result = obj.to_dict()

        assert "class" in result
        assert "data" in result
        assert len(result) == 2  # Should have exactly these two keys


class TestFieldSerializersRoundtrip:
    """Tests for roundtrips with field serializers."""

    def test_roundtrip_with_numpy_array_serializer(self) -> None:
        """Test roundtrip with numpy array field using serializer."""

        class SerializableArray(BaseModel, SerializableMixin):
            """Test class with numpy array field."""

            name: str
            data: Any

            model_config = {"arbitrary_types_allowed": True}

            @classmethod
            def _get_field_serializers(cls) -> dict:
                return {"data": SerializableMixin.serialize_ndarray}

        arr = np.array([1.0, 2.0, 3.0])
        original = SerializableArray(name="test", data=arr)
        serialized = original.to_dict()

        # Verify serialization
        assert serialized["data"]["data"] == [1.0, 2.0, 3.0]

        # Roundtrip back
        restored = SerializableArray.from_dict(serialized)

        assert restored.name == "test"
        # Data should be a list now (deserialized)
        assert restored.data == [1.0, 2.0, 3.0]

    def test_roundtrip_with_xarray_serializer(self) -> None:
        """Test roundtrip with xarray.DataArray field using serializer."""

        class SerializableXarray(BaseModel, SerializableMixin):
            """Test class with xarray field."""

            name: str
            data: Any

            model_config = {"arbitrary_types_allowed": True}

            @classmethod
            def _get_field_serializers(cls) -> dict:
                return {"data": SerializableMixin.serialize_xarray}

        arr = xr.DataArray([1, 2, 3], dims=["x"])
        original = SerializableXarray(name="test", data=arr)
        serialized = original.to_dict()

        # Verify serialization
        assert serialized["data"]["data"] == [1, 2, 3]

        # Roundtrip back
        restored = SerializableXarray.from_dict(serialized)

        assert restored.name == "test"
        assert restored.data == [1, 2, 3]


class TestBackwardCompatibility:
    """Tests for backward compatibility with existing behavior."""

    def test_classes_without_custom_serializers_work(self) -> None:
        """Test that existing classes without custom serializers still work."""
        obj = SimpleSerializable(name="test", value=42)
        serialized = obj.to_dict()
        restored = SimpleSerializable.from_dict(serialized)

        assert restored == obj

    def test_empty_field_serializers_dict_doesnt_affect_to_dict(self) -> None:
        """Test that empty field serializers dict doesn't change behavior."""

        class NoSerializers(BaseModel, SerializableMixin):
            """Test class with no custom serializers."""

            value: int

        obj = NoSerializers(value=42)
        result = obj.to_dict()

        assert result["data"]["value"] == 42

    def test_fourier_backward_compatibility(self) -> None:
        """Test that YearlyFourier still works as before."""
        fourier = YearlyFourier(n_order=3)
        serialized = fourier.to_dict()
        restored = YearlyFourier.from_dict(serialized)

        assert restored.n_order == fourier.n_order
