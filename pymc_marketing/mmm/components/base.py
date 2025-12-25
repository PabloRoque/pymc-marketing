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
"""Base class for adstock and saturation functions used in MMM.

Use the subclasses directly for custom transformations:

- Adstock Transformations: :class:`pymc_marketing.mmm.components.adstock.AdstockTransformation`
- Saturation Transformations: :class:`pymc_marketing.mmm.components.saturation.SaturationTransformation`

"""

import warnings
from collections.abc import Callable, Iterable
from copy import deepcopy
from inspect import signature
from typing import Any, ClassVar, TypeAlias

import numpy as np
import numpy.typing as npt
import pymc as pm
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pydantic import (
    BaseModel,
    ConfigDict,
    InstanceOf,
    PrivateAttr,
    field_serializer,
)
from pydantic._internal._model_construction import ModelMetaclass
from pymc.distributions.shape_utils import Dims
from pymc_extras.deserialize import deserialize
from pymc_extras.prior import Prior, VariableFactory, create_dim_handler
from pytensor import tensor as pt
from pytensor.tensor.variable import TensorVariable

from pymc_marketing.model_config import parse_model_config
from pymc_marketing.plot import (
    SelToString,
    plot_curve,
    plot_hdi,
    plot_samples,
)

# "x" for saturation, "time since exposure" for adstock
NON_GRID_NAMES: frozenset[str] = frozenset({"x", "time since exposure"})

SupportedPrior: TypeAlias = (
    InstanceOf[Prior]
    | float
    | InstanceOf[TensorVariable]
    | InstanceOf[VariableFactory]
    | list
    | InstanceOf[npt.NDArray[np.floating]]
)


class SerializableMixin:
    """Mixin providing unified serialization interface for Pydantic BaseModel classes.

    This mixin implements default to_dict() and from_dict() methods for classes
    that inherit from both BaseModel and SerializableMixin. It provides:

    - **to_dict()**: Serializes to wrapped format with custom field serializers
    - **from_dict()**: Deserializes from wrapped format only
    - **_get_field_serializers()**: Hook for custom field-level serialization
    - **serialize_prior()**: Helper for serializing Prior objects to dicts
    - **serialize_xarray()**: Helper for serializing xarray.DataArray to lists
    - **serialize_ndarray()**: Helper for serializing numpy arrays to lists
    - **serialize_dict_recursive()**: Helper for serializing dicts with complex values

    The wrapped format is: ``{"class": "ClassName", "data": {...}}``

    This pattern enables polymorphic deserialization via discriminated unions,
    allowing proper type resolution when loading models with different
    transformation implementations.

    Classes can override _get_field_serializers() to provide custom serialization
    for complex field types (Prior, xarray.DataArray, numpy arrays, etc.) without
    needing to implement custom to_dict/from_dict methods.

    Examples
    --------
    Simple usage (no custom serialization):

    >>> from pydantic import BaseModel
    >>> class MyTransformation(BaseModel, SerializableMixin):
    ...     param: str
    >>> obj = MyTransformation(param="value")
    >>> d = obj.to_dict()
    >>> d["class"]
    'MyTransformation'
    >>> d["data"]["param"]
    'value'
    >>> restored = MyTransformation.from_dict(d)
    >>> restored.param
    'value'

    With custom field serialization for Prior:

    >>> class ModelWithPrior(BaseModel, SerializableMixin):
    ...     prior: Prior
    ...
    ...     @classmethod
    ...     def _get_field_serializers(cls):
    ...         return {"prior": SerializableMixin.serialize_prior}

    With custom serialization for xarray.DataArray:

    >>> class ModelWithArray(BaseModel, SerializableMixin):
    ...     mask: xarray.DataArray
    ...
    ...     @classmethod
    ...     def _get_field_serializers(cls):
    ...         return {"mask": SerializableMixin.serialize_xarray}

    Notes
    -----
    Classes using this mixin must inherit from Pydantic BaseModel as well.
    Order matters: ``class Foo(BaseModel, SerializableMixin):`` works best.

    Serialization format must use wrapped format with "class" and "data" keys.
    Flat format is not supported.

    """

    @classmethod
    def _get_field_serializers(cls) -> dict[str, Callable[[Any], Any]]:
        """Get field-specific serializers for this class.

        Override this method in subclasses to provide custom serialization
        logic for complex field types that can't be serialized by standard
        Pydantic JSON serialization.

        Returns
        -------
        dict[str, Callable]
            Mapping of field name (str) to serializer function.
            Each serializer function receives the original field value
            (before model_dump) and returns a JSON-serializable version.

        Notes
        -----
        - Serializer functions should be pure (no side effects)
        - Serializers receive original field values from getattr(), not model_dump results
        - Default implementation returns empty dict (no custom serialization)
        - Serializers are applied in to_dict() after model_dump()

        Examples
        --------
        For a class with a Prior field:

        >>> @classmethod
        ... def _get_field_serializers(cls):
        ...     return {
        ...         "prior": SerializableMixin.serialize_prior,
        ...     }

        For a class with multiple complex fields:

        >>> @classmethod
        ... def _get_field_serializers(cls):
        ...     return {
        ...         "prior": SerializableMixin.serialize_prior,
        ...         "mask": SerializableMixin.serialize_xarray,
        ...         "data": SerializableMixin.serialize_ndarray,
        ...     }

        For a class with a dict of Priors:

        >>> @classmethod
        ... def _get_field_serializers(cls):
        ...     def serialize_priors_dict(priors):
        ...         return {
        ...             k: SerializableMixin.serialize_prior(v)
        ...             for k, v in priors.items()
        ...         }
        ...
        ...     return {"priors": serialize_priors_dict}
        """
        return {}

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary with class wrapper and custom field serialization.

        Wraps the Pydantic model_dump output with the class name and applies
        any custom field serializers defined by _get_field_serializers().

        This format is required for polymorphic deserialization and enables
        proper type resolution when loading models with different implementations.

        Returns
        -------
        dict[str, Any]
            Dictionary with structure:
            ``{"class": "ClassName", "data": {...}}``

        Notes
        -----
        Custom field serializers are applied to original field values
        (via getattr), not to the already-serialized values from model_dump.
        This is important for types like Prior that have custom to_dict methods.

        Examples
        --------
        >>> from pydantic import BaseModel
        >>> class MyModel(BaseModel, SerializableMixin):
        ...     value: int
        >>> obj = MyModel(value=42)
        >>> d = obj.to_dict()
        >>> d["class"]
        'MyModel'
        >>> d["data"]["value"]
        42

        """
        # Apply custom field serializers first to handle non-serializable types
        field_serializers = self._get_field_serializers()

        if field_serializers:
            # If we have custom serializers, we need to exclude those fields from model_dump
            # and apply the serializers separately
            # Type: ignore for accessing BaseModel's model_fields (mixin constraint)
            fields_to_serialize = set(field_serializers.keys()) & set(
                self.model_fields.keys()  # type: ignore[attr-defined]
            )
            data = self.model_dump(  # type: ignore[attr-defined]
                mode="json",  # type: ignore
                exclude=fields_to_serialize,
            )

            # Apply custom field serializers to original field values
            for field_name in fields_to_serialize:
                original_value = getattr(self, field_name)
                data[field_name] = field_serializers[field_name](original_value)
        else:
            # No custom serializers, use standard serialization
            data = self.model_dump(mode="json")  # type: ignore[attr-defined]

        return {
            "class": self.__class__.__name__,
            "data": data,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], strict: bool = True) -> Any:
        """Deserialize from dictionary in wrapped format.

        Expects data in wrapped format with "class" and "data" keys.
        Raises ValueError if format is invalid.

        Parameters
        ----------
        data : dict[str, Any]
            Serialized data in wrapped format: {"class": "ClassName", "data": {...}}
        strict : bool, optional
            Reserved for future use. Default is True.
            Currently always enforces wrapped format.

        Returns
        -------
        Any
            Deserialized instance of the class

        Raises
        ------
        ValueError
            If data is not in wrapped format with "class" and "data" keys

        Examples
        --------
        >>> from pydantic import BaseModel
        >>> class MyModel(BaseModel, SerializableMixin):
        ...     value: int
        >>> wrapped = {"class": "MyModel", "data": {"value": 42}}
        >>> obj = MyModel.from_dict(wrapped)
        >>> obj.value
        42

        """
        if "class" not in data or "data" not in data:
            raise ValueError(
                f"Invalid serialization format for {cls.__name__}. "
                f"Expected wrapped format: {{'class': '{cls.__name__}', 'data': {{...}}}} "
                f"but got: {data}"
            )
        return cls.model_validate(data["data"])  # type: ignore

    @staticmethod
    def serialize_prior(value: Any) -> Any:
        """Serialize a Prior object to dict or return value as-is.

        Helper method for field serializers that need to handle Prior objects.
        Prior objects have a to_dict() method that produces the serialization.

        Parameters
        ----------
        value : Any
            Value to serialize (may or may not be a Prior)

        Returns
        -------
        Any
            Serialized value (dict if value has to_dict method, original value otherwise)

        Examples
        --------
        >>> class Model(BaseModel, SerializableMixin):
        ...     prior: Prior
        ...
        ...     @classmethod
        ...     def _get_field_serializers(cls):
        ...         return {"prior": SerializableMixin.serialize_prior}
        """
        if hasattr(value, "to_dict") and callable(value.to_dict):
            return value.to_dict()
        return value

    @staticmethod
    def serialize_xarray(arr: Any) -> Any:
        """Serialize xarray.DataArray to list of values.

        Extracts the underlying numpy array from xarray.DataArray and converts
        to a nested list structure suitable for JSON serialization.

        Parameters
        ----------
        arr : Any
            xarray.DataArray or array-like object

        Returns
        -------
        list
            Nested list representation of array values

        Notes
        -----
        Uses arr.values.tolist() for xarray.DataArray (preferred)
        Falls back to np.asarray(arr).tolist() for other array-like objects.

        Examples
        --------
        >>> import xarray as xr
        >>> import numpy as np
        >>> arr = xr.DataArray([1, 2, 3], dims=["x"])
        >>> result = SerializableMixin.serialize_xarray(arr)
        >>> result
        [1, 2, 3]

        >>> class Model(BaseModel, SerializableMixin):
        ...     mask: xarray.DataArray
        ...
        ...     @classmethod
        ...     def _get_field_serializers(cls):
        ...         return {"mask": SerializableMixin.serialize_xarray}
        """
        import numpy as np

        if hasattr(arr, "values"):
            # xarray.DataArray - use .values to get numpy array
            return arr.values.tolist()
        # Fallback for other array-like objects
        return np.asarray(arr).tolist()

    @staticmethod
    def serialize_ndarray(arr: Any) -> list:
        """Serialize numpy.ndarray to list.

        Converts a numpy array to a nested list structure suitable for
        JSON serialization.

        Parameters
        ----------
        arr : Any
            numpy.ndarray or array-like object

        Returns
        -------
        list
            Nested list representation of array

        Notes
        -----
        Handles numpy arrays with any shape and dtype.
        Complex dtypes may not serialize to standard JSON.

        Examples
        --------
        >>> import numpy as np
        >>> arr = np.array([1, 2, 3])
        >>> result = SerializableMixin.serialize_ndarray(arr)
        >>> result
        [1, 2, 3]

        >>> class Model(BaseModel, SerializableMixin):
        ...     data: np.ndarray
        ...
        ...     @classmethod
        ...     def _get_field_serializers(cls):
        ...         return {"data": SerializableMixin.serialize_ndarray}
        """
        import numpy as np

        return np.asarray(arr).tolist()

    @staticmethod
    def serialize_dict_recursive(
        d: dict[str, Any],
        value_serializer: Callable[[Any], Any],
    ) -> dict[str, Any]:
        """Recursively serialize dict values using a provided serializer.

        Applies a serializer function to each value in a dictionary.
        Useful for dicts containing complex types like Prior or numpy arrays.

        Parameters
        ----------
        d : dict[str, Any]
            Dictionary to serialize
        value_serializer : Callable
            Function to apply to each value.
            Receives the value, returns the serialized version.

        Returns
        -------
        dict[str, Any]
            New dictionary with serialized values

        Notes
        -----
        - Only applies the serializer to top-level values
        - Use nested serializers for deeply nested structures
        - Original dict is not modified (immutable operation)

        Examples
        --------
        >>> from pymc_marketing.prior import Prior
        >>> priors = {
        ...     "mu": Prior("Normal", mu=0, sigma=1),
        ...     "sigma": Prior("HalfNormal", sigma=1),
        ... }
        >>> result = SerializableMixin.serialize_dict_recursive(
        ...     priors,
        ...     SerializableMixin.serialize_prior,
        ... )

        >>> class Model(BaseModel, SerializableMixin):
        ...     priors: dict[str, Prior]
        ...
        ...     @classmethod
        ...     def _get_field_serializers(cls):
        ...         def serialize_priors(priors_dict):
        ...             return SerializableMixin.serialize_dict_recursive(
        ...                 priors_dict,
        ...                 SerializableMixin.serialize_prior,
        ...             )
        ...
        ...         return {"priors": serialize_priors}
        """
        return {key: value_serializer(value) for key, value in d.items()}


class ParameterPriorException(Exception):
    """Error when the functions and specified priors don't match up."""

    def __init__(self, priors: set[str], parameters: set[str]) -> None:
        self.priors = priors
        self.parameters = parameters

        msg = "The function parameters and priors don't line up."

        if self.priors:
            msg = f"{msg} Missing default prior: {self.priors}."

        if self.parameters:
            msg = f"{msg} Missing function parameter: {self.parameters}."

        super().__init__(msg)


RESERVED_DATA_PARAMETER_NAMES = {"x", "data"}


class MissingDataParameter(Exception):
    """Error if the function doesn't have a data parameter."""

    def __init__(self) -> None:
        msg = (
            f"The function must have a data parameter."
            " The first parameter is assumed to be the data"
            f" with name being one of: {RESERVED_DATA_PARAMETER_NAMES}"
        )

        super().__init__(msg)


def index_variable(var, dims, idx) -> TensorVariable:
    """Index a variable based on the provided dimensions and index.

    Parameters
    ----------
    var : TensorVariable
        The variable to index.
    dims : tuple[str, ...]
        The dims of the variable.
    idx : dict[str, pt.TensorLike]
        The index to use for the variable.

    Returns
    -------
    TensorVariable
        The indexed variable.

    """
    return var[tuple(idx[dim] if dim in idx else slice(None) for dim in dims)]


class Transformation(BaseModel, SerializableMixin):
    """Base class for adstock and saturation functions.

    The subclasses will need to implement the following attributes:

    - function: The function that will be applied to the data.
    - prefix: The prefix for the variables that will be created.
    - default_priors: The default priors for the parameters of the function.

    In order to make a new saturation or adstock function, use the specific subclasses:

    - :class:`pymc_marketing.mmm.components.saturation.SaturationTransformation`
    - :class:`pymc_marketing.mmm.components.adstock.AdstockTransformation`

    View the documentation for those classes for more information.

    Parameters
    ----------
    priors : dict[str, Prior | float | TensorVariable | VariableFactory | list  | numpy array], optional
        Dictionary with the priors for the parameters of the function. The keys should be the
        parameter names and the values the priors. If not provided, it will use the default
        priors from the subclass.
    prefix : str, optional
        The prefix for the variables that will be created. If not provided, it will use the prefix
        from the subclass.

    """

    # Class variables (not Pydantic fields - using ClassVar to exclude from model)
    prefix: ClassVar[str]
    default_priors: ClassVar[dict[str, Prior]]
    function: ClassVar[Any]
    lookup_name: ClassVar[str]

    # Private attribute for internal use (not serialized)
    _function_priors: dict[str, Prior] = PrivateAttr(default_factory=dict)

    model_config = ConfigDict(extra="forbid")

    def __init__(
        self,
        priors: dict[str, SupportedPrior] | None = None,
        prefix: str | None = None,
        **kwargs,
    ) -> None:
        """Initialize Transformation with priors and prefix.

        Parameters
        ----------
        priors : dict[str, Prior | float | TensorVariable | VariableFactory | list | numpy array], optional
            Dictionary with the priors for the parameters of the function.
        prefix : str, optional
            The prefix for the variables that will be created.
        **kwargs
            Additional keyword arguments passed to BaseModel.__init__.

        """
        # Parse priors BEFORE initializing (but don't set yet)
        priors = priors or {}
        non_distributions = [
            key
            for key, value in priors.items()
            if not isinstance(value, Prior) and not isinstance(value, dict)
        ]
        parsed_priors = parse_model_config(priors, non_distributions=non_distributions)

        # Call parent init FIRST (this initializes Pydantic and PrivateAttr defaults)
        super().__init__(**kwargs)

        # NOW set the private attribute with parsed priors (after Pydantic initialization)
        object.__setattr__(self, "_function_priors", parsed_priors)

        # Set prefix ONLY if explicitly provided
        if prefix is not None:
            object.__setattr__(self, "prefix", prefix)

        # Apply defaults and validate
        default_priors = getattr(type(self), "default_priors", {})
        try:
            current_priors = object.__getattribute__(self, "_function_priors")
        except AttributeError:
            current_priors = {}
        self._function_priors = {**deepcopy(default_priors), **current_priors}

        # Run checks
        self._checks()

    def __repr__(self) -> str:
        """Representation of the transformation."""
        return (
            f"{self.__class__.__name__}("
            f"prefix={self.prefix!r}, "
            f"priors={self.function_priors}"
            ")"
        )

    @classmethod
    def _get_field_serializers(cls) -> dict[str, Callable[[Any], Any]]:
        """Get field serializers for Transformation.

        Returns empty dict since Transformation uses custom to_dict() with
        _serialize_value() helper for prior serialization. This method is
        provided for consistency with SerializableMixin interface.

        Returns
        -------
        dict[str, Callable]
            Empty dict (custom serialization handled in to_dict())

        Notes
        -----
        Transformation stores priors in _function_priors (PrivateAttr),
        not as a Pydantic model field, so the field serializers hook
        doesn't apply. Instead, to_dict() uses _serialize_value() helper
        to handle Prior serialization.
        """
        return {}

    def set_dims_for_all_priors(self, dims: Dims):
        """Set the dims for all priors.

        Convenience method to loop through all the priors and set the dims.

        Parameters
        ----------
        dims : Dims
            The dims for the priors.

        Returns
        -------
        Transformation
        """
        for prior in self._function_priors.values():
            prior.dims = dims

        return self

    def __getattribute__(self, name: str) -> Any:
        """Override to provide model_config as prefixed priors mapping.

        Parameters
        ----------
        name : str
            Attribute name.

        Returns
        -------
        Any
            Attribute value or prefixed priors for model_config.

        """
        if name == "model_config":
            try:
                # Get _function_priors - it's stored in instance.__dict__
                instance_dict = object.__getattribute__(self, "__dict__")
                if "_function_priors" in instance_dict:
                    function_priors = instance_dict["_function_priors"]
                    # Get variable_mapping to create prefixed dict
                    var_mapping = object.__getattribute__(self, "variable_mapping")
                    return {
                        variable_name: function_priors[parameter_name]
                        for parameter_name, variable_name in var_mapping.items()
                    }
            except (AttributeError, KeyError):
                pass
        return object.__getattribute__(self, name)

    def __getattr__(self, name: str) -> Any:
        """Support backward compatibility for function_priors property access.

        Parameters
        ----------
        name : str
            Attribute name.

        Returns
        -------
        Any
            Attribute value.

        """
        if name == "function_priors":
            return self._function_priors
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )

    def __setattr__(self, name: str, value: Any) -> None:
        """Support backward compatibility for function_priors property setting.

        Parameters
        ----------
        name : str
            Attribute name.
        value : Any
            Attribute value.

        """
        if name == "function_priors":
            priors = value or {}
            non_distributions = [
                key
                for key, value in priors.items()
                if not isinstance(value, Prior) and not isinstance(value, dict)
            ]
            priors = parse_model_config(priors, non_distributions=non_distributions)
            object.__setattr__(
                self, "_function_priors", {**deepcopy(self.default_priors), **priors}
            )
        else:
            object.__setattr__(self, name, value)

    @field_serializer("*", mode="wrap")
    def serialize_fields(self, value: Any, handler, info) -> Any:
        """Globally serialize Prior, TensorVariable, and ndarray objects.

        This serializer runs after Pydantic's default serialization and handles
        special types that need custom serialization:
        - Prior objects are serialized to dict via their to_dict() method
        - TensorVariable objects are evaluated and converted to float
        - ndarray objects are converted to lists

        Parameters
        ----------
        value : Any
            The field value to serialize.
        handler : Callable
            Pydantic's default serialization handler.
        info : SerializationInfo
            Metadata about the serialization context.

        Returns
        -------
        Any
            The serialized value.

        """
        serialized = handler(value)

        # Handle Prior objects
        if hasattr(serialized, "to_dict"):
            return serialized.to_dict()

        # Handle TensorVariable objects
        if isinstance(serialized, TensorVariable):
            return float(serialized.eval())

        # Handle numpy arrays
        if isinstance(serialized, np.ndarray):
            return serialized.tolist()

        return serialized

    def to_dict(self) -> dict[str, Any]:
        """Convert the transformation to a dictionary in wrapped format.

        Produces wrapped format with class metadata and serialized priors.
        Format: {"class": "ClassName", "version": 1, "data": {...}}

        The "data" section contains:
        - "lookup_name": The class lookup name (for type identification)
        - "prefix": The variable prefix
        - "priors": Dict of serialized priors

        Returns
        -------
        dict
            The dictionary defining the transformation with structure:
            {
                "class": "ClassName",
                "version": 1,
                "data": {
                    "lookup_name": "...",
                    "prefix": "...",
                    "priors": {...}
                }
            }

        Notes
        -----
        Uses _serialize_value() helper to handle complex prior types
        including Prior objects, numpy arrays, and other types.
        """
        return {
            "class": self.__class__.__name__,
            "version": 1,
            "data": {
                "lookup_name": self.lookup_name,
                "prefix": self.prefix,
                "priors": {
                    key: _serialize_value(value)
                    for key, value in self.function_priors.items()
                },
            },
        }

    def __eq__(self, other: Any) -> bool:
        """Check if two transformations are equal."""
        if not isinstance(other, self.__class__):
            return False

        return self.to_dict() == other.to_dict()

    @classmethod
    def from_dict(cls, data: dict[str, Any], strict: bool = True) -> "Transformation":
        """Deserialize from dictionary with defensive pattern.

        Supports both wrapped format (new) and flat format (backward compat):
        - Wrapped: {"class": "ClassName", "version": 1, "data": {...}}
        - Flat: {"lookup_name": "...", "prefix": "...", "priors": {...}}

        Parameters
        ----------
        data : dict
            Dictionary to deserialize in wrapped or flat format.
        strict : bool, optional
            Reserved for compatibility with SerializableMixin. Default is True.
            Currently not used - Transformation always uses defensive deserialization.

        Returns
        -------
        Transformation
            The deserialized transformation.

        """
        # Handle wrapped format: {"class": "ClassName", "version": 1, "data": {...}}
        if "class" in data and "data" in data:
            inner_data = (
                data["data"].copy() if isinstance(data["data"], dict) else data["data"]
            )
        # Handle flat format (backward compatibility): {"lookup_name": "...", ...}
        elif "lookup_name" in data or "prefix" in data:
            inner_data = data.copy() if isinstance(data, dict) else data
        else:
            # Unknown format
            inner_data = data.copy() if isinstance(data, dict) else data

        # Deserialize priors if present
        if "priors" in inner_data and isinstance(inner_data["priors"], dict):
            inner_data["priors"] = {
                key: deserialize(value) if isinstance(value, dict) else value
                for key, value in inner_data["priors"].items()
            }

        # Remove lookup_name (not a constructor parameter)
        inner_data.pop("lookup_name", None)

        # Filter to only valid model fields for this class OR known constructor params
        # This allows subclasses to receive their specific parameters (l_max, normalize, mode, etc.)
        # while preventing validation errors from unexpected keys
        valid_fields = set(cls.model_fields.keys())
        constructor_params = {
            "priors",
            "prefix",
        }  # known constructor params not in model_fields
        valid_fields.update(constructor_params)
        filtered_data = {k: v for k, v in inner_data.items() if k in valid_fields}

        # Create instance with filtered parameters
        instance = cls(**filtered_data)

        return instance

    def update_priors(self, priors: dict[str, Prior]) -> None:
        """Update the priors for a function after initialization.

        Uses {prefix}_{parameter_name} as the key for the priors instead of the parameter name
        in order to be used in the larger MMM.

        Parameters
        ----------
        priors : dict[str, Prior]
            Dictionary with the new priors for the parameters of the function.

        Examples
        --------
        Update the priors for a transformation after initialization.

        .. code-block:: python

            from pymc_marketing.mmm.components.base import Transformation
            from pymc_extras.prior import Prior


            class MyTransformation(Transformation):
                lookup_name: str = "my_transformation"
                prefix: str = "transformation"
                function = lambda x, lam: x * lam
                default_priors = {"lam": Prior("Gamma", alpha=3, beta=1)}


            transformation = MyTransformation()
            transformation.update_priors(
                {"transformation_lam": Prior("HalfNormal", sigma=1)},
            )

        """
        new_priors = {
            parameter_name: priors[variable_name]
            for parameter_name, variable_name in self.variable_mapping.items()
            if variable_name in priors
        }
        if not new_priors:
            available_priors = list(self.variable_mapping.values())
            warnings.warn(
                f"No priors were updated. Available parameters are {available_priors}",
                UserWarning,
                stacklevel=2,
            )

        self.function_priors.update(new_priors)

    @property
    def transformation_config(self) -> dict[str, Any]:
        """Mapping from variable name to prior for the model.

        This property provides backward compatibility access to model configuration
        through the original name. Use directly for new code.

        """
        return {
            variable_name: self._function_priors[parameter_name]
            for parameter_name, variable_name in self.variable_mapping.items()
        }

    def _checks(self) -> None:
        self._has_all_attributes()
        self._function_works_on_instances()
        self._has_defaults_for_all_arguments()

    def _has_all_attributes(self) -> None:
        if not hasattr(self, "prefix"):
            raise NotImplementedError("prefix must be implemented in the subclass")

        if not hasattr(self, "default_priors"):
            raise NotImplementedError(
                "default_priors must be implemented in the subclass"
            )

        if not hasattr(self, "function"):
            raise NotImplementedError("function must be implemented in the subclass")

        if not hasattr(self, "lookup_name"):
            raise NotImplementedError("lookup_name must be implemented in the subclass")

    def _has_defaults_for_all_arguments(self) -> None:
        function_signature = signature(self.function)

        # Remove the first one as assumed to be the data
        parameters_that_need_priors = set(
            list(function_signature.parameters.keys())[1:]
        )
        parameters_with_priors = set(self.default_priors.keys())

        missing_priors = parameters_that_need_priors - parameters_with_priors
        missing_parameters = parameters_with_priors - parameters_that_need_priors

        if missing_priors or missing_parameters:
            raise ParameterPriorException(missing_priors, missing_parameters)

    def _function_works_on_instances(self) -> None:
        class_function = self.__class__.function
        function_parameters = list(signature(class_function).parameters)

        is_method = function_parameters[0] == "self"
        data_parameter_idx = 1 if is_method else 0

        has_data_parameter = (
            function_parameters[data_parameter_idx] in RESERVED_DATA_PARAMETER_NAMES
        )
        if not has_data_parameter:
            raise MissingDataParameter()

        if is_method:
            return

        object.__setattr__(self, "function", class_function)  # type: ignore[misc]

    @property
    def variable_mapping(self) -> dict[str, str]:
        """Mapping from parameter name to variable name in the model."""
        return {
            parameter: f"{self.prefix}_{parameter}"
            for parameter in self.default_priors.keys()
        }

    @property
    def combined_dims(self) -> tuple[str, ...]:
        """Get the combined dims for all the parameters."""
        return tuple(self._infer_output_core_dims())

    def _infer_output_core_dims(self) -> tuple[str, ...]:
        parameter_dims = sorted(
            [
                (dims,) if isinstance(dims, str) else dims
                for dist in self.function_priors.values()
                if (dims := getattr(dist, "dims", None)) is not None
            ],
            key=len,
            reverse=True,
        )
        return tuple(list({str(dim): None for dims in parameter_dims for dim in dims}))

    def _create_distributions(
        self,
        dims: Dims | None = None,
        idx: dict[str, pt.TensorLike] | None = None,
    ) -> dict[str, TensorVariable]:
        if isinstance(dims, str):
            dims = (dims,)

        dims = dims or self.combined_dims
        if idx is not None:
            dims = ("N", *dims)

        dim_handler = create_dim_handler(dims)

        def create_variable(parameter_name: str, variable_name: str) -> TensorVariable:
            dist = self.function_priors[parameter_name]
            if not hasattr(dist, "create_variable"):
                return dist

            var = dist.create_variable(variable_name)

            dist_dims = dist.dims
            if idx is not None and any(dim in idx for dim in dist_dims):
                var = index_variable(var, dist.dims, idx)

                dist_dims = [dim for dim in dist_dims if dim not in idx]
                dist_dims = ("N", *dist_dims)

            return dim_handler(var, dist_dims)

        return {
            parameter_name: create_variable(parameter_name, variable_name)
            for parameter_name, variable_name in self.variable_mapping.items()
        }

    def sample_prior(
        self, coords: dict | None = None, **sample_prior_predictive_kwargs
    ) -> xr.Dataset:
        """Sample the priors for the transformation.

        Parameters
        ----------
        coords : dict, optional
            The coordinates for the associated with dims
        **sample_prior_predictive_kwargs
            Keyword arguments for the pm.sample_prior_predictive function.

        Returns
        -------
        xr.Dataset
            The dataset with the sampled priors.

        """
        coords = coords or {}
        dims = tuple(coords.keys())
        with pm.Model(coords=coords):
            self._create_distributions(dims=dims)
            return pm.sample_prior_predictive(**sample_prior_predictive_kwargs).prior

    def plot_curve(
        self,
        curve: xr.DataArray,
        n_samples: int = 10,
        hdi_probs: float | list[float] | None = None,
        random_seed: np.random.Generator | None = None,
        subplot_kwargs: dict | None = None,
        sample_kwargs: dict | None = None,
        hdi_kwargs: dict | None = None,
        axes: npt.NDArray[Axes] | None = None,
        same_axes: bool = False,
        colors: Iterable[str] | None = None,
        legend: bool | None = None,
        sel_to_string: SelToString | None = None,
    ) -> tuple[Figure, npt.NDArray[Axes]]:
        """Plot curve HDI and samples.

        Parameters
        ----------
        curve : xr.DataArray
            The curve to plot.
        n_samples : int, optional
            Number of samples
        hdi_probs : float | list[float], optional
            HDI probabilities. Defaults to None which uses arviz default for
            stats.ci_prob which is 94%
        random_seed : int | random number generator, optional
            Random number generator. Defaults to None
        subplot_kwargs : dict, optional
            Keyword arguments for plt.subplots
        sample_kwargs : dict, optional
            Keyword arguments for the plot_curve_sample function. Defaults to None.
        hdi_kwargs : dict, optional
            Keyword arguments for the plot_curve_hdi function. Defaults to None.
        axes : npt.NDArray[plt.Axes], optional
            The exact axes to plot on. Overrides any subplot_kwargs
        same_axes : bool, optional
            If the axes should be the same for all plots. Defaults to False.
        colors : Iterable[str], optional
            The colors to use for the plot. Defaults to None.
        legend : bool, optional
            If the legend should be shown. Defaults to None.
        sel_to_string : SelToString, optional
            The function to convert the selection to a string. Defaults to None.

        Returns
        -------
        tuple[plt.Figure, npt.NDArray[plt.Axes]]

        """
        return plot_curve(
            curve,
            non_grid_names=set(NON_GRID_NAMES),
            n_samples=n_samples,
            hdi_probs=hdi_probs,
            random_seed=random_seed,
            subplot_kwargs=subplot_kwargs,
            sample_kwargs=sample_kwargs,
            hdi_kwargs=hdi_kwargs,
            axes=axes,
            same_axes=same_axes,
            colors=colors,
            legend=legend,
            sel_to_string=sel_to_string,
        )

    def _sample_curve(
        self,
        var_name: str,
        parameters: xr.Dataset,
        x: pt.TensorLike,
        coords: dict[str, Any],
    ) -> xr.DataArray:
        output_core_dims = self._infer_output_core_dims()

        keys = list(coords.keys())
        if len(keys) != 1:
            msg = "The coords should only have one key."
            raise ValueError(msg)
        x_dim = keys[0]

        # Allow broadcasting
        x = np.expand_dims(
            x,
            axis=tuple(range(1, len(output_core_dims) + 1)),
        )

        coords.update(
            {
                dim: np.asarray(coord)
                for dim, coord in parameters.coords.items()
                if dim not in ["chain", "draw"]
            }
        )

        with pm.Model(coords=coords):
            pm.Deterministic(
                var_name,
                self.apply(x, dims=output_core_dims),
                dims=(x_dim, *output_core_dims),
            )

            return pm.sample_posterior_predictive(
                parameters,
                var_names=[var_name],
            ).posterior_predictive[var_name]

    def plot_curve_samples(
        self,
        curve: xr.DataArray,
        n: int = 10,
        rng: np.random.Generator | None = None,
        plot_kwargs: dict | None = None,
        subplot_kwargs: dict | None = None,
        axes: npt.NDArray[Axes] | None = None,
    ) -> tuple[Figure, npt.NDArray[Axes]]:
        """Plot samples from the curve.

        Parameters
        ----------
        curve : xr.DataArray
            The curve to plot.
        n : int, optional
            The number of samples to plot. Defaults to 10.
        rng : np.random.Generator, optional
            The random number generator to use. Defaults to None.
        plot_kwargs : dict, optional
            Keyword arguments for the DataFrame plot function. Defaults to None.
        subplot_kwargs : dict, optional
            Keyword arguments for plt.subplots
        axes : npt.NDArray[plt.Axes], optional
            The exact axes to plot on. Overrides any subplot_kwargs

        Returns
        -------
        tuple[plt.Figure, npt.NDArray[plt.Axes]]
        plt.Axes
            The axes with the plot.

        """
        return plot_samples(
            curve,
            non_grid_names=set(NON_GRID_NAMES),
            n=n,
            rng=rng,
            axes=axes,
            subplot_kwargs=subplot_kwargs,
            plot_kwargs=plot_kwargs,
        )

    def plot_curve_hdi(
        self,
        curve: xr.DataArray,
        hdi_kwargs: dict | None = None,
        plot_kwargs: dict | None = None,
        subplot_kwargs: dict | None = None,
        axes: npt.NDArray[Axes] | None = None,
    ) -> tuple[Figure, npt.NDArray[Axes]]:
        """Plot the HDI of the curve.

        Parameters
        ----------
        curve : xr.DataArray
            The curve to plot.
        hdi_kwargs : dict, optional
            Keyword arguments for the az.hdi function. Defaults to None.
        plot_kwargs : dict, optional
            Keyword arguments for the fill_between function. Defaults to None.
        subplot_kwargs : dict, optional
            Keyword arguments for plt.subplots
        axes : npt.NDArray[plt.Axes], optional
            The exact axes to plot on. Overrides any subplot_kwargs

        Returns
        -------
        tuple[plt.Figure, npt.NDArray[plt.Axes]]

        """
        return plot_hdi(
            curve,
            non_grid_names=set(NON_GRID_NAMES),
            axes=axes,
            subplot_kwargs=subplot_kwargs,
            plot_kwargs=plot_kwargs,
            hdi_kwargs=hdi_kwargs,
        )

    def apply(
        self,
        x: pt.TensorLike,
        dims: Dims | None = None,
        idx: dict[str, pt.TensorLike] | None = None,
    ) -> TensorVariable:
        """Call within a model context.

        Used internally of the MMM to apply the transformation to the data.

        Parameters
        ----------
        x : pt.TensorLike
            The data to be transformed.
        dims : str, sequence[str], optional
            The dims of the parameters. Defaults to None. Not the dims of the
            data!

        Returns
        -------
        pt.TensorVariable
            The transformed data.

        Examples
        --------
        Call the function for custom use-case

        .. code-block:: python

            import pymc as pm

            transformation = ...

            coords = {"channel": ["TV", "Radio", "Digital"]}
            with pm.Model(coords=coords):
                transformed_data = transformation.apply(data, dims="channel")

        """
        kwargs = self._create_distributions(dims=dims, idx=idx)
        return self.function(x, **kwargs)


def _serialize_value(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return value.to_dict()

    if isinstance(value, TensorVariable):
        value = value.eval()

    if isinstance(value, np.ndarray):
        return value.tolist()

    return value


class DuplicatedTransformationError(Exception):
    """Exception when a transformation is duplicated."""

    def __init__(self, name: str, lookup_name: str):
        self.name = name
        self.lookup_name = lookup_name
        super().__init__(f"Duplicate {name}. The name {lookup_name!r} already exists.")


def create_registration_meta(subclasses: dict[str, Any]) -> type[type]:
    """Create a metaclass for registering subclasses.

    Parameters
    ----------
    subclasses : dict[str, type[Transformation]]
        The subclasses to register.

    Returns
    -------
    type
        The metaclass for registering subclasses.

    """

    class RegistrationMeta(ModelMetaclass):
        def __new__(cls, name, bases, attrs):
            # Check if any base inherits from BaseModel
            # If not, use type.__new__ instead of ModelMetaclass.__new__
            is_basemodel_subclass = any(
                isinstance(base, type) and issubclass(base, BaseModel) for base in bases
            )

            if is_basemodel_subclass:
                new_cls = super().__new__(cls, name, bases, attrs)
            else:
                # For non-BaseModel classes, just use type.__new__
                new_cls = type.__new__(cls, name, bases, attrs)

            if "lookup_name" not in attrs:
                return new_cls

            base_name = bases[0].__name__

            lookup_name = attrs["lookup_name"]
            if lookup_name in subclasses:
                raise DuplicatedTransformationError(base_name, lookup_name)

            subclasses[lookup_name] = new_cls

            return new_cls

    return RegistrationMeta
