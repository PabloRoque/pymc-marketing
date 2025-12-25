# Phase 5: Serialization Best Practices & Pattern Guide

**Status**: ✅ PATTERNS DOCUMENTED
**Date**: 2025-12-25
**Scope**: Transformation classes, Container classes, Specialized formats

---

## Executive Summary

This guide establishes serialization best practices based on Phase 4-5 analysis and implementation experience. It provides decision trees and code patterns for choosing the right serialization approach for different class types.

---

## Decision Tree: Choosing Serialization Strategy

```
START: Analyzing class for serialization needs
│
├─ Is this a Transformation implementation?
│  ├─ YES → Use FLAT FORMAT (registry dispatch)
│  │   - Inherits from Transformation base
│  │   - Lookup by lookup_name in registry
│  │   - Custom to_dict() extends parent
│  │   └─ EXAMPLE: Basis classes (GaussianBasis, etc.)
│  │
│  └─ NO → Continue to next question
│
├─ Is this a Container class (holds components)?
│  ├─ YES → Use WRAPPED FORMAT (type metadata)
│  │   - Wraps in {"class": "Name", "data": {...}}
│  │   - Explicit type metadata for polymorphism
│  │   - Implement SerializableMixin
│  │   └─ EXAMPLE: MediaTransformation, MediaConfig
│  │
│  └─ NO → Continue to next question
│
├─ Does this have deeply integrated custom format?
│  ├─ YES → Keep CUSTOM FORMAT (if proven effective)
│  │   - Only if format is mission-critical
│  │   - User impact if changed = very high
│  │   - Migration cost too high to justify change
│  │   └─ EXAMPLE: SpecialPrior
│  │
│  └─ NO → Default to WRAPPED FORMAT (SerializableMixin)
│
END: Format chosen
```

---

## Format Comparison Matrix

### 1. FLAT FORMAT (Registry-Based)

**Used For**: Transformation implementations
**Example**: Basis classes, custom adstock/saturation
**Identification**: Uses `lookup_name` field for type routing

#### Characteristics

| Aspect | Value |
|--------|-------|
| **Type Identifier** | `lookup_name` field value |
| **Dispatch Method** | Registry dictionary lookup |
| **Wrapper** | None (raw dict) |
| **Inheritance** | Extends parent class directly |
| **Complexity** | Low (simple extension pattern) |
| **Performance** | O(1) registry lookup + instantiation |
| **Flexibility** | Low (bound to registry) |
| **Backward Compat** | Excellent (proven pattern) |

#### Example Code

```python
# ✅ CORRECT: Flat format for Transformation subclass
class MyTransformation(Transformation):
    lookup_name = "my_transformation"
    prefix = "my"

    def __init__(self, custom_param, **kwargs):
        super().__init__(**kwargs)
        self.custom_param = custom_param

    def to_dict(self) -> dict[str, Any]:
        """Extend parent flat format."""
        return {
            **super().to_dict(),                    # lookup_name, prefix, priors
            "custom_param": self.custom_param,     # Additional field
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], strict: bool = True) -> "MyTransformation":
        """Deserialize from dict (handled by base class pattern)."""
        return super().from_dict(data, strict)

    default_priors = {
        "param": Prior("Gamma", mu=7, sigma=1),
    }

# Polymorphic dispatch via registry
MY_TRANSFORMATIONS = {
    "my_transformation": MyTransformation,
    # other subtypes...
}

def my_transformation_from_dict(data: dict) -> MyTransformation:
    """Factory function for polymorphic loading."""
    lookup_name = data.pop("lookup_name")
    cls = MY_TRANSFORMATIONS[lookup_name]
    if "priors" in data:
        data["priors"] = {k: deserialize(v) for k, v in data["priors"].items()}
    return cls(**data)
```

#### When to Use ✅

- Implementing Transformation subclasses
- Using registry-based polymorphic dispatch
- Extending parent transformation functionality
- Working with existing Transformation hierarchy

#### When NOT to Use ❌

- Building container classes (use wrapped format)
- When explicit type metadata needed beyond name
- When versioning/schema evolution needed (use wrapped)

---

### 2. WRAPPED FORMAT (Type Metadata)

**Used For**: Container classes, component holders
**Example**: MediaTransformation, MediaConfig, EventEffect
**Identification**: Has `"class"` field at top level

#### Characteristics

| Aspect | Value |
|--------|-------|
| **Type Identifier** | `"class"` field value |
| **Dispatch Method** | Type metadata + polymorphic deserialize() |
| **Wrapper** | `{"class": "Name", "data": {...}}` |
| **Inheritance** | SerializableMixin + BaseModel |
| **Complexity** | Medium (structured pattern) |
| **Performance** | O(1) lookup + polymorphic handling |
| **Flexibility** | High (supports versioning, metadata) |
| **Backward Compat** | Excellent (defensive deserialization) |

#### Example Code

```python
# ✅ CORRECT: Wrapped format for Container class
class MyContainer(BaseModel, SerializableMixin):
    """Container holding components."""

    component1: ComplexType
    component2: AnotherType
    dims: tuple[str, ...]

    @classmethod
    def _get_field_serializers(cls) -> dict[str, Callable[[Any], Any]]:
        """Define custom serializers for component fields."""
        return {
            "component1": SerializableMixin.serialize_complex_type,
            "component2": SerializableMixin.serialize_another_type,
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize to wrapped format."""
        inner = {
            "component1": (
                self.component1.to_dict()
                if isinstance(self.component1, SerializableMixin)
                else self.component1
            ),
            "component2": (
                self.component2.to_dict()
                if isinstance(self.component2, SerializableMixin)
                else self.component2
            ),
            "dims": self.dims,
        }
        return {
            "class": "MyContainer",
            "data": inner,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], strict: bool = True) -> "MyContainer":
        """Deserialize from wrapped or flat format (defensive)."""
        # Handle both wrapped and flat for backward compatibility
        if "class" in data and "data" in data:
            inner_data = data["data"]
        elif "component1" in data or "component2" in data:
            inner_data = data
        else:
            raise ValueError(f"Invalid format: {data}")

        inner_data = inner_data.copy()

        # Defensively deserialize component fields
        for key in ["component1", "component2"]:
            if key in inner_data and isinstance(inner_data[key], dict):
                inner_data[key] = deserialize(inner_data[key])

        # Validate dims if needed
        if "dims" in inner_data and isinstance(inner_data["dims"], list):
            inner_data["dims"] = tuple(inner_data["dims"])

        return cls.model_validate(inner_data)
```

#### When to Use ✅

- Building container classes (holds multiple components)
- Implementing BaseModel subclasses with complex fields
- When explicit type metadata beneficial
- Supporting polymorphic deserialization
- Planning for future schema versioning

#### When NOT to Use ❌

- Simple data classes (use Pydantic defaults)
- Transformation implementations (use flat format)
- When registry dispatch more appropriate
- Trying to maintain backward compatibility with old flat format (unless defensive)

---

### 3. CUSTOM FORMAT (Specialized)

**Used For**: Special cases with mission-critical serialization
**Example**: SpecialPrior, custom event handlers
**Identification**: Unique format not matching standard patterns

#### Characteristics

| Aspect | Value |
|--------|-------|
| **Type Identifier** | Custom field (varies) |
| **Dispatch Method** | Custom registry + specialized deserializer |
| **Wrapper** | Varies (e.g., `{"special_prior": ...}`) |
| **Inheritance** | Custom pattern (not SerializableMixin) |
| **Complexity** | High (custom handling required) |
| **Performance** | Custom (depends on implementation) |
| **Flexibility** | Varies |
| **Backward Compat** | Depends on custom implementation |

#### Example Code

```python
# ✅ CORRECT: Custom format when truly justified
class SpecialPrior(SerializableMixin):
    """Custom format for deeply integrated prior class."""

    def to_dict(self) -> dict[str, Any]:
        """Custom format (proven effective)."""
        return {
            "special_prior": self.__class__.__name__,  # Custom identifier
            "kwargs": {...},                            # Custom data
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], strict: bool = True) -> "SpecialPrior":
        """Custom deserialization logic."""
        name = data.get("special_prior")
        cls = SPECIAL_PRIOR_TYPES[name]
        return cls(**data.get("kwargs", {}))

def _is_special_prior(data: dict) -> bool:
    """Type check for registry."""
    return "special_prior" in data and data["special_prior"] in SPECIAL_PRIOR_TYPES

register_deserialization(
    is_type=_is_special_prior,
    deserialize=SpecialPrior.from_dict,
)
```

#### When to Use ✅

- **ONLY when**:
  1. Format is mission-critical (baked into artifacts)
  2. Migration cost is too high
  3. Custom approach clearly better than standard
  4. User impact of change would be severe

#### When NOT to Use ❌

- For new classes (use standard patterns)
- When wrapped/flat format would work
- Unless migration cost is genuinely prohibitive
- For convenience/preference (use standard patterns)

---

## Pattern Recognition Guide

### How to Identify Format Type

#### Flat Format Recognition
```python
# Characteristic: lookup_name or prefix at top level
{
    "lookup_name": "gaussian",
    "prefix": "basis",
    "priors": {...},
    "custom_field": "value",  # Optional custom fields
}
```

**Detection**: Check for `lookup_name` field

---

#### Wrapped Format Recognition
```python
# Characteristic: "class" and "data" wrapper
{
    "class": "MediaTransformation",
    "data": {
        "adstock": {...},
        "saturation": {...},
        "dims": (...),
    }
}
```

**Detection**: Check for `"class"` field at top level + `"data"` field

---

#### Custom Format Recognition
```python
# Characteristic: Unique structure not matching standard patterns
{
    "special_prior": "LogNormalPrior",
    "kwargs": {...},
}
```

**Detection**: Doesn't match flat or wrapped patterns; custom registry check

---

## Defensive Deserialization Pattern

### Problem
Different versions/sources may produce different serialization formats. Hard-coded parsers fail on format changes.

### Solution
Implement defensive from_dict() that handles multiple formats:

```python
@classmethod
def from_dict(cls, data: dict[str, Any], strict: bool = True) -> "ClassName":
    """Support multiple formats for robustness."""

    # Handle wrapped format: {"class": "ClassName", "data": {...}}
    if "class" in data and "data" in data:
        inner_data = data["data"]
    # Handle flat format: {"field1": ..., "field2": ...}
    elif "field1" in data or "field2" in data:
        inner_data = data
    # Unknown format
    else:
        raise ValueError(f"Unrecognized format: {data}")

    # Make a copy to avoid mutation
    inner_data = inner_data.copy() if isinstance(inner_data, dict) else inner_data

    # Deserialize nested objects
    for key in ["nested_field1", "nested_field2"]:
        if key in inner_data and isinstance(inner_data[key], dict):
            inner_data[key] = deserialize(inner_data[key])

    # Type conversions (JSON compatibility)
    if "tuple_field" in inner_data and isinstance(inner_data["tuple_field"], list):
        inner_data["tuple_field"] = tuple(inner_data["tuple_field"])

    return cls.model_validate(inner_data)
```

### Benefits ✅

- Handles both wrapped and flat formats
- Supports version migrations
- Polymorphic deserialization via deserialize()
- No breaking changes for users

### Used In ✅

- EventEffect.from_dict()
- MediaTransformation.from_dict()
- MediaConfig.from_dict()

---

## Testing Patterns for Serialization

### Pattern 1: Round-Trip Testing

```python
def test_class_serialization():
    """Test serialization round-trip."""
    # Create instance
    obj = MyClass(field1="value1", field2="value2")

    # Serialize
    serialized = obj.to_dict()

    # Deserialize
    restored = MyClass.from_dict(serialized)

    # Verify equality
    assert restored.field1 == obj.field1
    assert restored.field2 == obj.field2
```

---

### Pattern 2: Format Verification

```python
def test_class_format():
    """Verify serialization format."""
    obj = MyClass(...)
    serialized = obj.to_dict()

    # Check wrapped format
    assert "class" in serialized
    assert serialized["class"] == "MyClass"
    assert "data" in serialized
    assert isinstance(serialized["data"], dict)
```

---

### Pattern 3: Backward Compatibility

```python
def test_backward_compatibility():
    """Test loading old format."""
    # Old flat format
    old_format = {
        "field1": "value1",
        "field2": "value2",
    }

    # Should still load (defensive deserialization)
    restored = MyClass.from_dict(old_format)
    assert restored.field1 == "value1"
    assert restored.field2 == "value2"
```

---

### Pattern 4: Nested Deserialization

```python
def test_nested_serialization():
    """Test nested component deserialization."""
    obj = Container(
        component=Component(param="value"),
        dims=("d1", "d2"),
    )

    serialized = obj.to_dict()
    restored = Container.from_dict(serialized)

    # Verify nested component restored correctly
    assert isinstance(restored.component, Component)
    assert restored.component.param == "value"
```

---

## Code Review Checklist

When reviewing serialization code:

- [ ] **Format Choice**: Is the format choice appropriate for the class type?
- [ ] **Round-Trip**: Do serialization round-trip tests exist?
- [ ] **Defensive**: Does from_dict() handle multiple formats?
- [ ] **Docstring**: Is the format documented in docstring?
- [ ] **Type Safety**: Are type hints on from_dict() signature?
- [ ] **Polymorphism**: Is polymorphic deserialization handled?
- [ ] **Edge Cases**: Are JSON compatibility conversions handled (list→tuple)?
- [ ] **Backward Compat**: Does defensive pattern maintain compatibility?
- [ ] **Field Serializers**: Are complex types in _get_field_serializers()?
- [ ] **Registry**: If registry dispatch, is registration in place?

---

## Common Mistakes & Fixes

### ❌ Mistake 1: Too-Strict Deserialization

```python
# ❌ WRONG: Only accepts wrapped format
@classmethod
def from_dict(cls, data: dict):
    if "class" not in data or "data" not in data:
        raise ValueError("Must have wrapped format")
    # ...
```

**Fix**: Use defensive pattern to handle both formats ✅

---

### ❌ Mistake 2: Not Converting JSON Types

```python
# ❌ WRONG: Doesn't handle JSON list→tuple conversion
@classmethod
def from_dict(cls, data: dict):
    # dims is a list from JSON but expects tuple
    return cls(dims=data["dims"])  # Will fail with "type is list not tuple"
```

**Fix**: Convert list to tuple in validator or from_dict() ✅

---

### ❌ Mistake 3: Forgetting Nested Deserialization

```python
# ❌ WRONG: Doesn't deserialize nested objects
@classmethod
def from_dict(cls, data: dict):
    return cls(**data)  # component is still a dict, not deserialized
```

**Fix**: Use deserialize() for nested fields ✅

```python
# ✅ CORRECT: Deserialize nested objects
for key in ["component1", "component2"]:
    if key in inner_data and isinstance(inner_data[key], dict):
        inner_data[key] = deserialize(inner_data[key])
```

---

### ❌ Mistake 4: Breaking Existing Format

```python
# ❌ WRONG: Changing format without migration path
class OldVersion:
    def to_dict(self):
        return {"flat": "format"}

class NewVersion:  # Breaking change!
    def to_dict(self):
        return {"class": "Name", "data": {...}}
```

**Fix**: Use defensive from_dict() or provide migration script ✅

---

## Summary of Best Practices

### ✅ DO

1. **Choose format based on class type** (transformation=flat, container=wrapped)
2. **Implement defensive from_dict()** (handles multiple formats)
3. **Add round-trip tests** (verify serialization fidelity)
4. **Handle JSON type conversions** (list→tuple, etc.)
5. **Document format in docstring** (explain structure)
6. **Use polymorphic deserialize()** (for nested types)
7. **Inherit patterns from similar classes** (consistency)
8. **Test edge cases** (overlapping fields, missing fields, etc.)

### ❌ DON'T

1. **Mix format patterns** (don't use wrapped for Transformation)
2. **Make from_dict() too strict** (breaks backward compatibility)
3. **Forget nested deserialization** (nested objects stay as dicts)
4. **Ignore JSON type conversions** (lists won't become tuples)
5. **Break existing serialization** (without migration plan)
6. **Create custom format without justification** (use standard patterns)
7. **Skip tests for serialization** (serialization is critical)
8. **Assume structure is self-documenting** (document format clearly)

---

## Future Considerations

### Schema Versioning
If format evolution becomes necessary:

```python
# Wrapped format allows adding version field
{
    "class": "ClassName",
    "version": 2,  # Can add version field
    "data": {...}
}
```

### Migration Patterns
For breaking changes, establish:
1. Migration script in version X
2. Support both formats in version X
3. Drop old format in version X+1

### Polymorphic Dispatch Evolution
Current approach (deserialize function + registry) scales well:
- O(1) lookup by name
- Extensible via registration
- Supports future versioning

---

## Conclusion

**Key Insight**: There is no one-size-fits-all serialization format.

- **Flat format** (registry dispatch) best for Transformation implementations
- **Wrapped format** (type metadata) best for Container classes
- **Custom format** only when truly necessary

Choose format based on class role, not consistency. Document choice clearly. Test round-trip thoroughly. Handle backward compatibility defensively.

---

**Document Status**: Complete and ready for reference
**Last Updated**: 2025-12-25
**Version**: 1.0 (Stable)
