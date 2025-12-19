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
"""Discriminated unions for Pydantic serialization.

This module defines discriminated union types for all polymorphic classes
in the MMM module. These unions enable automatic type dispatch during
deserialization without custom registration functions.

Each union uses a discriminator field (e.g., "lookup_name", "frequency")
to determine which concrete type to instantiate.

Example
-------
```python
from pymc_marketing.mmm.discriminated_unions import AdstockUnion

# Automatic dispatch based on lookup_name field
adstock = AdstockUnion.model_validate(
    {"lookup_name": "geometric", "prefix": "adstock", "priors": {...}}
)
# Returns: GeometricAdstock instance
```

Usage in Deserialization
------------------------
Instead of custom from_dict() methods and manual dispatch:

```python
# Before (custom logic)
@classmethod
def from_dict(cls, data):
    lookup_name = data.pop("lookup_name")
    cls = REGISTRY[lookup_name]
    return cls(**data)


# After (Pydantic discriminated union)
obj = AdstockUnion.model_validate(data)  # Automatic dispatch
```

Registering with pymc_extras.deserialize
-----------------------------------------
Each union should be registered with pymc_extras.deserialize:

```python
from pymc_extras.deserialize import register_deserialization

register_deserialization(
    is_type=lambda d: "lookup_name" in d and d["lookup_name"] in ADSTOCK_NAMES,
    deserialize=AdstockUnion.model_validate,
)
```

This enables seamless deserialization across the entire codebase.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

from pydantic import Discriminator

if TYPE_CHECKING:
    from pymc_marketing.mmm.components.adstock import (
        BinomialAdstock,
        DelayedAdstock,
        GeometricAdstock,
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
    from pymc_marketing.mmm.events import (
        AsymmetricGaussianBasis,
        GaussianBasis,
        HalfGaussianBasis,
    )
    from pymc_marketing.mmm.fourier import (
        MonthlyFourier,
        WeeklyFourier,
        YearlyFourier,
    )
    from pymc_marketing.special_priors import (
        LaplacePrior,
        LogNormalPrior,
    )

__all__ = [
    "AdstockUnion",
    "BasisUnion",
    "FourierUnion",
    "SaturationUnion",
    "SpecialPriorUnion",
]


# ============================================================================
# ADSTOCK TRANSFORMATIONS
# ============================================================================
AdstockUnion = Annotated[
    "BinomialAdstock | GeometricAdstock | DelayedAdstock | WeibullPDFAdstock | WeibullCDFAdstock | NoAdstock",
    Discriminator("lookup_name"),
]
"""Discriminated union for all adstock transformation types.

Discriminator: lookup_name
  - "binomial": BinomialAdstock
  - "geometric": GeometricAdstock
  - "delayed": DelayedAdstock
  - "weibull_pdf": WeibullPDFAdstock
  - "weibull_cdf": WeibullCDFAdstock
  - "no_adstock": NoAdstock

The lookup_name field automatically routes deserialization to the correct class.
"""


# ============================================================================
# SATURATION TRANSFORMATIONS
# ============================================================================
SaturationUnion = Annotated[
    "LogisticSaturation | HillSaturation | TanhSaturation | MichaelisMentenSaturation",
    Discriminator("lookup_name"),
]
"""Discriminated union for all saturation transformation types.

Discriminator: lookup_name
  - "logistic": LogisticSaturation
  - "hill": HillSaturation
  - "tanh": TanhSaturation
  - "michaelis_menten": MichaelisMentenSaturation

The lookup_name field automatically routes deserialization to the correct class.
"""


# ============================================================================
# FOURIER SEASONALITY COMPONENTS
# ============================================================================
FourierUnion = Annotated[
    "YearlyFourier | MonthlyFourier | WeeklyFourier",
    Discriminator("frequency"),
]
"""Discriminated union for all Fourier seasonality types.

Discriminator: frequency
  - "yearly": YearlyFourier
  - "monthly": MonthlyFourier
  - "weekly": WeeklyFourier

The frequency field automatically routes deserialization to the correct class.
"""


# ============================================================================
# BASIS FUNCTIONS (EVENT EFFECTS)
# ============================================================================
BasisUnion = Annotated[
    "GaussianBasis | HalfGaussianBasis | AsymmetricGaussianBasis",
    Discriminator("lookup_name"),
]
"""Discriminated union for all basis function types.

Discriminator: lookup_name
  - "gaussian": GaussianBasis
  - "half_gaussian": HalfGaussianBasis
  - "asymmetric_gaussian": AsymmetricGaussianBasis

The lookup_name field automatically routes deserialization to the correct class.
"""


# ============================================================================
# SPECIAL PRIORS
# ============================================================================
SpecialPriorUnion = Annotated[
    "LogNormalPrior | LaplacePrior",
    Discriminator("special_prior_type"),
]
"""Discriminated union for all special prior types.

Discriminator: special_prior_type
  - "LogNormalPrior": LogNormalPrior
  - "LaplacePrior": LaplacePrior

The special_prior_type field automatically routes deserialization to the correct class.

Note: This union can be extended by adding more prior types and updating
the discriminator field values.
"""
