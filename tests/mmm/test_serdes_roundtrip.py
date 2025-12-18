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
"""Serialization round-trip tests for MMM components after Pydantic v2 migration."""

import json
from pathlib import Path
from typing import Any

import pandas as pd
from pymc_extras.prior import Prior

from pymc_marketing.mmm import (
    GeometricAdstock,
    LogisticSaturation,
)
from pymc_marketing.mmm.additive_effect import (
    FourierEffect,
    LinearTrendEffect,
)
from pymc_marketing.mmm.events import EventEffect, GaussianBasis
from pymc_marketing.mmm.fourier import WeeklyFourier
from pymc_marketing.mmm.linear_trend import LinearTrend
from pymc_marketing.mmm.multidimensional import MMM


def _assert_mmm_attributes_equal(mmm1: MMM, mmm2: Any) -> None:
    """Assert that two MMM instances have equivalent configuration attributes.

    Leverages __eq__ implementations where available (Transformation classes).

    Parameters
    ----------
    mmm1 : MMM
        First MMM instance.
    mmm2 : Any
        Second MMM instance (may be ModelIO from load).

    """
    # Core configuration
    assert mmm1.date_column == mmm2.date_column
    assert mmm1.target_column == mmm2.target_column
    assert mmm1.channel_columns == mmm2.channel_columns
    assert mmm1.dims == mmm2.dims

    # Components - use __eq__ implementations from Transformation base class
    assert mmm1.adstock == mmm2.adstock, (
        f"Adstock mismatch: {mmm1.adstock.to_dict()} != {mmm2.adstock.to_dict()}"
    )
    assert mmm1.saturation == mmm2.saturation, (
        f"Saturation mismatch: {mmm1.saturation.to_dict()} != "
        f"{mmm2.saturation.to_dict()}"
    )

    # Flags and settings
    assert mmm1.adstock_first == mmm2.adstock_first
    assert mmm1.control_columns == mmm2.control_columns
    assert mmm1.yearly_seasonality == mmm2.yearly_seasonality


def test_adstock_serialization_roundtrip() -> None:
    """Test that AdstockTransformation serializes and deserializes correctly."""
    # Create adstock with custom priors
    adstock = GeometricAdstock(
        l_max=10,
        priors={"alpha": Prior("Beta", alpha=1.5, beta=3)},
    )

    # Serialize to dict
    adstock_dict = adstock.to_dict()
    assert isinstance(adstock_dict, dict)
    assert "lookup_name" in adstock_dict
    assert adstock_dict["lookup_name"] == "geometric"

    # Serialize to JSON
    json_str = json.dumps(adstock_dict)
    assert isinstance(json_str, str)

    # Deserialize from JSON
    json_dict = json.loads(json_str)
    adstock_restored = GeometricAdstock.from_dict(json_dict)

    # Verify properties are preserved
    assert adstock_restored.l_max == adstock.l_max
    assert adstock_restored.lookup_name == adstock.lookup_name


def test_saturation_serialization_roundtrip() -> None:
    """Test that SaturationTransformation serializes and deserializes correctly."""
    # Create saturation with custom priors
    saturation = LogisticSaturation(
        priors={
            "lam": Prior("Gamma", mu=2, sigma=1),
            "beta": Prior("Gamma", mu=1.5, sigma=1),
        }
    )

    # Serialize to dict
    saturation_dict = saturation.to_dict()
    assert isinstance(saturation_dict, dict)
    assert "lookup_name" in saturation_dict
    assert saturation_dict["lookup_name"] == "logistic"

    # Serialize to JSON
    json_str = json.dumps(saturation_dict)
    assert isinstance(json_str, str)

    # Deserialize from JSON
    json_dict = json.loads(json_str)
    saturation_restored = LogisticSaturation.from_dict(json_dict)

    # Verify properties are preserved
    assert saturation_restored.lookup_name == saturation.lookup_name


def test_gaussian_basis_serialization_roundtrip() -> None:
    """Test that GaussianBasis serializes and deserializes correctly."""
    # Create Gaussian basis with custom priors
    basis = GaussianBasis(
        priors={"sigma": Prior("Gamma", mu=7, sigma=1)},
    )

    # Serialize to dict
    basis_dict = basis.to_dict()
    assert isinstance(basis_dict, dict)
    assert "lookup_name" in basis_dict
    assert basis_dict["lookup_name"] == "gaussian"

    # Serialize to JSON
    json_str = json.dumps(basis_dict)
    assert isinstance(json_str, str)

    # Deserialize from JSON
    json_dict = json.loads(json_str)
    basis_restored = GaussianBasis.from_dict(json_dict)

    # Verify properties are preserved
    assert basis_restored.lookup_name == basis.lookup_name


def test_event_effect_serialization_roundtrip() -> None:
    """Test that EventEffect serializes and deserializes correctly."""
    # Create Gaussian basis
    basis = GaussianBasis(
        priors={"sigma": Prior("Gamma", mu=7, sigma=1)},
    )

    # Create event effect
    effect_size = Prior("Normal", mu=1, sigma=1)
    event_effect = EventEffect(
        basis=basis,
        effect_size=effect_size,
        dims=("event",),
    )

    # Serialize to dict
    effect_dict = event_effect.to_dict()
    assert isinstance(effect_dict, dict)
    assert "data" in effect_dict
    assert "basis" in effect_dict["data"]
    assert "effect_size" in effect_dict["data"]

    # Serialize to JSON
    json_str = json.dumps(effect_dict)
    assert isinstance(json_str, str)

    # Deserialize from JSON
    json_dict = json.loads(json_str)
    effect_restored = EventEffect.from_dict(json_dict["data"])

    # Verify properties are preserved
    assert effect_restored.dims == event_effect.dims
    assert effect_restored.basis.lookup_name == event_effect.basis.lookup_name


def test_multiple_components_serialization_integrity() -> None:
    """Test that multiple components can be serialized together without conflicts."""
    # Create multiple components
    adstock = GeometricAdstock(l_max=10)
    saturation = LogisticSaturation()
    basis = GaussianBasis()

    # Serialize all components
    adstock_dict = adstock.to_dict()
    saturation_dict = saturation.to_dict()
    basis_dict = basis.to_dict()

    # Create combined JSON
    combined = {
        "adstock": adstock_dict,
        "saturation": saturation_dict,
        "basis": basis_dict,
    }
    json_str = json.dumps(combined)

    # Deserialize and verify integrity
    loaded = json.loads(json_str)
    assert loaded["adstock"]["lookup_name"] == "geometric"
    assert loaded["saturation"]["lookup_name"] == "logistic"
    assert loaded["basis"]["lookup_name"] == "gaussian"

    # Verify each component can be restored independently
    adstock_restored = GeometricAdstock.from_dict(loaded["adstock"])
    saturation_restored = LogisticSaturation.from_dict(loaded["saturation"])
    basis_restored = GaussianBasis.from_dict(loaded["basis"])

    assert adstock_restored.lookup_name == "geometric"
    assert saturation_restored.lookup_name == "logistic"
    assert basis_restored.lookup_name == "gaussian"


def _create_multidim_data(toy_X, toy_y, dims_dict):
    """Helper to create multidimensional data.

    Parameters
    ----------
    toy_X : pd.DataFrame
        Base data with date and channels.
    toy_y : pd.Series
        Base target series.
    dims_dict : dict
        Dictionary mapping dimension name to list of values.
        E.g., {"region": ["North", "South"]}

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        X (with dims columns) and y replicated for each dimension combination.

    """
    # Build all combinations of dimensions
    dim_names = list(dims_dict.keys())
    dim_values = list(dims_dict.values())

    records = []
    y_list = []

    # Generate Cartesian product of all dimension values
    from itertools import product

    for dim_combo in product(*dim_values):
        # For each dimension combination, add all time periods
        for idx, row in toy_X.iterrows():
            record = row.to_dict()
            for dim_name, dim_val in zip(dim_names, dim_combo, strict=False):
                record[dim_name] = dim_val
            records.append(record)
            y_list.append(toy_y.iloc[idx])

    X_multi = pd.DataFrame(records)
    y_multi = pd.Series(y_list, name="y")

    return X_multi, y_multi


def test_multidimensional_mmm_basic_persistence(
    tmp_path, toy_X, toy_y, mock_pymc_sample
) -> None:
    """Test basic multidimensional MMM save/load persistence.

    Verifies that a multidimensional MMM model can be saved to a .nc file and
    loaded back with all core attributes preserved.

    Parameters
    ----------
    tmp_path : Path
        Temporary directory provided by pytest for test file storage.
    toy_X : pd.DataFrame
        Sample data frame with date and channel columns.
    toy_y : pd.Series
        Sample target series.
    mock_pymc_sample : fixture
        Mock PyMC sampling fixture.

    """
    # Arrange: Create multidimensional data
    X_multi, y_multi = _create_multidim_data(
        toy_X, toy_y, {"region": ["North", "South"]}
    )

    # Create multidimensional MMM with dims
    mmm = MMM(
        date_column="date",
        channel_columns=["channel_1", "channel_2"],
        target_column="y",
        dims=("region",),
        adstock=GeometricAdstock(l_max=10),
        saturation=LogisticSaturation(),
    )

    # Build and fit model
    mmm.build_model(X_multi, y_multi)
    mmm.fit(X=X_multi, y=y_multi, chains=1, draws=5, tune=5, progressbar=False)

    # Create save path
    save_path = str(tmp_path / "mmm_multidim_basic.nc")

    # Act: Save model
    mmm.save(save_path)
    assert Path(save_path).exists(), "Model file was not created"

    # Load model
    loaded_mmm = MMM.load(save_path)

    # Assert: Verify core attributes preserved with helper function
    _assert_mmm_attributes_equal(mmm, loaded_mmm)
    assert loaded_mmm.idata.attrs == mmm.idata.attrs


def test_multidimensional_mmm_fourier_effect_persistence(
    tmp_path, toy_X, toy_y, mock_pymc_sample
) -> None:
    """Test multidimensional MMM with FourierEffect persistence.

    Verifies that a multidimensional MMM with FourierEffect can be saved and
    loaded with all MuEffect metadata preserved in idata.attrs.

    Parameters
    ----------
    tmp_path : Path
        Temporary directory provided by pytest for test file storage.
    toy_X : pd.DataFrame
        Sample data frame with date and channel columns.
    toy_y : pd.Series
        Sample target series.
    mock_pymc_sample : fixture
        Mock PyMC sampling fixture.

    """
    # Arrange: Create multidimensional data
    X_multi, y_multi = _create_multidim_data(
        toy_X, toy_y, {"store": ["Store_A", "Store_B"]}
    )

    # Create multidimensional MMM
    mmm = MMM(
        date_column="date",
        channel_columns=["channel_1", "channel_2"],
        target_column="y",
        dims=("store",),
        adstock=GeometricAdstock(l_max=10),
        saturation=LogisticSaturation(),
    )

    # Create and add FourierEffect
    fourier = WeeklyFourier(n_order=3, prefix="weekly")
    fourier_effect = FourierEffect(fourier=fourier, date_dim_name="date")
    mmm.mu_effects.append(fourier_effect)

    # Build and fit model
    mmm.build_model(X_multi, y_multi)
    mmm.fit(X=X_multi, y=y_multi, chains=1, draws=5, tune=5, progressbar=False)

    # Create save path
    save_path = str(tmp_path / "mmm_fourier_effect.nc")

    # Act: Save model
    mmm.save(save_path)
    assert Path(save_path).exists(), "Model file was not created"

    # Load model
    loaded_mmm = MMM.load(save_path)

    # Assert: Verify FourierEffect metadata persisted in idata.attrs
    # Note: mu_effects objects themselves are not re-instantiated during load
    # (they are complex and require state), but their metadata is preserved
    mu_effects_meta = loaded_mmm.idata.attrs.get("mu_effects")
    assert mu_effects_meta is not None, "mu_effects metadata not found in idata.attrs"

    # Parse the JSON metadata
    import json

    mu_effects_data = json.loads(mu_effects_meta)
    assert len(mu_effects_data) == 1
    assert mu_effects_data[0].get("class") == "FourierEffect"
    # Verify fourier configuration is in the metadata
    assert "fourier" in mu_effects_data[0]
    fourier_data = mu_effects_data[0]["fourier"]
    assert "weekly" in fourier_data.get("data", {}).get("prefix", "")

    # Verify model attributes with helper function
    _assert_mmm_attributes_equal(mmm, loaded_mmm)


def test_multidimensional_mmm_linear_trend_effect_persistence(
    tmp_path, toy_X, toy_y, mock_pymc_sample
) -> None:
    """Test multidimensional MMM with LinearTrendEffect persistence.

    Verifies that a multidimensional MMM with LinearTrendEffect can be saved and
    loaded with all effect parameters preserved.

    Parameters
    ----------
    tmp_path : Path
        Temporary directory provided by pytest for test file storage.
    toy_X : pd.DataFrame
        Sample data frame with date and channel columns.
    toy_y : pd.Series
        Sample target series.
    mock_pymc_sample : fixture
        Mock PyMC sampling fixture.

    """
    # Arrange: Create multidimensional data
    X_multi, y_multi = _create_multidim_data(toy_X, toy_y, {"country": ["USA", "CAN"]})

    # Create multidimensional MMM
    mmm = MMM(
        date_column="date",
        channel_columns=["channel_1", "channel_2"],
        target_column="y",
        dims=("country",),
        adstock=GeometricAdstock(l_max=10),
        saturation=LogisticSaturation(),
    )

    # Create and add LinearTrendEffect
    linear_trend = LinearTrend(
        n_changepoints=2,
        include_intercept=True,
        priors={},
    )
    trend_effect = LinearTrendEffect(
        trend=linear_trend,
        prefix="trend",
        date_dim_name="date",
    )
    mmm.mu_effects.append(trend_effect)

    # Build and fit model
    mmm.build_model(X_multi, y_multi)
    mmm.fit(X=X_multi, y=y_multi, chains=1, draws=5, tune=5, progressbar=False)

    # Create save path
    save_path = str(tmp_path / "mmm_linear_trend_effect.nc")

    # Act: Save model
    mmm.save(save_path)
    assert Path(save_path).exists(), "Model file was not created"

    # Load model
    loaded_mmm = MMM.load(save_path)

    # Assert: Verify LinearTrendEffect metadata persisted in idata.attrs
    # Note: mu_effects objects themselves are not re-instantiated during load
    mu_effects_meta = loaded_mmm.idata.attrs.get("mu_effects")
    assert mu_effects_meta is not None, "mu_effects metadata not found in idata.attrs"

    # Parse the JSON metadata
    import json

    mu_effects_data = json.loads(mu_effects_meta)
    assert len(mu_effects_data) == 1
    assert mu_effects_data[0].get("class") == "LinearTrendEffect"
    assert mu_effects_data[0]["prefix"] == "trend"

    # Verify model attributes with helper function
    _assert_mmm_attributes_equal(mmm, loaded_mmm)


def test_multidimensional_mmm_event_additive_effect_persistence(
    tmp_path, toy_X, toy_y, mock_pymc_sample
) -> None:
    """Test multidimensional MMM with EventAdditiveEffect persistence.

    Verifies that a multidimensional MMM with events (EventAdditiveEffect) can be
    saved and loaded with all event metadata preserved.

    Parameters
    ----------
    tmp_path : Path
        Temporary directory provided by pytest for test file storage.
    toy_X : pd.DataFrame
        Sample data frame with date and channel columns.
    toy_y : pd.Series
        Sample target series.
    mock_pymc_sample : fixture
        Mock PyMC sampling fixture.

    """
    # Arrange: Create multidimensional data
    X_multi, y_multi = _create_multidim_data(toy_X, toy_y, {"region": ["East", "West"]})

    # Create event dataframe
    dates = X_multi["date"].unique()
    df_events = pd.DataFrame(
        {
            "name": ["event_1", "event_2"],
            "start_date": [dates[5], dates[15]],
            "end_date": [dates[7], dates[17]],
        }
    )
    df_events["start_date"] = pd.to_datetime(df_events["start_date"])
    df_events["end_date"] = pd.to_datetime(df_events["end_date"])

    # Create multidimensional MMM
    mmm = MMM(
        date_column="date",
        channel_columns=["channel_1", "channel_2"],
        target_column="y",
        dims=("region",),
        adstock=GeometricAdstock(l_max=10),
        saturation=LogisticSaturation(),
    )

    # Create and add event effect
    event_effect = EventEffect(
        basis=GaussianBasis(),
        effect_size=Prior("Normal", mu=0, sigma=1, dims="event"),
        dims=("event",),
    )
    mmm.add_events(df_events=df_events, prefix="event", effect=event_effect)

    # Build and fit model
    mmm.build_model(X_multi, y_multi)
    mmm.fit(X=X_multi, y=y_multi, chains=1, draws=5, tune=5, progressbar=False)

    # Create save path
    save_path = str(tmp_path / "mmm_event_effect.nc")

    # Act: Save model
    mmm.save(save_path)
    assert Path(save_path).exists(), "Model file was not created"

    # Load model
    loaded_mmm = MMM.load(save_path)

    # Assert: Verify EventAdditiveEffect metadata persisted in idata.attrs
    # Note: mu_effects objects themselves are not re-instantiated during load
    mu_effects_meta = loaded_mmm.idata.attrs.get("mu_effects")
    assert mu_effects_meta is not None, "mu_effects metadata not found in idata.attrs"

    # Parse the JSON metadata
    import json

    mu_effects_data = json.loads(mu_effects_meta)
    assert len(mu_effects_data) == 1
    assert mu_effects_data[0].get("class") == "EventAdditiveEffect"
    assert mu_effects_data[0]["prefix"] == "event"

    # Verify model attributes with helper function
    _assert_mmm_attributes_equal(mmm, loaded_mmm)


def test_multidimensional_mmm_combined_mu_effects_persistence(
    tmp_path, toy_X, toy_y, mock_pymc_sample
) -> None:
    """Test multidimensional MMM with combined MuEffects persistence.

    Verifies that a multidimensional MMM with multiple MuEffect types (FourierEffect,
    LinearTrendEffect, EventAdditiveEffect) can be saved and loaded with all effects
    and metadata preserved.

    Parameters
    ----------
    tmp_path : Path
        Temporary directory provided by pytest for test file storage.
    toy_X : pd.DataFrame
        Sample data frame with date and channel columns.
    toy_y : pd.Series
        Sample target series.
    mock_pymc_sample : fixture
        Mock PyMC sampling fixture.

    """
    # Arrange: Create multidimensional data
    X_multi, y_multi = _create_multidim_data(toy_X, toy_y, {"geo": ["GEO_1", "GEO_2"]})

    # Create event dataframe
    dates = X_multi["date"].unique()
    df_events = pd.DataFrame(
        {
            "name": ["promo_1"],
            "start_date": [dates[10]],
            "end_date": [dates[12]],
        }
    )
    df_events["start_date"] = pd.to_datetime(df_events["start_date"])
    df_events["end_date"] = pd.to_datetime(df_events["end_date"])

    # Create multidimensional MMM
    mmm = MMM(
        date_column="date",
        channel_columns=["channel_1", "channel_2"],
        target_column="y",
        dims=("geo",),
        adstock=GeometricAdstock(l_max=10),
        saturation=LogisticSaturation(),
    )

    # Add FourierEffect
    fourier = WeeklyFourier(n_order=2, prefix="weekly")
    fourier_effect = FourierEffect(fourier=fourier, date_dim_name="date")
    mmm.mu_effects.append(fourier_effect)

    # Add LinearTrendEffect
    linear_trend = LinearTrend(
        n_changepoints=1,
        include_intercept=True,
        priors={},
    )
    trend_effect = LinearTrendEffect(
        trend=linear_trend,
        prefix="trend",
        date_dim_name="date",
    )
    mmm.mu_effects.append(trend_effect)

    # Add EventAdditiveEffect
    event_effect = EventEffect(
        basis=GaussianBasis(),
        effect_size=Prior("Normal", mu=0, sigma=1, dims="promo"),
        dims=("promo",),
    )
    mmm.add_events(df_events=df_events, prefix="promo", effect=event_effect)

    # Build and fit model
    mmm.build_model(X_multi, y_multi)
    mmm.fit(X=X_multi, y=y_multi, chains=1, draws=5, tune=5, progressbar=False)

    # Create save path
    save_path = str(tmp_path / "mmm_combined_effects.nc")

    # Act: Save model
    mmm.save(save_path)
    assert Path(save_path).exists(), "Model file was not created"

    # Load model
    loaded_mmm = MMM.load(save_path)

    # Assert: Verify all effects metadata persisted in idata.attrs
    # Note: mu_effects objects themselves are not re-instantiated during load
    mu_effects_meta = loaded_mmm.idata.attrs.get("mu_effects")
    assert mu_effects_meta is not None, "mu_effects metadata not found in idata.attrs"

    # Parse the JSON metadata
    import json

    mu_effects_data = json.loads(mu_effects_meta)
    assert len(mu_effects_data) == 3, (
        f"Expected 3 effects metadata, got {len(mu_effects_data)}"
    )

    # Verify effect types in metadata
    effect_classes = [eff.get("class") for eff in mu_effects_data]
    assert "FourierEffect" in effect_classes
    assert "LinearTrendEffect" in effect_classes
    assert "EventAdditiveEffect" in effect_classes

    # Verify model attributes with helper function
    _assert_mmm_attributes_equal(mmm, loaded_mmm)
