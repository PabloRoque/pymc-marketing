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
import pytest
from pymc_extras.prior import Prior

from pymc_marketing.mmm.builders.factories import locate
from pymc_marketing.mmm.mmm import MMM


@pytest.mark.parametrize(
    "qualname, expected",
    [
        pytest.param("pymc_marketing.mmm.MMM", MMM, id="alternative-import"),
        pytest.param("pymc_extras.prior.Prior", Prior, id="full-import"),
    ],
)
def test_locate(qualname, expected) -> None:
    assert locate(qualname) is expected
