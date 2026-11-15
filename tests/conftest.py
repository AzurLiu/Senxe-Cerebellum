import pytest

from core.neurons import SIMULATOR_ENV_VAR


@pytest.fixture(autouse=True)
def _explicit_simulator_mode(monkeypatch):
    """The test suite runs against the cl-sdk replay simulator.

    Simulator mode is an explicit opt-in, so every test selects it here.
    Tests that exercise hardware behaviour override or delete the variable.
    """

    monkeypatch.setenv(SIMULATOR_ENV_VAR, "1")
