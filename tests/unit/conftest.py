"""Gemensamt för enhetstesterna. Inga tester här läser data/ eller results/ — de
använder bara config/ (som finns i repot), rena funktioner och små syntetiska
nätverk, så att de kan köras i CI på en ren checkout."""
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# pandas 2.x/3.x använder Arrow-strängar som standard; PyPSA/xarray stöder inte det
pd.options.future.infer_string = False


@pytest.fixture
def results_dir(tmp_path, monkeypatch):
    """En tom results/-mapp som settings läser källkörningar från."""
    from nordpsa import settings
    monkeypatch.setattr(settings, "RESULTS_DIR", tmp_path)
    return tmp_path
