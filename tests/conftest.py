import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
B0_DIR = ROOT / "third_party/PIV-LiteFlowNet-en/caffe/models/PIV-LiteFlowNet-en"
B0_PROTO = B0_DIR / "PIV-LiteFlowNet-en_deploy.prototxt"
B0_WEIGHTS = B0_DIR / "PIV-LiteFlowNet-en.caffemodel"
needs_b0 = pytest.mark.skipif(not B0_WEIGHTS.is_file(), reason="B0 weights not fetched (scripts/fetch_third_party.sh)")


@pytest.fixture(scope="session")
def b0():
    if not B0_WEIGHTS.is_file():
        pytest.skip("B0 weights missing")
    from piv.models.baseline_adapter import LiteFlowNetBackend
    return LiteFlowNetBackend(B0_PROTO, B0_WEIGHTS)
