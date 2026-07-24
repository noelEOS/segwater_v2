"""Builder over the REAL HPO-subset census files.

The census parquet/npy live under docs/ (gitignored, so absent on the VM / a
clean checkout) — skip when they are missing rather than fail. When present,
assert the four locked invariants and the internal structure (dense pair ids,
eligibility == mixed-count gate).
"""
import numpy as np
import pytest

from scripts.evaluation.build_hpo_strata_index import (
    DEFAULT_CENSUS,
    DEFAULT_META,
    build_index,
)

pytestmark = pytest.mark.skipif(
    not (DEFAULT_CENSUS.exists() and DEFAULT_META.exists()),
    reason="HPO-subset census files not present (docs/ is gitignored)",
)


def test_build_index_invariants():
    arrays, stats = build_index(DEFAULT_CENSUS, DEFAULT_META, min_mixed=5)

    assert stats["N"] == 29177
    assert stats["pairs"] == 686
    assert stats["mixed"] == 11027
    assert stats["eligible"] == 554
    assert stats["all_ignore"] == 0

    stratum_id = arrays["stratum_id"]
    pair_id = arrays["pair_id"]
    pair_names = arrays["pair_names"]
    pair_mixed_count = arrays["pair_mixed_count"]
    eligible = arrays["eligible"]

    # shapes / dtypes
    assert stratum_id.shape == (29177,) and stratum_id.dtype == np.int8
    assert pair_id.shape == (29177,) and pair_id.dtype == np.int32
    assert len(pair_names) == 686
    assert len(pair_mixed_count) == 686
    assert len(eligible) == 686

    # pair_id is dense 0..P-1 and appears in first-appearance order
    assert pair_id.min() == 0
    assert pair_id.max() == 685
    assert set(np.unique(pair_id)) == set(range(686))

    # eligibility is exactly the mixed-count gate
    assert np.array_equal(eligible, pair_mixed_count >= 5)

    # stratum vocab is {0,1,2} (all-ignore=3 has count 0 today)
    assert set(np.unique(stratum_id)) <= {0, 1, 2, 3}
    assert stats["pure_land"] + stats["pure_water"] + stats["mixed"] + stats["all_ignore"] == 29177
