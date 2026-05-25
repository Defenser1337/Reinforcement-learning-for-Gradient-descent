import numpy as np
import hashlib
from typing import Optional

def get_rng(seed : Optional[int], location_name):
    if seed is None:
        return np.random.default_rng()

    local_seed = int(hashlib.md5(location_name.encode()).hexdigest(), 16) & 0xFFFFFFFF
    combined_seed = (local_seed + int(seed)) & 0xFFFFFFFF
    return np.random.default_rng(combined_seed)