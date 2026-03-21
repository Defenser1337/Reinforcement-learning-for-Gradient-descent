import numpy as np
import hashlib

def get_rng(seed, location_name):
    local_seed = int(hashlib.md5(location_name.encode()).hexdigest(), 16) & 0xFFFFFFFF
    combined_seed = (local_seed + int(seed)) & 0xFFFFFFFF
    return np.random.default_rng(combined_seed)