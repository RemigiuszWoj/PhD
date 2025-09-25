import random
from typing import List


def generate_taillard_instance(n: int, m: int, seed: int = 0) -> List[List[int]]:
    """Generate a Taillard benchmark-like flow shop instance."""
    rng = random.Random(seed)
    processing_times = [
        [rng.randint(1, 99) for _ in range(n)] for _ in range(m)  # n jobs  # m machines
    ]
    return processing_times
