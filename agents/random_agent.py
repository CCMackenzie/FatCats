
import numpy as np

class RandomAgent:

    def __init__(self, bid_probability: float = 0.2) -> None:
        if not (0.0 < bid_probability < 1.0):
            raise ValueError("bid_probability must be between 0 and 1 (exclusive)")
        self.bid_probability: float = bid_probability

    def act_batch(self, observations: np.ndarray, allow_multi_bid: bool) -> np.ndarray:
        if observations.ndim != 2:
            raise ValueError("observations must be a 2‑D array [batch, features]")

        batch_size: int = observations.shape[0]

        zeros_after_treat_idx = (observations[:, 1:] == 0).argmax(axis=1)
        max_hand_size: int = int(np.median(zeros_after_treat_idx))

        if allow_multi_bid:
            mask = np.random.rand(batch_size, max_hand_size) < self.bid_probability
            return mask.astype(np.int8)  #
        else:
            return np.random.randint(0, max_hand_size + 1, size=batch_size, dtype=np.int32)