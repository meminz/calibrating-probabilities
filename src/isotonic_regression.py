import numpy as np

class IsotonicCalibrator:
    """step-function calibrator, monotone by definition"""

    def fit(self, scores: np.ndarray, y: np.ndarray) -> IsotonicCalibrator:
        # PAV algorithm
        scores = np.asarray(scores, dtype=float)
        y = np.asarray(y, dtype=float)
        order = np.argsort(scores, kind="mergesort")
        scores_sorted, y_sorted = scores[order], y[order]

        # equal scores must share a fitted value
        unique_scores, start_idx, counts = np.unique(scores_sorted, return_index=True, return_counts=True)
        means = np.add.reduceat(y_sorted, start_idx) / counts
        one_per_unique = np.ones(len(unique_scores), dtype=int)

        # merge adjacent violators until monotone
        values, weights, unique_counts = [], [], []
        for value, weight, unique_n in zip(means, counts, one_per_unique):
            values.append(value)
            weights.append(weight)
            unique_counts.append(unique_n)

            while len(values) >= 2 and values[-2] > values[-1]:
                w2, v2 = weights.pop(), values.pop()
                n2 = unique_counts.pop()
                w1, v1 = weights.pop(), values.pop()
                n1 = unique_counts.pop()

                weights.append(w1 + w2)
                values.append((v1 * w1 + v2 * w2) / (w1 + w2))
                unique_counts.append(n1 + n2)

        # Block bookkeeping in units of unique scores
        # block k covers unique scores [ends_[k-1], ends_[k])
        # TODO use brain. this can def be done without unique lists
        self.thresholds_ = unique_scores
        self.ends_ = np.cumsum(unique_counts)
        self.block_values_ = np.asarray(values)
        return self

    def predict(self, scores: np.ndarray, clip: float = None) -> np.ndarray:
        scores = np.asarray(scores, dtype=float)
        uniques_below = np.searchsorted(self.thresholds_, scores, side="right")
        block_idx = np.searchsorted(self.ends_, uniques_below, side="left")
        prediction = self.block_values_[block_idx]

        if clip is not None:
            prediction = np.clip(prediction, clip, 1.0-clip)

        return prediction
