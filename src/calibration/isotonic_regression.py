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

        # merge adjacent violators until monotone
        values, weights, block_boundaries = [], [], []
        for value, weight, score in zip(means, counts, unique_scores):
            values.append(value)
            weights.append(weight)
            block_boundaries.append(score)

            while len(values) >= 2 and values[-2] > values[-1]:
                w2, v2 = weights.pop(), values.pop()
                w1, v1 = weights.pop(), values.pop()

                weights.append(w1 + w2)
                values.append((v1 * w1 + v2 * w2) / (w1 + w2))
                block_boundaries.pop(-2) # remove interior boundary, keep right end

        self.block_values_ = np.asarray(values)
        self.block_boundaries_ = np.asarray(block_boundaries)
        return self

    def predict(self, scores: np.ndarray, clip: float = None) -> np.ndarray:
        scores = np.asarray(scores, dtype=float)

        # left-continuous because we store each block max value
        block_idx = np.searchsorted(self.block_boundaries_, scores, side="left")
        block_idx = np.clip(block_idx, 0, len(self.block_values_) - 1) # clamp
        prediction = self.block_values_[block_idx]

        if clip is not None:
            prediction = np.clip(prediction, clip, 1.0 - clip)

        return prediction
