"""
custom_smote.py
----------------
A from-scratch implementation of SMOTE (Synthetic Minority Over-sampling
Technique), used to fix the severe class imbalance in the credit card
fraud dataset (~0.17% fraud).

Why a custom implementation instead of `imbalanced-learn`?
This environment doesn't have internet access to install extra packages,
so SMOTE is implemented here directly on top of scikit-learn's
NearestNeighbors. It follows the original SMOTE paper (Chawla et al., 2002):

    For every minority-class sample, find its k nearest minority-class
    neighbours. Pick one neighbour at random and create a new synthetic
    sample somewhere on the line segment between the sample and that
    neighbour.

This keeps synthetic points inside the existing minority-class region
instead of just duplicating rows (which is what naive random oversampling
would do), and is the reason SMOTE tends to generalize better than
plain oversampling.
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors


class CustomSMOTE:
    """
    Minimal, dependency-free SMOTE implementation.

    Parameters
    ----------
    sampling_strategy : float, default=0.1
        Desired ratio of minority-class samples to majority-class samples
        AFTER resampling (minority_count / majority_count). For fraud
        detection, fully balancing to 1.0 would create ~280,000 synthetic
        fraud rows from only ~400 real ones, which is both slow and prone
        to overfitting on synthetic noise. A smaller ratio (e.g. 0.1-0.2)
        gives the model many more fraud examples to learn from while
        staying grounded in real data.
    k_neighbors : int, default=5
        Number of nearest minority-class neighbours considered when
        generating each synthetic sample.
    random_state : int, default=42
        Seed for reproducibility.
    """

    def __init__(self, sampling_strategy=0.1, k_neighbors=5, random_state=42):
        self.sampling_strategy = sampling_strategy
        self.k_neighbors = k_neighbors
        self.random_state = random_state

    def fit_resample(self, X, y):
        """
        Resample the dataset so the minority class reaches
        `sampling_strategy` * majority_count samples.

        X : np.ndarray, shape (n_samples, n_features)
        y : np.ndarray, shape (n_samples,)  -- binary labels {0, 1}

        Returns synthetically-augmented (X_resampled, y_resampled).
        """
        rng = np.random.RandomState(self.random_state)

        X = np.asarray(X)
        y = np.asarray(y)

        classes, counts = np.unique(y, return_counts=True)
        class_counts = dict(zip(classes, counts))

        minority_class = min(class_counts, key=class_counts.get)
        majority_class = max(class_counts, key=class_counts.get)

        minority_count = class_counts[minority_class]
        majority_count = class_counts[majority_class]

        target_minority_count = int(self.sampling_strategy * majority_count)
        n_synthetic = max(target_minority_count - minority_count, 0)

        if n_synthetic == 0:
            return X.copy(), y.copy()

        X_minority = X[y == minority_class]

        # Guard against k being larger than the available minority samples
        k = min(self.k_neighbors, len(X_minority) - 1)
        k = max(k, 1)

        nn = NearestNeighbors(n_neighbors=k + 1)  # +1 because a point is its own neighbour
        nn.fit(X_minority)
        _, neighbor_indices = nn.kneighbors(X_minority)

        synthetic_samples = np.zeros((n_synthetic, X.shape[1]))

        for i in range(n_synthetic):
            sample_idx = rng.randint(0, len(X_minority))
            sample = X_minority[sample_idx]

            # Pick one of its k nearest neighbours (skip index 0, which is itself)
            neighbor_choice = rng.randint(1, k + 1)
            neighbor_idx = neighbor_indices[sample_idx][neighbor_choice]
            neighbor = X_minority[neighbor_idx]

            # Interpolate a new point somewhere between sample and neighbor
            gap = rng.uniform(0, 1)
            synthetic_samples[i] = sample + gap * (neighbor - sample)

        X_resampled = np.vstack([X, synthetic_samples])
        y_resampled = np.hstack([y, np.full(n_synthetic, minority_class)])

        # Shuffle so synthetic rows aren't all stacked at the end
        shuffle_idx = rng.permutation(len(X_resampled))
        return X_resampled[shuffle_idx], y_resampled[shuffle_idx]
