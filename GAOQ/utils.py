from typing import Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


def hungarian_match(A: np.ndarray, B: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    r"""Match source vectors to distinct anchors by cosine similarity.

    Parameters
    ----------
    A : np.ndarray, shape (N, D)
        Source vectors.
    B : np.ndarray, shape (M, D)
        Anchor vectors, where ``M >= N``.

    Returns
    -------
    matched_idx_B : np.ndarray, shape (N,)
        Anchor index matched to each source vector.
    similarities : np.ndarray, shape (N,)
        Cosine similarity of each matched pair.
    """

    A_norms = np.linalg.norm(A, axis=1, keepdims=True)
    B_norms = np.linalg.norm(B, axis=1, keepdims=True)
    A_norm = np.divide(
        A,
        A_norms,
        out=np.zeros_like(A, dtype=np.result_type(A.dtype, np.float32)),
        where=A_norms > 0,
    )
    B_norm = np.divide(
        B,
        B_norms,
        out=np.zeros_like(B, dtype=np.result_type(B.dtype, np.float32)),
        where=B_norms > 0,
    )
    similarities = A_norm @ B_norm.T
    row_ind, col_ind = linear_sum_assignment(-similarities)
    return col_ind, similarities[row_ind, col_ind]
