from datetime import timedelta

import numpy as np

from embedding_utils import cosine_sim_matrix


def retrieve_similar_indices(i, df, embeddings, n_sim=None, window_days=7, min_sim=0.5):
    current_time = df.loc[i, "time"]
    start_time = current_time - timedelta(days=window_days)
    mask = (df["time"] < current_time) & (df["time"] >= start_time)
    candidate_idx = df.index[mask].tolist()
    if not candidate_idx:
        return []
    sims = cosine_sim_matrix(embeddings[i], embeddings[candidate_idx])
    sims = np.asarray(sims)
    keep = np.where(sims >= min_sim)[0]
    if keep.size == 0:
        return []
    top = keep[np.argsort(-sims[keep])]
    if n_sim is not None:
        top = top[:n_sim]
    return [candidate_idx[j] for j in top]
