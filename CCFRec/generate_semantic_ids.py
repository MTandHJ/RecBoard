

# %%
from typing import List

import os
import torch, faiss
import numpy as np
from itertools import chain
from sklearn.decomposition import PCA
from freerec.utils import import_pickle, export_pickle


# %%

# loading
root = "../../data"
dataset = "Amazon2014Beauty_550_LOU"
path = os.path.join(root, "Processed", dataset)
encoder = "sentence-t5-xl"
fields = ("Title", "CATEGORIES", "BRAND")

features: List[np.ndarray] = []
for field in fields:
    tfile = f"{encoder}_{field}.pkl".lower()
    features.append(
        import_pickle(os.path.join(path, tfile)).numpy() # (N, D)
    )

# %%

# pca preprocessing

pca_size: int = 128

processor = PCA(n_components=pca_size, whiten=True)
features = [
    np.ascontiguousarray(
        processor.fit_transform(feat)
    )
    for feat in features
]

# %%

# vector quantization

num_codebooks = 4 * len(fields)
num_codewords = 256
num_bits = int(np.log2(num_codewords)) # 256 == 8bit
quantizer = 'product'
# quantizer = 'residual'

sem_ids = []
if quantizer == "product":

    features = np.concatenate(features, axis=1)
    print(features.shape)
    index = faiss.IndexPQ(
        features.shape[-1],
        num_codebooks,
        num_bits,
        faiss.METRIC_INNER_PRODUCT
    )

    index.train(features)
    uint8_code = index.pq.compute_codes(features)
    num_bytes = uint8_code.shape[1]
    for u8_code in uint8_code:
        bs = faiss.BitstringReader(faiss.swig_ptr(u8_code), num_bytes)
        code = []
        for i in range(num_codebooks):
            code.append(bs.read(num_bits))
        sem_ids.append(code)
    sem_ids = np.array(sem_ids)

elif quantizer == "residual":
    sem_ids = []
    for feat in features:
        index = faiss.IndexResidualQuantizer(
            feat.shape[-1],
            num_codebooks,
            num_bits,
            faiss.METRIC_INNER_PRODUCT
        )
        index.train(feat)
        index.add(feat)
        faiss_sem_ids = []
        uint8_code = index.rq.compute_codes(feat)
        num_bytes = uint8_code.shape[1]
        for u8_code in uint8_code:
            bs = faiss.BitstringReader(faiss.swig_ptr(u8_code), num_bytes)
            code = []
            for i in range(num_codebooks):
                code.append(bs.read(num_bits))
            faiss_sem_ids.append(code)
        faiss_sem_ids = np.array(faiss_sem_ids)
        sem_ids.append(faiss_sem_ids.tolist())
    sem_ids = [
        list(chain(*piece))
        for piece in zip(*sem_ids)
    ]
    sem_ids = np.array(sem_ids)


sem_ids = torch.from_numpy(sem_ids).long()
export_pickle(
    sem_ids,
    os.path.join(
        path, f"{quantizer}_{num_codebooks}_{num_codewords}.pkl"
    )
)