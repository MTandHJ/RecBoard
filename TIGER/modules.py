

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F





class SemIDEmbedding(nn.Embedding):

    def __init__(
        self,
        sem_ids: torch.Tensor,
        embedding_dim: int,
        padding: bool = False
    ):

        N, self.num_buckets = sem_ids.shape
        self.num_codes = sem_ids.max()
        num_embeddings = self.num_codes * self.num_buckets
        sem_ids = sem_ids + torch.arange(self.num_buckets).unsqueeze(0) * self.num_codes

        padding_idx = None
        if padding:
            sem_ids = sem_ids + 1
            sem_ids = torch.cat(
                (
                    torch.zeros_like(sem_ids)[[0]],
                    sem_ids
                ),
                dim=0
            )

            num_embeddings += 1
            padding_idx = 0

        super().__init__(num_embeddings, embedding_dim, padding_idx)
        
        self.register_buffer(
            "sem_ids", sem_ids
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # input (B, S) -> (B, S, K) -> (B, S * K)
        input = self.sem_ids[input].flatten(dim=1) 
        return super().forward(input)