

from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from freerec.utils import infoLogger

PADDING_VALUE = 0
LOCAL_BATCH_SIZE = 64

class SemIDEmbedding(nn.Embedding):

    def __init__(
        self,
        sem_ids: torch.Tensor,
        embedding_dim: int,
        padding: bool = True
    ):
        sem_ids = self.remove_conflict(sem_ids)
        N, self.num_codebooks = sem_ids.shape
        num_codewords = sem_ids.max(dim=0)[0] + 1
        num_embeddings = num_codewords.sum().item()
        sem_ids[:, 1:] += num_codewords.cumsum(dim=0)[:-1]
        self.num_codewords: List[int] = num_codewords.tolist()

        padding_idx = None
        if padding:
            padding_idx = PADDING_VALUE
            num_embeddings += 1
            sem_ids = sem_ids + 1
            sem_ids = F.pad(
                sem_ids,
                (0, 0, 1, 0),
                value=PADDING_VALUE
            )

        super().__init__(num_embeddings, embedding_dim, padding_idx)
        
        self.register_buffer(
            "sem_ids", sem_ids
        )
        self.sem_id_map = {tuple(ids.tolist()): idx for idx, ids in enumerate(self.sem_ids.cpu())}

    def remove_conflict(self, sem_ids: torch.Tensor):
        r"""
        Add one check digit to avoid allocation conflicts.

        Parameters:
        -----------
        sem_ids: torch.Tensor, (N, L)

        Returns:
        --------
        sem_ids: torch.Tensor, (N, L + 1)
        """
        codes = defaultdict(int)
        check_code = []
        for sem_id in sem_ids.cpu().tolist():
            sem_id = tuple(sem_id)
            check_code.append(codes[sem_id])
            codes[sem_id] += 1
        sem_ids = torch.cat(
            (
                sem_ids,
                torch.tensor(check_code, dtype=torch.long).unsqueeze(-1)
            ),
            dim=-1
        ) # (N, L + 1)
        infoLogger(f"Number of conflicts: {sem_ids[:, -1].sum().item()}")
        infoLogger(f"Additional tokens: {sem_ids[:, -1].max().item()}")
        return sem_ids

    def _check_validity(self, generated: torch.Tensor):
        L = generated.size(1)
        valid_patterns = self.sem_ids[:, :L]
        return (generated.unsqueeze(1).eq(valid_patterns)).all(dim=-1).any(dim=-1)

    def check_validity(self, generated: torch.Tensor):
        r"""
        Check the validity of generated semantic ids.

        Parameters:
        -----------
        generated: torch.Tensor, (B, L)

        Returns:
        --------
        is_valid: torch.bool, (B,)
        """
        is_valid = []
        for chunk in torch.split(generated, LOCAL_BATCH_SIZE, dim=0):
            is_valid.append(
                self._check_validity(chunk)
            )
        return torch.cat(is_valid, dim=0)

    def item_ids_to_sem_ids(self, item_ids: torch.Tensor) -> torch.Tensor:
        r"""
        Parameters:
        -----------
        item_ids: torch.Tensor, (B, S)

        Returns:
        sem_ids: torch.Tensor, (B, S * L)
        """
        return self.sem_ids[item_ids].flatten(start_dim=1)

    def sem_ids_to_item_ids(self, sem_ids: torch.Tensor) -> torch.Tensor:
        r"""
        Parameters:
        -----------
        sem_ids: torch.Tensor, (*, L)

        Returns:
        --------
        item_ids: torch.Tensor, (*,)
        """
        sizes = sem_ids.shape[:-1]
        item_ids = [
            self.sem_id_map.get(tuple(sem_id), PADDING_VALUE) 
            for sem_id in sem_ids.flatten(end_dim=-2).cpu().tolist()
        ]
        item_ids = torch.tensor(item_ids, dtype=torch.long, device=sem_ids.device)
        return item_ids.view(*sizes)