

from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from freerec.utils import infoLogger


class SemIDTokenzier(nn.Module):

    PADDING_VALUE = 0
    LOCAL_BATCH_SIZE = 64

    def __init__(
        self, sem_ids: torch.Tensor,
    ):
        super().__init__()
        sem_ids = self.remove_conflict(sem_ids)
        sem_ids = self.remap(sem_ids)
        N, self.num_codebooks = sem_ids.shape
        num_codewords = sem_ids.max(dim=0)[0] + 1
        sem_ids[:, 1:] += num_codewords.cumsum(dim=0)[:-1]
        self.vocab_size: int = num_codewords.sum().item() + 1
        self.num_codewords: List[int] = num_codewords.tolist()

        sem_ids = sem_ids + 1
        sem_ids = F.pad(
            sem_ids,
            (0, 0, 1, 0),
            value=self.PADDING_VALUE
        )
        
        self.register_buffer(
            "sem_ids", sem_ids
        )
        self.sem_id_map = {
            tuple(ids.tolist()): idx 
            for idx, ids in enumerate(self.sem_ids.cpu())
        }

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
        infoLogger(f"Additional tokens: {sem_ids[:, -1].max().item()}")
        return sem_ids

    def remap(self, sem_ids: torch.Tensor):
        sem_ids = sem_ids.clone()
        N, C = sem_ids.shape
        for j in range(C):
            _, inverse = torch.unique(sem_ids[:, j], return_inverse=True)
            sem_ids[:, j] = inverse
        return sem_ids

    def encode(self, item_ids: torch.Tensor) -> torch.Tensor:
        r"""
        Parameters:
        -----------
        item_ids: torch.Tensor, (B, S)

        Returns:
        sem_ids: torch.Tensor, (B, S * L)
        """
        return self.sem_ids[item_ids].flatten(start_dim=1)

    def decode(self, sem_ids: torch.Tensor) -> torch.Tensor:
        r"""
        Parameters:
        -----------
        sem_ids: torch.Tensor, (*, #codebooks)

        Returns:
        --------
        item_ids: torch.Tensor, (*,)
        """
        sizes = sem_ids.shape[:-1]
        item_ids = [
            self.sem_id_map.get(tuple(sem_id), self.PADDING_VALUE) 
            for sem_id in sem_ids.flatten(end_dim=-2).cpu().tolist()
        ]
        item_ids = torch.tensor(item_ids, dtype=torch.long, device=sem_ids.device)
        return item_ids.view(*sizes)