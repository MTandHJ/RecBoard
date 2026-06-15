from typing import List, Optional

import torch
import torch.nn as nn

__all__ = [
    "DIGERIDDecoder",
    "DIGERIDEncoder",
    "DIGERT5",
    "MLPLayers",
]


class MLPLayers(nn.Module):
    r"""Compact MLP used by DIGER encoders and adapters.

    Parameters
    ----------
    hidden_dims : List[int]
        Layer dimensions including input and output sizes.
    dropout_rate : float, default=0.0
        Dropout before every linear layer.
    bias : bool, default=True
        Whether linear layers use bias terms.
    """

    ACT = nn.ReLU

    def __init__(
        self,
        hidden_dims: List[int],
        dropout_rate: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()

        self.fc = nn.Sequential()
        for level, (input_dim, output_dim) in enumerate(
            zip(hidden_dims[:-1], hidden_dims[1:]),
            start=1,
        ):
            self.fc.append(nn.Dropout(dropout_rate))
            self.fc.append(nn.Linear(input_dim, output_dim, bias=bias))
            if level < len(hidden_dims) - 1:
                self.fc.append(self.ACT())

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class DIGERIDEncoder(nn.Module):
    r"""Map item semantic features into DIGER's residual code space.

    Parameters
    ----------
    in_dim : int
        Input item feature dimension.
    hidden_dims : List[int]
        Hidden MLP dimensions.
    codebook_dim : int
        Latent dimension consumed by the residual quantizer.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dims: List[int],
        codebook_dim: int,
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.encoder = MLPLayers(
            [in_dim] + list(hidden_dims) + [codebook_dim],
            dropout_rate=dropout_rate,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class DIGERIDDecoder(nn.Module):
    r"""Reconstruct item semantic features from DIGER quantized latents."""

    def __init__(
        self,
        out_dim: int,
        hidden_dims: List[int],
        codebook_dim: int,
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.decoder = MLPLayers(
            [codebook_dim] + list(hidden_dims) + [out_dim],
            dropout_rate=dropout_rate,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)


class DIGERT5(nn.Module):
    r"""T5 recommender that consumes and generates DIGER code tokens.

    Parameters
    ----------
    t5 : nn.Module
        HuggingFace T5-like encoder-decoder.
    num_items : int
        Number of shifted items including the padding item.
    item_feat_dim : int
        Dimension of item semantic features.
    num_codebooks : int
        Number of code tokens per item, including the conflict token.
    num_codewords : int
        Number of valid code values at each code position.
    num_beams : int
        Beam width for full-ranking candidate generation.
    """

    def __init__(
        self,
        t5: nn.Module,
        num_items: int,
        item_feat_dim: int,
        num_codebooks: int,
        num_codewords: int,
        num_beams: int,
    ) -> None:
        super().__init__()

        self.t5 = t5
        self.get_encoder = t5.get_encoder

        self.hidden_size = t5.config.d_model
        self.num_codebooks = num_codebooks
        self.num_codewords = num_codewords
        self.num_beams = num_beams

        self.item_embeddings = nn.Embedding(num_items, item_feat_dim)
        self.item_embeddings.requires_grad_(False)
        self.code_embeddings = nn.ModuleList(
            [nn.Embedding(num_codewords, self.hidden_size) for _ in range(num_codebooks)]
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def shift_right(self, target_ids: torch.Tensor) -> torch.Tensor:
        prefix = target_ids.new_zeros(target_ids.shape[:-1] + (1,))
        return torch.cat([prefix, target_ids[..., :-1]], dim=-1)

    def embed_context(self, context_ids: torch.Tensor, context_mask: torch.Tensor) -> torch.Tensor:
        safe_ids = context_ids.masked_fill(context_ids.eq(-1), 0)  # (B, L)
        context_embeds = torch.zeros(
            *safe_ids.shape,
            self.hidden_size,
            device=safe_ids.device,
        )  # (B, L, D)
        for level, embedding in enumerate(self.code_embeddings):
            context_embeds[:, level :: self.num_codebooks] = embedding(
                safe_ids[:, level :: self.num_codebooks]
            )

        context_embeds = context_embeds.view(-1, self.hidden_size)  # (B * L, D)
        context_embeds[~context_mask.reshape(-1)] = self.t5.shared.weight[0]
        return context_embeds.view(*safe_ids.shape, self.hidden_size)  # (B, L, D)

    def embed_decoder(self, decoder_ids: torch.Tensor) -> torch.Tensor:
        embeds = []
        max_steps = min(decoder_ids.size(1), self.num_codebooks)
        for level in range(max_steps):
            # decoder starts with <pad> token
            embedding = self.t5.shared if level == 0 else self.code_embeddings[level - 1]
            embeds.append(embedding(decoder_ids[:, level]))
        return torch.stack(embeds, dim=1)

    def forward(
        self,
        context_ids: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        target_ids: Optional[torch.Tensor] = None,
        decoder_ids: Optional[torch.Tensor] = None,
        encoder_outputs=None,
    ) -> torch.Tensor:
        r"""Run DIGER's T5 code predictor.

        Parameters
        ----------
        context_ids : torch.Tensor, optional
            Flattened history code tokens with shape ``(B, S * num_codebooks)``.
            Padding code tokens are marked as ``-1``.
        context_mask : torch.Tensor, optional
            Boolean mask for valid context code tokens with the same leading shape
            as ``context_ids``.
        target_ids : torch.Tensor, optional
            Training labels for the next item code with shape ``(B, num_codebooks)``.
            When provided without ``decoder_ids``, they are shifted right to build
            teacher-forcing decoder inputs.
        decoder_ids : torch.Tensor, optional
            Decoder-side code prefix. During generation this is the current beam
            prefix; during training it is usually ``shift_right(target_ids)``.
        encoder_outputs : optional
            Cached T5 encoder outputs reused by beam search.

        Raises
        ------
        ValueError
            If neither ``target_ids`` nor ``decoder_ids`` is provided.
        """
        if decoder_ids is None:
            if target_ids is None:
                raise ValueError("Either target_ids or decoder_ids should be provided.")
            decoder_ids = self.shift_right(target_ids)

        outputs = self.t5(
            attention_mask=context_mask,
            inputs_embeds=None
            if encoder_outputs is not None
            else self.embed_context(context_ids, context_mask),
            decoder_inputs_embeds=self.embed_decoder(decoder_ids),
            encoder_outputs=encoder_outputs,
            output_hidden_states=True,
            return_dict=True,
        )
        decoder_outputs = outputs.decoder_hidden_states[-1]
        logits = torch.stack(
            [
                decoder_outputs[:, level] @ self.code_embeddings[level].weight.t()  # (B, N)
                for level in range(min(decoder_outputs.size(1), self.num_codebooks))
            ],
            dim=1,
        )  # (B, #num_codebooks, N)
        return logits

    @torch.no_grad()
    def generate_candidates(
        self,
        context_ids: torch.Tensor,
        context_mask: torch.Tensor,
        num_candidates: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sequences, sequence_scores = self.beam_search(
            context_ids=context_ids,
            context_mask=context_mask,
            max_length=self.num_codebooks + 1,
            num_beams=self.num_beams,
            num_candidates=num_candidates,
        )
        candidate_codes = sequences[:, 1:].reshape(-1, num_candidates, self.num_codebooks)
        sequence_scores = sequence_scores.reshape(-1, num_candidates)
        return candidate_codes, sequence_scores

    def beam_search(
        self,
        context_ids: torch.Tensor,
        context_mask: torch.Tensor,
        max_length: int,
        num_beams: int,
        num_candidates: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = context_ids.size(0)
        context_ids = context_ids.repeat_interleave(num_beams, dim=0)
        context_mask = context_mask.repeat_interleave(num_beams, dim=0)
        decoder_ids = context_ids.new_zeros(batch_size * num_beams, 1)
        beam_scores = context_ids.new_zeros(batch_size, num_beams, dtype=torch.float)
        beam_scores[:, 1:] = -1.0e9
        beam_scores = beam_scores.reshape(-1)
        beam_offsets = (
            torch.arange(batch_size, device=context_ids.device).repeat_interleave(num_beams)
            * num_beams
        )

        encoder_outputs = self.get_encoder()(
            inputs_embeds=self.embed_context(context_ids, context_mask),
            attention_mask=context_mask,
            return_dict=True,
        )
        while decoder_ids.size(1) < max_length:
            outputs = self.forward(
                context_ids=context_ids,
                context_mask=context_mask,
                decoder_ids=decoder_ids,
                encoder_outputs=encoder_outputs,
            )
            decoder_ids, beam_scores = self.beam_search_step(
                outputs,
                decoder_ids,
                beam_scores,
                beam_offsets,
                batch_size,
                num_beams,
            )

        keep = torch.zeros(batch_size, num_beams, dtype=torch.bool, device=context_ids.device)
        keep[:, :num_candidates] = True
        return decoder_ids[keep.view(-1)], beam_scores[keep.view(-1)] / (decoder_ids.size(1) - 1)

    def beam_search_step(
        self,
        logits: torch.Tensor,
        decoder_ids: torch.Tensor,
        beam_scores: torch.Tensor,
        beam_offsets: torch.Tensor,
        batch_size: int,
        num_beams: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_codewords = logits.size(-1)
        scores = logits[:, -1].log_softmax(dim=-1) + beam_scores[:, None]
        scores = scores.view(batch_size, num_beams * num_codewords)
        scores, tokens = scores.topk(k=2 * num_beams, dim=1, largest=True, sorted=True)

        beam_ids = torch.div(tokens, num_codewords, rounding_mode="floor").reshape(-1)
        next_tokens = (tokens % num_codewords).reshape(-1)
        scores = scores[:, :num_beams]
        beam_ids = beam_ids.view(batch_size, -1)[:, :num_beams].reshape(-1)
        next_tokens = next_tokens.view(batch_size, -1)[:, :num_beams].reshape(-1)
        decoder_ids = torch.cat(
            [decoder_ids[beam_ids + beam_offsets], next_tokens[:, None]],
            dim=1,
        )
        return decoder_ids, scores.reshape(-1)
