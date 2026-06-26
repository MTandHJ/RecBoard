from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from quantizer import ResidualQuantizer
from torch.nn.init import xavier_normal_
from transformers import GenerationMixin
from transformers.modeling_outputs import ModelOutput

__all__ = [
    "ETEGRecGenerator",
    "ETEGRecGeneratorOutput",
    "ETEGRecTokenizer",
    "ETEGRecTokenizerOutput",
    "MLPLayers",
]


class MLPLayers(nn.Module):
    def __init__(self, layers, dropout: float = 0.0, activation: str = "relu", bn: bool = False):
        super().__init__()
        self.layers = layers
        self.dropout = dropout
        self.activation = activation
        self.use_bn = bn

        modules = []
        for idx, (input_size, output_size) in enumerate(zip(layers[:-1], layers[1:])):
            modules.append(nn.Dropout(p=dropout))
            modules.append(nn.Linear(input_size, output_size))
            if bn and idx != len(layers) - 2:
                modules.append(nn.BatchNorm1d(num_features=output_size))
            activation_func = self.activation_layer(activation)
            if activation_func is not None and idx != len(layers) - 2:
                modules.append(activation_func)

        self.mlp_layers = nn.Sequential(*modules)
        self.apply(self.init_weights)

    @staticmethod
    def activation_layer(activation_name: str):
        if activation_name is None or activation_name.lower() == "none":
            return None
        if activation_name.lower() == "sigmoid":
            return nn.Sigmoid()
        if activation_name.lower() == "tanh":
            return nn.Tanh()
        if activation_name.lower() == "relu":
            return nn.ReLU()
        if activation_name.lower() == "leakyrelu":
            return nn.LeakyReLU()
        raise NotImplementedError(f"activation function {activation_name} is not implemented")

    @staticmethod
    def init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                module.bias.data.fill_(0.0)

    def forward(self, input_feature: torch.Tensor) -> torch.Tensor:
        return self.mlp_layers(input_feature)


@dataclass
class ETEGRecGeneratorOutput(ModelOutput):
    logits: Optional[torch.FloatTensor] = None
    seq_latents: Optional[torch.FloatTensor] = None
    seq_project_latents: Optional[torch.FloatTensor] = None
    dec_latents: Optional[torch.FloatTensor] = None


@dataclass
class ETEGRecTokenizerOutput(ModelOutput):
    reconstructions: Optional[torch.FloatTensor] = None
    quantized: Optional[torch.FloatTensor] = None
    loss: Optional[torch.FloatTensor] = None
    indices: Optional[torch.LongTensor] = None
    one_hot: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None


class ETEGRecGenerator(nn.Module, GenerationMixin):
    r"""Code-token T5 wrapper used by ETEGRec."""

    def __init__(
        self,
        config,
        model,
        num_items: int,
        code_length: int = 1,
        num_codewords: int = 256,
    ):
        super().__init__()
        self.model = model
        # self._supports_cache_class = model._supports_cache_class
        self.config = model.config
        self.base_model_prefix = "model"
        self.generation_config = model.generation_config
        self.main_input_name = model.main_input_name
        self.get_encoder = model.get_encoder
        self.device = model.device
        self.can_generate = lambda: True

        self.hidden_size = model.config.hidden_size
        self.item_feat_dim = config["item_feat_dim"]
        self.num_items = num_items
        self.code_length = code_length
        self.num_codewords = num_codewords
        self.num_beams = config["num_beams"]

        self.semantic_embedding = nn.Embedding(num_items, self.item_feat_dim)
        self.semantic_embedding.requires_grad_(False)

        self.token_embeddings = nn.ModuleList(
            [nn.Embedding(num_codewords, self.hidden_size) for _ in range(code_length)]
        )
        self.token_embeddings.requires_grad_(True)

        self.enc_adapter = MLPLayers(layers=[self.hidden_size, config["codebook_dim"]])
        self.dec_adapter = MLPLayers(layers=[self.hidden_size, self.item_feat_dim])
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def prepare_inputs_for_generation(
        self, input_ids, attention_mask=None, encoder_outputs=None, **kwargs
    ):
        return {
            "decoder_input_ids": input_ids,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
        }

    def _shift_right(self, input_ids: torch.Tensor) -> torch.Tensor:
        pad_token_id = self.config.pad_token_id
        shifted = torch.full(input_ids.shape[:-1] + (1,), pad_token_id, device=input_ids.device)
        return torch.cat([shifted, input_ids], dim=-1)

    def get_input_embeddings(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        attention_mask_flatten = attention_mask.reshape(-1)
        safe_input_ids = input_ids.masked_fill(input_ids.eq(-1), 0)

        inputs_embeds = torch.zeros(*safe_input_ids.shape, self.hidden_size, device=self.device)
        for i in range(self.code_length):
            inputs_embeds[:, i :: self.code_length] = self.token_embeddings[i](
                safe_input_ids[:, i :: self.code_length]
            )

        inputs_embeds = inputs_embeds.view(-1, self.hidden_size)
        inputs_embeds[~attention_mask_flatten] = self.model.shared.weight[0]
        return inputs_embeds.view(safe_input_ids.shape[0], -1, self.hidden_size)

    def forward(
        self,
        input_ids=None,
        inputs_embeds=None,
        attention_mask=None,
        labels=None,
        decoder_input_ids=None,
        decoder_inputs_embeds=None,
        encoder_outputs=None,
        **kwargs,
    ):
        if input_ids is not None:
            inputs_embeds = self.get_input_embeddings(input_ids, attention_mask)

        if decoder_input_ids is None and labels is None:
            decoder_input_ids = (
                torch.zeros(
                    input_ids.size(0),
                    self.code_length,
                )
                .long()
                .to(input_ids.device)
            )
        elif decoder_input_ids is None and labels is not None:
            decoder_input_ids = self._shift_right(labels)

        if decoder_inputs_embeds is None and decoder_input_ids is not None:
            decoder_inputs_embeds = []
            for i in range(min(decoder_input_ids.shape[1], self.code_length)):
                code_embedding = self.model.shared if i == 0 else self.token_embeddings[i - 1]
                decoder_inputs_embeds.append(code_embedding(decoder_input_ids[:, i]))
            decoder_inputs_embeds = torch.stack(decoder_inputs_embeds, dim=1)

        model_outputs = self.model(
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            output_hidden_states=True,
            encoder_outputs=encoder_outputs,
        )
        decoder_outputs = model_outputs.decoder_hidden_states[-1]

        code_logits = []
        for i in range(min(decoder_inputs_embeds.shape[1], self.code_length)):
            centroid = self.token_embeddings[i].weight.t()
            code_logits.append(torch.matmul(decoder_outputs[:, i], centroid))
        code_logits = torch.stack(code_logits, dim=1)

        seq_latents = model_outputs.encoder_last_hidden_state.clone()
        seq_latents[~attention_mask] = 0
        seq_last_latents = torch.sum(seq_latents, dim=1) / attention_mask.sum(dim=1).unsqueeze(1)
        seq_project_latents = self.enc_adapter(seq_last_latents)

        dec_latents = model_outputs.decoder_hidden_states[-1].clone()
        dec_latents = self.dec_adapter(dec_latents[:, 0, :])
        return ETEGRecGeneratorOutput(
            logits=code_logits,
            seq_latents=seq_last_latents,
            seq_project_latents=seq_project_latents,
            dec_latents=dec_latents,
        )

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        n_return_sequences: int = 1,
        prefix_allowed_tokens_fn=None,
    ) -> torch.Tensor:
        if prefix_allowed_tokens_fn is not None:
            inputs_embeds = self.get_input_embeddings(input_ids, attention_mask)
            outputs = super().generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_length=self.code_length + 1,
                num_beams=self.num_beams,
                num_return_sequences=n_return_sequences,
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            )
        else:
            outputs = self.my_beam_search(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=self.code_length + 1,
                num_beams=self.num_beams,
                num_return_sequences=n_return_sequences,
                return_score=False,
            )
        return outputs[:, 1:].reshape(-1, n_return_sequences, self.code_length)

    def my_beam_search(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_length: int = 6,
        num_beams: int = 1,
        num_return_sequences: int = 1,
        return_score: bool = False,
    ):
        batch_size = input_ids.shape[0]
        input_ids, attention_mask, decoder_input_ids, beam_scores, beam_idx_offset = (
            self.prepare_beam_search_inputs(input_ids, attention_mask, batch_size, num_beams)
        )
        inputs_embeds = self.get_input_embeddings(input_ids, attention_mask)

        with torch.no_grad():
            encoder_outputs = self.get_encoder()(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
            )

        while decoder_input_ids.shape[1] < max_length:
            with torch.no_grad():
                outputs = self.forward(
                    encoder_outputs=encoder_outputs,
                    attention_mask=attention_mask,
                    decoder_input_ids=decoder_input_ids,
                )
            decoder_input_ids, beam_scores = self.beam_search_step(
                outputs.logits,
                decoder_input_ids,
                beam_scores,
                beam_idx_offset,
                batch_size,
                num_beams,
            )

        selection_mask = torch.zeros(
            batch_size,
            num_beams,
            dtype=torch.bool,
            device=input_ids.device,
        )
        selection_mask[:, :num_return_sequences] = True
        selection_mask = selection_mask.view(-1)
        if return_score:
            return decoder_input_ids[selection_mask, :], beam_scores[selection_mask] / (
                decoder_input_ids.shape[1] - 1
            )
        return decoder_input_ids[selection_mask, :]

    def prepare_beam_search_inputs(
        self,
        input_ids,
        attention_mask,
        batch_size: int,
        num_beams: int,
    ):
        decoder_input_ids = torch.ones(
            (batch_size * num_beams, 1),
            device=self.device,
            dtype=torch.long,
        )
        initial_decoder_input_ids = decoder_input_ids * self.config.decoder_start_token_id

        beam_scores = torch.zeros(
            (batch_size, num_beams),
            dtype=torch.float,
            device=input_ids.device,
        )
        beam_scores[:, 1:] = -1e9
        initial_beam_scores = beam_scores.view(batch_size * num_beams)

        beam_idx_offset = (
            torch.arange(batch_size, device=self.device).repeat_interleave(num_beams) * num_beams
        )
        input_ids = input_ids.repeat_interleave(num_beams, dim=0)
        attention_mask = attention_mask.repeat_interleave(num_beams, dim=0)
        return (
            input_ids,
            attention_mask,
            initial_decoder_input_ids,
            initial_beam_scores,
            beam_idx_offset,
        )

    @staticmethod
    def beam_search_step(
        logits,
        decoder_input_ids,
        beam_scores,
        beam_idx_offset,
        batch_size: int,
        num_beams: int,
    ):
        vocab_size = logits.shape[-1]
        next_token_scores = torch.log_softmax(logits[:, -1, :], dim=-1)
        next_token_scores = next_token_scores + beam_scores[:, None].expand_as(next_token_scores)
        next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)
        next_token_scores, next_tokens = torch.topk(
            next_token_scores,
            2 * num_beams,
            dim=1,
            largest=True,
            sorted=True,
        )

        next_indices = torch.div(next_tokens, vocab_size, rounding_mode="floor")
        next_tokens = next_tokens % vocab_size

        beam_scores = next_token_scores[:, :num_beams].reshape(-1)
        beam_next_tokens = next_tokens[:, :num_beams].reshape(-1)
        beam_idx = next_indices[:, :num_beams].reshape(-1)
        decoder_input_ids = torch.cat(
            [decoder_input_ids[beam_idx + beam_idx_offset, :], beam_next_tokens.unsqueeze(-1)],
            dim=-1,
        )
        return decoder_input_ids, beam_scores


class ETEGRecTokenizer(nn.Module):
    r"""RQ-VAE item tokenizer used by ETEGRec."""

    def __init__(self, config, in_dim: int = 768):
        super().__init__()
        self.in_dim = in_dim
        self.codebook_dim = config["codebook_dim"]
        self.hidden_dims = config["hidden_dims"]
        self.dropout_rate = config["tokenizer_dropout_rate"]

        self.encode_layer_dims = [in_dim] + self.hidden_dims + [self.codebook_dim]
        self.decode_layer_dims = self.encode_layer_dims[::-1]
        self.encoder = MLPLayers(self.encode_layer_dims, dropout=self.dropout_rate)
        self.rq = ResidualQuantizer(config=config)
        self.decoder = MLPLayers(self.decode_layer_dims, dropout=self.dropout_rate)

    def forward(self, x: torch.Tensor) -> ETEGRecTokenizerOutput:
        latent = self.encoder(x)
        quantizer_output = self.rq(latent)
        reconstructions = self.decoder(quantizer_output.quantized)
        return ETEGRecTokenizerOutput(
            reconstructions=reconstructions,
            quantized=quantizer_output.quantized,
            loss=quantizer_output.loss,
            indices=quantizer_output.indices,
            one_hot=quantizer_output.one_hot,
            logits=quantizer_output.logits,
        )

    @torch.no_grad()
    def get_indices(self, xs: torch.Tensor) -> torch.Tensor:
        return self.rq.get_indices(self.encoder(xs))

    @torch.no_grad()
    def get_maxk_indices(self, xs: torch.Tensor, maxk: int = 1, used: bool = False):
        return self.rq.get_maxk_indices(self.encoder(xs), maxk=maxk, used=used)

    def get_codebook(self) -> torch.Tensor:
        return self.rq.get_codebook()
