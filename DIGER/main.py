import os
from collections import defaultdict
from typing import Dict, Literal, Tuple

import freerec
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import DIGERT5, DIGERIDEncoder
from quantizer import ResidualQuantizerGumbel
from transformers import T5Config, T5ForConditionalGeneration

DTYPE = torch.bfloat16
BACKGROUND_SCORE_MAX = 1.0e-3
BEAM_SCORE_BASE = 1.0

freerec.declare(version="1.0.1")

cfg = freerec.parser.Parser()

cfg.add_argument("--maxlen", type=int, default=20, help="maximum item sequence length")
cfg.add_argument("--sem-feat-file", type=str, default="sentence-t5-xl_title_categories_brand.pkl")

cfg.add_argument("--embedding-dim", "--d-model", type=int, default=128, help="T5 d_model")
cfg.add_argument("--attention-size", "--d-kv", type=int, default=64, help="T5 d_kv")
cfg.add_argument("--intermediate-size", "--d-ff", type=int, default=512, help="T5 d_ff")
cfg.add_argument("--num-heads", type=int, default=4, help="number of attention heads")
cfg.add_argument("--num-layers", "--encoder-layers", type=int, default=6)
cfg.add_argument("--decoder-layers", type=int, default=None)
cfg.add_argument("--dropout-rate", type=float, default=0.1, help="T5 dropout rate")
cfg.add_argument("--activation-function", type=str, default="relu")
cfg.add_argument("--feed-forward-proj", type=str, default="relu")
cfg.add_argument("--num-beams", type=int, default=20, help="beam width for full ranking")

cfg.add_argument("--num-codebooks", type=int, default=3)
cfg.add_argument("--num-codewords", type=int, default=256)
cfg.add_argument("--codebook-dim", type=int, default=128)
cfg.add_argument("--hidden-dims", type=str, default="512,256,128")
cfg.add_argument("--apply-shared-codebook", type=eval, default=False)
cfg.add_argument("--commit-weight", type=float, default=0.25)
cfg.add_argument("--tokenizer-dropout-rate", type=float, default=0.0)
cfg.add_argument("--sk-epsilons", type=str, default="0.003,0.003,0.003")
cfg.add_argument("--sk-iters", type=int, default=50)
cfg.add_argument("--rqvae-path", type=str, default=None)
cfg.add_argument("--freeze-id-encoder", type=eval, default=False)

cfg.add_argument("--lr-rec", type=float, default=None)
cfg.add_argument("--lr-id", type=float, default=None)
cfg.add_argument("--lr-sigma", type=float, default=None)
cfg.add_argument("--code-loss-weight", type=float, default=1.0)
cfg.add_argument("--recon-loss-weight", type=float, default=1.0)
cfg.add_argument("--vq-loss-weight", type=float, default=1.0)

cfg.add_argument("--gumbel-tau", type=float, default=2.0)
cfg.add_argument("--hot-threshold-ratio", type=float, default=1.5)
cfg.add_argument("--usage-momentum", type=float, default=0.99)
cfg.add_argument("--initial-std", type=float, default=1.0)
cfg.add_argument("--sigma-lambda", type=float, default=0.5)

cfg.set_defaults(
    description="DIGER",
    root="../../RecData",
    dataset="Amazon2014Beauty_550_LOU",
    tasktag="NEXTITEM",
    epochs=120,
    batch_size=256,
    optimizer="AdamW",
    lr=1.0e-3,
    weight_decay=0.05,
    seed=1,
)
cfg.compile()

cfg.hidden_dims = list(map(int, cfg.hidden_dims.split(",")))
cfg.sk_epsilons = list(map(float, cfg.sk_epsilons.split(",")))


class DIGER(freerec.models.SeqRecArch):
    r"""DIGER recommender with freerec/TIGER-style data flow.

    Workflow
    --------
    1. load item semantic features from the processed dataset.
    2. build an end-to-end ID encoder, residual quantizer, and code-token T5 recommender.
    3. refresh semantic IDs from the current tokenizer before and after each epoch.
    4. train T5 code prediction together with RQ-VAE reconstruction/commitment losses.
    5. generate code candidates, map them back to item IDs, and return freerec scores.
    """

    def __init__(self, dataset: freerec.data.datasets.RecDataSet) -> None:
        super().__init__(dataset)

        self.num_codebooks = cfg.num_codebooks
        self.num_codewords = cfg.num_codewords
        self.item_codes: torch.Tensor | None = None
        self.code_to_item: Dict[Tuple[int, ...], int] = {}

        feats = freerec.utils.import_pickle(os.path.join(self.dataset.path, cfg.sem_feat_file))
        feats = F.normalize(feats, dim=-1)

        self.t5 = DIGERT5(
            t5=T5ForConditionalGeneration(self.build_t5_config()),
            num_items=self.Item.count + self.NUM_PADS,
            item_feat_dim=feats.size(1),
            num_codebooks=self.num_codebooks + 1,  # extra conflict token
            num_codewords=self.num_codewords,
            num_beams=cfg.num_beams,
        )
        self.id_encoder = DIGERIDEncoder(
            in_dim=feats.size(1),
            hidden_dims=cfg.hidden_dims,
            codebook_dim=cfg.codebook_dim,
            dropout_rate=cfg.tokenizer_dropout_rate,
        )
        self.id_quantizer = self.build_id_quantizer()
        self.load_item_embeddings(feats)
        self.load_id_tokenizer()

    def build_t5_config(self) -> T5Config:
        return T5Config(
            vocab_size=1,
            d_model=cfg.embedding_dim,
            d_kv=cfg.attention_size,
            d_ff=cfg.intermediate_size,
            num_layers=cfg.num_layers,
            num_decoder_layers=cfg.decoder_layers or cfg.num_layers,
            num_heads=cfg.num_heads,
            dropout_rate=cfg.dropout_rate,
            activation_function=cfg.activation_function,
            feed_forward_proj=cfg.feed_forward_proj,
            output_attentions=False,
            pad_token_id=0,
            eos_token_id=0,
            decoder_start_token_id=0,
        )

    def load_item_embeddings(self, item_features: torch.Tensor) -> None:
        weights = torch.zeros(
            self.Item.count + self.NUM_PADS,
            item_features.size(1),
            dtype=item_features.dtype,
        )
        weights[self.NUM_PADS :] = item_features
        self.t5.item_embeddings.weight.data.copy_(weights)
        self.t5.item_embeddings.requires_grad_(False)

    def build_id_quantizer(self) -> ResidualQuantizerGumbel:
        quantizer = ResidualQuantizerGumbel(
            self.dataset,
            cfg.codebook_dim,
            num_codebooks=cfg.num_codebooks,
            num_codewords=cfg.num_codewords,
            apply_shared_codebook=cfg.apply_shared_codebook,
            commit_weight=cfg.commit_weight,
            sk_iters=cfg.sk_iters,
            sk_epsilons=cfg.sk_epsilons,
            sigma_scale_init_std=cfg.initial_std,
            hot_threshold_ratio=cfg.hot_threshold_ratio,
        )
        quantizer.num_codewords = cfg.num_codewords
        quantizer.usage_momentum = cfg.usage_momentum
        quantizer.__dict__["auto_sigmas"] = quantizer.autosigmas
        return quantizer

    def load_id_tokenizer(self) -> None:
        if cfg.rqvae_path is None:
            raise NotImplementedError

        state = torch.load(cfg.rqvae_path, map_location="cpu")
        if not isinstance(state, dict) or "encoder" not in state or "quantizer" not in state:
            raise ValueError(
                "DIGER expects a module-wise RQ-VAE checkpoint with "
                "`encoder` and `quantizer` state dicts."
            )

        self.id_encoder.load_state_dict(state["encoder"])
        missing, unexpected = self.id_quantizer.load_state_dict(state["quantizer"], strict=False)

        if unexpected:
            freerec.utils.infoLogger(
                f"[ETEGRec] >>> ignored unexpected RQ-VAE checkpoint keys: {unexpected}"
            )
        if missing:
            freerec.utils.infoLogger(
                f"[ETEGRec] >>> missing RQ-VAE checkpoint keys: {missing}"
            )

        if cfg.freeze_id_encoder:
            self.id_encoder.requires_grad_(False)

    def sure_trainpipe(self, maxlen: int, batch_size: int):
        return (
            self.dataset.train()
            .shuffled_roll_seqs_source(minlen=2, maxlen=maxlen, keep_at_least_itself=True)
            .seq_train_yielding_pos_(start_idx_for_target=-1, end_idx_for_input=-1)
            .add_(offset=self.NUM_PADS, modified_fields=(self.ISeq, self.IPos))
            .rpad_(maxlen, modified_fields=(self.ISeq,), padding_value=self.PADDING_VALUE)
            .batch_(batch_size)
            .tensor_()
        )

    def sure_validpipe(self, maxlen: int, ranking: str = "full", batch_size: int = 512):
        return (
            self.dataset.valid()
            .ordered_user_ids_source()
            .valid_sampling_(ranking=ranking)
            .lprune_(maxlen, modified_fields=(self.ISeq,))
            .add_(offset=self.NUM_PADS, modified_fields=(self.ISeq,))
            .rpad_(maxlen, modified_fields=(self.ISeq,), padding_value=self.PADDING_VALUE)
            .batch_(batch_size)
            .tensor_()
        )

    def sure_testpipe(self, maxlen: int, ranking: str = "full", batch_size: int = 512):
        return (
            self.dataset.test()
            .ordered_user_ids_source()
            .test_sampling_(ranking=ranking)
            .lprune_(maxlen, modified_fields=(self.ISeq,))
            .add_(offset=self.NUM_PADS, modified_fields=(self.ISeq,))
            .rpad_(maxlen, modified_fields=(self.ISeq,), padding_value=self.PADDING_VALUE)
            .batch_(batch_size)
            .tensor_()
        )

    @torch.no_grad()
    def refresh_sem_ids(self, verbose: bool = False) -> None:
        encoder_was_training = self.id_encoder.training
        quantizer_was_training = self.id_quantizer.training
        self.id_encoder.eval()
        self.id_quantizer.eval()

        item_embeddings = self.t5.item_embeddings.weight.data[self.NUM_PADS :]
        item_latents = self.id_encoder(item_embeddings)
        prefix_codes = self.id_quantizer.get_indices(item_latents).detach().cpu().tolist()
        code_to_items = defaultdict(list)
        item_codes = [[-1] * (self.num_codebooks + 1)]
        max_collision = 0
        for item_id, code in enumerate(prefix_codes, start=self.NUM_PADS):
            key = tuple(code)
            code_to_items[key].append(item_id)
            collision_id = len(code_to_items[key]) - 1
            if collision_id >= self.num_codewords:
                raise ValueError(
                    f"DIGER code collision exceeds num_codewords: "
                    f"{collision_id + 1} > {self.num_codewords}"
                )
            item_codes.append(code + [collision_id])
            max_collision = max(max_collision, len(code_to_items[key]))

        self.item_codes = torch.tensor(
            item_codes,
            dtype=torch.long,
            device=item_embeddings.device,
        )
        self.code_to_item = {tuple(code): item for item, code in enumerate(item_codes)}
        self.id_encoder.train(encoder_was_training)
        self.id_quantizer.train(quantizer_was_training)

        if verbose:
            unique_codes = len({tuple(code) for code in item_codes[self.NUM_PADS :]})
            freerec.infoLogger(
                f"[DIGER] semantic IDs refreshed: unique={unique_codes}, "
                f"items={self.Item.count}, max_collision={max_collision}"
            )

    def encode_item_ids(
        self,
        context: torch.Tensor,
        targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.item_codes is None:
            self.refresh_sem_ids()
        batch_size = context.size(0)
        context = self.prune_trailing_paddings(context)
        context_ids = self.item_codes[context].reshape(batch_size, -1)
        target_ids = self.item_codes[targets.flatten()].reshape(batch_size, -1)
        return context_ids, target_ids

    def prune_trailing_paddings(self, context: torch.Tensor) -> torch.Tensor:
        valid_columns = context.ne(self.PADDING_VALUE).any(dim=0)
        if valid_columns.all():
            return context
        if not valid_columns.any():
            return context[:, :1]
        length = valid_columns.nonzero().flatten()[-1].item() + 1
        return context[:, :length]

    def fit(self, data: Dict[freerec.data.fields.Field, torch.Tensor]) -> Dict[str, torch.Tensor]:
        targets = data[self.IPos].flatten()
        context_ids, target_ids = self.encode_item_ids(data[self.ISeq], targets)
        context_mask = context_ids.ne(-1)

        item_embeddings = self.t5.item_embeddings(targets)
        item_latents = self.id_encoder(item_embeddings)
        item_quantized, vq_loss, _ = self.id_quantizer(
            item_latents,
            gumbel_temperature=cfg.gumbel_tau,
        )
        recon_loss = F.mse_loss(item_quantized, item_latents, reduction="sum") / len(item_latents)

        logits = self.t5(
            context_ids=context_ids,
            context_mask=context_mask,
            target_ids=target_ids,
        )
        code_loss = F.cross_entropy(
            logits.reshape(-1, self.num_codewords),
            target_ids.reshape(-1),
        )
        code_loss = self.apply_uncertainty_loss(code_loss)
        rec_loss = (
            cfg.code_loss_weight * code_loss
            + cfg.recon_loss_weight * recon_loss
            + cfg.vq_loss_weight * vq_loss
        )
        losses = {
            "rec_loss": rec_loss,
            "code_loss": code_loss,
            "recon_loss": recon_loss,
            "vq_loss": vq_loss,
            "sigma": self.mean_sigma(),
        }
        return losses

    def mean_sigma(self) -> torch.Tensor:
        return torch.stack([auto_sigma.sigma for auto_sigma in self.id_quantizer.autosigmas]).mean()

    def apply_uncertainty_loss(self, code_loss: torch.Tensor) -> torch.Tensor:
        sigma = self.mean_sigma()
        effective_sigma = (sigma.abs() + cfg.sigma_lambda).clamp(min=1.0e-6)
        return code_loss / (2 * effective_sigma.pow(2)) + torch.log(effective_sigma)

    @torch.no_grad()
    def _generate_full_candidates(
        self,
        data: Dict[freerec.data.fields.Field, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.item_codes is None:
            self.refresh_sem_ids()
        batch_size = data[self.ISeq].size(0)
        context = self.prune_trailing_paddings(data[self.ISeq])
        context_ids = self.item_codes[context].reshape(batch_size, -1)
        context_mask = context_ids.ne(-1)
        return self.t5.generate_candidates(
            context_ids=context_ids,
            context_mask=context_mask,
            num_candidates=cfg.num_beams,
        )

    def decode_candidate_codes(
        self,
        candidate_codes: torch.Tensor,
    ) -> torch.Tensor:
        flat_codes = candidate_codes.reshape(-1, candidate_codes.size(-1))
        candidate_ids = []
        for code in flat_codes.detach().cpu().tolist():
            item_id = self.code_to_item.get(tuple(code), self.PADDING_VALUE)
            candidate_ids.append(item_id)
        return torch.tensor(candidate_ids, dtype=torch.long, device=candidate_codes.device).view(
                candidate_codes.size(0),
                candidate_codes.size(1),
            )

    @torch.no_grad()
    def recommend_from_full(
        self,
        data: Dict[freerec.data.fields.Field, torch.Tensor],
    ) -> torch.Tensor:
        candidate_codes, sequence_scores = self._generate_full_candidates(data)
        candidate_ids = self.decode_candidate_codes(candidate_codes)
        batch_size = candidate_ids.size(0)
        scores = torch.rand(
            (batch_size, self.Item.count + self.NUM_PADS),
            device=candidate_ids.device,
        ).mul_(BACKGROUND_SCORE_MAX)
        raised_scores = (
            sequence_scores - sequence_scores.min(dim=1, keepdim=True).values + BEAM_SCORE_BASE
        )
        scores = scores.scatter(dim=1, index=candidate_ids, src=raised_scores)
        return scores[:, self.NUM_PADS :]

    @torch.no_grad()
    def recommend_from_pool(
        self,
        data: Dict[freerec.data.fields.Field, torch.Tensor],
    ) -> torch.Tensor:
        if self.item_codes is None:
            self.refresh_sem_ids()

        candidates = data[self.IUnseen]
        batch_size, candidate_count = candidates.shape
        context = self.prune_trailing_paddings(data[self.ISeq])
        context_ids = self.item_codes[context].reshape(batch_size, -1)
        context_mask = context_ids.ne(-1)
        target_ids = self.item_codes[candidates + self.NUM_PADS].reshape(
            batch_size * candidate_count,
            -1,
        )
        logits = self.t5(
            context_ids=context_ids.repeat_interleave(candidate_count, dim=0),
            context_mask=context_mask.repeat_interleave(candidate_count, dim=0),
            target_ids=target_ids,
        )
        candidate_scores = (
            F.log_softmax(logits, dim=-1)
            .gather(dim=-1, index=target_ids.unsqueeze(-1))
            .squeeze(-1)
            .sum(dim=-1)
        )
        return candidate_scores.view(batch_size, candidate_count)

    def forward(self, data: Dict, ranking: Literal["pool", "full"] = "full"):
        with torch.amp.autocast("cuda", dtype=DTYPE, enabled=self.device.type == "cuda"):
            return super().forward(data, ranking)


class CoachForDIGER(freerec.launcher.Coach):

    def set_optimizer(self) -> None:
        model = self.get_res_sys_arch()
        lr_rec = cfg.lr if cfg.lr_rec is None else cfg.lr_rec
        lr_id = lr_rec if cfg.lr_id is None else cfg.lr_id

        id_params, sigma_params = [], []
        id_modules = {
            "encoder": model.id_encoder,
            "quantizer": model.id_quantizer,
        }
        for module_name, module in id_modules.items():
            for name, param in module.named_parameters():
                if not param.requires_grad:
                    continue
                full_name = f"{module_name}.{name}"
                if "sigma" in full_name.lower() and cfg.lr_sigma is not None:
                    sigma_params.append(param)
                else:
                    id_params.append(param)

        groups = [
            {"params": model.t5.parameters(), "lr": lr_rec, "name": "t5"},
            {"params": id_params, "lr": lr_id, "name": "id_tokenizer"},
        ]
        if sigma_params:
            groups.append({"params": sigma_params, "lr": cfg.lr_sigma, "name": "sigma"})

        if cfg.optimizer.lower() == "adamw":
            self.optimizer = torch.optim.AdamW(
                groups,
                betas=(cfg.optim_first_moment_decay, cfg.optim_second_moment_decay),
                weight_decay=cfg.weight_decay,
            )
        elif cfg.optimizer.lower() == "adam":
            self.optimizer = torch.optim.Adam(
                groups,
                betas=(cfg.optim_first_moment_decay, cfg.optim_second_moment_decay),
                weight_decay=cfg.weight_decay,
            )
        else:
            raise NotImplementedError(f"unexpected optimizer {cfg.optimizer!r}")

    def set_other(self) -> None:
        self.get_res_sys_arch().refresh_sem_ids(verbose=True)

    def train_per_epoch(self, epoch: int) -> None:
        model = self.get_res_sys_arch()

        for data in self.dataloader:
            data = self.dict_to_device(data)
            losses = self.model(data)
            loss = losses["rec_loss"]

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            self.optimizer.step()

            self.monitor(
                loss.detach().item(),
                n=len(data[self.User]),
                reduction="mean",
                mode="train",
                pool=["LOSS"],
            )

        model.refresh_sem_ids(verbose=True)

    def evaluate(self, epoch, step = -1, mode = "valid"):
        self.get_res_sys_arch().refresh_sem_ids(verbose=True)
        return super().evaluate(epoch, step, mode)


def main() -> None:
    dataset: freerec.data.datasets.RecDataSet
    try:
        dataset = getattr(freerec.data.datasets, cfg.dataset)(root=cfg.root)
    except AttributeError:
        dataset = freerec.data.datasets.RecDataSet(
            cfg.root,
            cfg.dataset,
            tasktag=cfg.tasktag,
        )

    model = DIGER(dataset)
    trainpipe = model.sure_trainpipe(cfg.maxlen, cfg.batch_size)
    validpipe = model.sure_validpipe(cfg.maxlen, ranking=cfg.ranking, batch_size=16)
    testpipe = model.sure_testpipe(cfg.maxlen, ranking=cfg.ranking, batch_size=16)

    coach = CoachForDIGER(
        dataset=dataset,
        trainpipe=trainpipe,
        validpipe=validpipe,
        testpipe=testpipe,
        model=model,
        cfg=cfg,
    )
    coach.fit()


if __name__ == "__main__":
    main()
