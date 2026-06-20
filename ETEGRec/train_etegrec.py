import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import freerec
import numpy as np
import torch
import torch.nn.functional as F
from freerec.data.tags import ID, ITEM
from modules import ETEGRecGenerator, ETEGRecTokenizer
from transformers import T5Config, T5ForConditionalGeneration

freerec.declare(version="1.0.1")


cfg = freerec.parser.Parser()

# Data
cfg.add_argument("--maxlen", type=int, default=50, help="maximum item history length")
cfg.add_argument(
    "--sem-feat-file",
    type=str,
    default=None,
    help="semantic item embedding file, relative to dataset.path unless absolute",
)
cfg.add_argument(
    "--finetune-epochs",
    type=int,
    default=100,
    help="extra ETEGRec finetuning epochs with code loss only; 0 disables it",
)

# T5
cfg.add_argument("--embedding-dim", type=int, default=128, help="T5 d_model")
cfg.add_argument("--attention-size", type=int, default=64, help="T5 d_kv")
cfg.add_argument("--intermediate-size", type=int, default=256, help="T5 d_ff")
cfg.add_argument("--num-heads", type=int, default=4, help="number of attention heads")
cfg.add_argument("--num-layers", type=int, default=6, help="number of encoder layers")
cfg.add_argument(
    "--num-decoder-layers",
    type=int,
    default=6,
    help="number of decoder layers",
)
cfg.add_argument("--dropout-rate", type=float, default=0.1, help="T5 dropout")
cfg.add_argument("--activation-function", type=str, default="relu")
cfg.add_argument("--feed-forward-proj", type=str, default="relu")
cfg.add_argument("--num-beams", type=int, default=10, help="beam width")

# Semantic ID / RQ-VAE
cfg.add_argument("--semantic-hidden-size", type=int, default=768)
cfg.add_argument("--code-num", type=int, default=256)
cfg.add_argument("--code-length", type=int, default=4)
cfg.add_argument("--num-emb-list", type=str, default="256,256,256")
cfg.add_argument("--e-dim", type=int, default=32)
cfg.add_argument("--layers", type=str, default="512,256,128")
cfg.add_argument("--alpha", type=float, default=1.0)
cfg.add_argument("--commit-weight", type=float, default=0.5)
cfg.add_argument("--apply-shared-codebook", type=eval, default=False)
cfg.add_argument("--sk-epsilons", type=str, default="0.,0.,0.")
cfg.add_argument("--sk-iters", type=int, default=50)
cfg.add_argument("--kmeans-init", type=eval, default=False)
cfg.add_argument("--kmeans-iters", type=int, default=100)
cfg.add_argument("--dropout-prob", type=float, default=0.1)
cfg.add_argument("--rqvae-path", type=str, default=None)

# ETEGRec alternating training weights
cfg.add_argument("--lr-rec", type=float, default=5e-4)
cfg.add_argument("--lr-id", type=float, default=5e-4)
cfg.add_argument("--cycle", type=int, default=4)
cfg.add_argument("--warm-epoch", type=int, default=10)
cfg.add_argument("--base-auxiliary-loss", type=float, default=1.0)
cfg.add_argument("--id-vq-loss", type=float, default=1.0)
cfg.add_argument("--id-code-loss", type=float, default=0.0)
cfg.add_argument("--id-kl-loss", type=float, default=1e-4)
cfg.add_argument("--id-dec-cl-loss", type=float, default=3e-4)
cfg.add_argument("--rec-vq-loss", type=float, default=0.0)
cfg.add_argument("--rec-code-loss", type=float, default=1.0)
cfg.add_argument("--rec-kl-loss", type=float, default=1e-4)
cfg.add_argument("--rec-dec-cl-loss", type=float, default=3e-4)
cfg.add_argument("--sim", type=str, default="cos", choices=("cos", "dot"))

cfg.set_defaults(
    description="ETEGRec",
    root="../../data",
    dataset="Amazon2014Beauty_550_LOU",
    epochs=400,
    batch_size=512,
    weight_decay=0.05,
    ranking="full",
    monitors=[
        "Recall@1",
        "Recall@5",
        "NDCG@5",
        "Recall@10",
        "NDCG@10",
    ],
    which4best="NDCG@10",
    eval_freq=8,
    seed=2020,
)
cfg.compile()

cfg.num_emb_list = list(map(int, cfg.num_emb_list.split(",")))
cfg.layers = list(map(int, cfg.layers.split(","))) if cfg.layers else []
cfg.sk_epsilons = list(map(float, cfg.sk_epsilons.split(",")))
assert cfg.code_length == len(cfg.num_emb_list) + 1, (
    "ETEGRec training expects `code_length` to equal "
    "`len(num_emb_list) + 1`, where the last token resolves collisions"
)
assert len(cfg.sk_epsilons) == len(cfg.num_emb_list), (
    "`sk_epsilons` should contain one value for each tokenizer codebook"
)
assert cfg.cycle > 0 and cfg.eval_freq % cfg.cycle == 0, (
    "for ETEGRec parity, eval_freq should be divisible by cycle"
)


def load_pretrained_tokenizer(model: ETEGRecTokenizer, path: str) -> None:
    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model_id" in checkpoint:
        checkpoint = checkpoint["model_id"]
    elif isinstance(checkpoint, dict) and "model" in checkpoint:
        checkpoint = checkpoint["model"]

    state_dict = {}
    for key, value in checkpoint.items():
        while key.startswith("module."):
            key = key[len("module.") :]
        if key.startswith("Item.embeddings."):
            continue
        if key.startswith("quantizer.codebooks."):
            parts = key.split(".")
            key = f"rq.vq_layers.{parts[2]}.embedding.{parts[3]}"
        state_dict[key] = value

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if unexpected:
        freerec.utils.infoLogger(
            f"[ETEGRec] >>> ignored unexpected RQ-VAE checkpoint keys: {unexpected}"
        )
    if missing:
        freerec.utils.infoLogger(
            f"[ETEGRec] >>> missing RQ-VAE checkpoint keys: {missing}"
        )


def resolve_semantic_path(dataset: freerec.data.datasets.RecDataSet) -> str:
    if cfg.sem_feat_file is None:
        raise ValueError("--sem-feat-file is required for train_etegrec.py")
    if os.path.isabs(cfg.sem_feat_file):
        return cfg.sem_feat_file
    return os.path.join(dataset.path, cfg.sem_feat_file)


def load_semantic_embedding(dataset: freerec.data.datasets.RecDataSet) -> torch.Tensor:
    path = resolve_semantic_path(dataset)
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        features = np.load(path)
    else:
        features = freerec.utils.import_pickle(path)
    features = torch.as_tensor(features, dtype=torch.float)
    if features.size(0) != dataset.fields[ITEM, ID].count:
        raise ValueError(
            f"semantic features contain {features.size(0)} rows, but dataset has "
            f"{dataset.fields[ITEM, ID].count} items"
        )
    return features


class ETEGRec(freerec.models.SeqRecArch):
    """freerec wrapper for ETEGRec's jointly trained code-T5 recommender."""

    NUM_PADS = 1
    PADDING_VALUE = 0

    def __init__(self, dataset: freerec.data.datasets.RecDataSet) -> None:
        super().__init__(dataset)

        semantic_features = load_semantic_embedding(dataset)
        if semantic_features.size(1) != cfg.semantic_hidden_size:
            raise ValueError(
                f"--semantic-hidden-size={cfg.semantic_hidden_size} does not match "
                f"semantic feature dim {semantic_features.size(1)}"
            )

        model_config = T5Config(
            num_layers=cfg.num_layers,
            num_decoder_layers=cfg.num_decoder_layers,
            d_model=cfg.embedding_dim,
            d_ff=cfg.intermediate_size,
            num_heads=cfg.num_heads,
            d_kv=cfg.attention_size,
            dropout_rate=cfg.dropout_rate,
            activation_function=cfg.activation_function,
            vocab_size=1,
            pad_token_id=0,
            eos_token_id=cfg.code_num + cfg.code_length,
            decoder_start_token_id=0,
            feed_forward_proj=cfg.feed_forward_proj,
            n_positions=cfg.maxlen * cfg.code_length,
        )
        t5 = T5ForConditionalGeneration(config=model_config)
        model_cfg = self.etegrec_config()
        self.model_rec = ETEGRecGenerator(
            config=model_cfg,
            model=t5,
            n_items=self.Item.count + self.NUM_PADS,
            code_length=cfg.code_length,
            code_number=cfg.code_num,
        )
        self.model_id = ETEGRecTokenizer(
            config=model_cfg,
            in_dim=cfg.semantic_hidden_size,
        )

        self.model_rec.semantic_embedding.weight.data.zero_()
        self.model_rec.semantic_embedding.weight.data[self.NUM_PADS :] = semantic_features

        if cfg.rqvae_path is not None:
            load_pretrained_tokenizer(self.model_id, cfg.rqvae_path)

        self.register_buffer(
            "all_item_code",
            torch.full(
                (self.Item.count + self.NUM_PADS, cfg.code_length),
                -1,
                dtype=torch.long,
            ),
        )
        self.code_to_items: Dict[Tuple[int, ...], List[int]] = {}
        self.train_id = False
        self.refresh_item_codes(verbose=False)

    @staticmethod
    def etegrec_config() -> Dict:
        return {
            "semantic_hidden_size": cfg.semantic_hidden_size,
            "code_num": cfg.code_num,
            "code_length": cfg.code_length,
            "num_beams": cfg.num_beams,
            "layers": cfg.layers,
            "e_dim": cfg.e_dim,
            "num_emb_list": cfg.num_emb_list,
            "alpha": cfg.alpha,
            "commit_weight": cfg.commit_weight,
            "apply_shared_codebook": cfg.apply_shared_codebook,
            "sk_epsilons": cfg.sk_epsilons,
            "sk_iters": cfg.sk_iters,
            "kmeans_init": cfg.kmeans_init,
            "kmeans_iters": cfg.kmeans_iters,
            "dropout_prob": cfg.dropout_prob,
        }

    def to(self, *args, **kwargs):
        module = super().to(*args, **kwargs)
        self.model_rec.device = next(self.parameters()).device
        return module

    def sure_trainpipe(self, maxlen: int, batch_size: int):
        return (
            self.dataset.train()
            # align with TIGER rolling samples; ETEGRec source uses only one sequence per user.
            .shuffled_roll_seqs_source(
                minlen=2, maxlen=maxlen + 1, keep_at_least_itself=True
            )
            .seq_train_yielding_pos_(start_idx_for_target=-1, end_idx_for_input=-1)
            .add_(offset=self.NUM_PADS, modified_fields=(self.ISeq, self.IPos))
            .lpad_(maxlen, modified_fields=(self.ISeq,), padding_value=self.PADDING_VALUE)
            .batch_(batch_size)
            .tensor_()
        )

    def sure_validpipe(self, maxlen: int, ranking: str = "full", batch_size: int = 512):
        return (
            self.dataset.valid()
            .ordered_user_ids_source()
            .valid_sampling_(ranking)
            .lprune_(maxlen, modified_fields=(self.ISeq,))
            .add_(offset=self.NUM_PADS, modified_fields=(self.ISeq,))
            .lpad_(maxlen, modified_fields=(self.ISeq,), padding_value=self.PADDING_VALUE)
            .batch_(batch_size)
            .tensor_()
        )

    def sure_testpipe(self, maxlen: int, ranking: str = "full", batch_size: int = 512):
        return (
            self.dataset.test()
            .ordered_user_ids_source()
            .test_sampling_(ranking)
            .lprune_(maxlen, modified_fields=(self.ISeq,))
            .add_(offset=self.NUM_PADS, modified_fields=(self.ISeq,))
            .lpad_(maxlen, modified_fields=(self.ISeq,), padding_value=self.PADDING_VALUE)
            .batch_(batch_size)
            .tensor_()
        )

    @torch.no_grad()
    def refresh_item_codes(self, verbose: bool = True) -> torch.Tensor:
        self.model_rec.eval()
        self.model_id.eval()

        semantic_embs = self.model_rec.semantic_embedding.weight.data[self.NUM_PADS :]
        prefix_codes = self.model_id.get_indices(semantic_embs).detach().cpu().tolist()

        tokens2item = defaultdict(list)
        item_codes = [[-1] * cfg.code_length]
        max_conflict = 0
        for item_id, code in enumerate(prefix_codes):
            key = " ".join(map(str, code))
            tokens2item[key].append(item_id)
            item_codes.append(code + [len(tokens2item[key]) - 1])
            max_conflict = max(max_conflict, len(tokens2item[key]))

        if max_conflict > cfg.code_num:
            raise ValueError(
                f"RQ-VAE semantic ID conflict exceeds codebook size: "
                f"{max_conflict} > {cfg.code_num}"
            )

        code_tensor = torch.tensor(
            item_codes,
            dtype=torch.long,
            device=self.all_item_code.device,
        )
        self.all_item_code.resize_as_(code_tensor).copy_(code_tensor)
        self.code_to_items = defaultdict(list)
        for item_id, code in enumerate(item_codes[self.NUM_PADS :]):
            self.code_to_items[tuple(code)].append(item_id)

        if verbose:
            unique_codes = len({tuple(code) for code in item_codes[self.NUM_PADS :]})
            freerec.utils.infoLogger(
                f"[ETEGRec] >>> refreshed item codes: "
                f"{unique_codes}/{self.Item.count} unique, max conflict {max_conflict}"
            )
        return self.all_item_code

    def code_inputs(
        self,
        item_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = item_ids.size(0)
        code_ids = self.all_item_code[item_ids].contiguous().view(batch_size, -1)
        attention_mask = code_ids.ne(-1)
        return code_ids, attention_mask

    def set_train_stage(self, train_id: bool) -> None:
        self.train_id = train_id

    @staticmethod
    def compute_discrete_contrastive_loss_kl(x_logits, y_logits):
        code_num = x_logits.size(-1)
        x_logits = F.log_softmax(x_logits.view(-1, code_num), dim=-1)
        y_logits = F.log_softmax(y_logits.view(-1, code_num), dim=-1)
        return F.kl_div(x_logits, y_logits, reduction="batchmean", log_target=True)

    @staticmethod
    def compute_contrastive_loss(query_embeds, semantic_embeds, temperature=0.07, sim="cos"):
        if sim == "cos":
            query_embeds = F.normalize(query_embeds, dim=-1)
            semantic_embeds = F.normalize(semantic_embeds, dim=-1)
        labels = torch.arange(query_embeds.size(0), dtype=torch.long, device=query_embeds.device)
        similarities = torch.matmul(query_embeds, semantic_embeds.t()) / temperature
        return F.cross_entropy(similarities, labels)

    @staticmethod
    def first_unique_indices(inputs: torch.Tensor) -> torch.Tensor:
        _, indices = np.unique(inputs.detach().cpu().numpy(), return_index=True)
        return torch.tensor(indices, dtype=torch.long, device=inputs.device)

    def fit(self, data: Dict[freerec.data.fields.Field, torch.Tensor]):
        input_ids, attention_mask = self.code_inputs(data[self.ISeq])
        labels = self.all_item_code[data[self.IPos]].contiguous().view(input_ids.size(0), -1)
        targets = data[self.IPos].flatten()
        unique_indices = self.first_unique_indices(targets)

        target_semantic_embs = self.model_rec.semantic_embedding(targets)
        target_recon_embs, commit_loss, _, _, target_code_logits = self.model_id(
            target_semantic_embs
        )

        outputs = self.model_rec(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        logits = outputs.logits
        seq_project_latents = outputs.seq_project_latents
        dec_latents = outputs.dec_latents
        _, _, _, _, seq_code_logits = self.model_id.rq(seq_project_latents)

        code_loss = F.cross_entropy(
            logits.view(-1, cfg.code_num),
            labels.detach().reshape(-1),
        )
        kl_loss = self.compute_discrete_contrastive_loss_kl(
            seq_code_logits[unique_indices],
            target_code_logits[unique_indices],
        ) + self.compute_discrete_contrastive_loss_kl(
            target_code_logits[unique_indices],
            seq_code_logits[unique_indices],
        )
        dec_cl_loss = self.compute_contrastive_loss(
            target_recon_embs[unique_indices],
            dec_latents[unique_indices],
            sim=cfg.sim,
        ) + self.compute_contrastive_loss(
            dec_latents[unique_indices],
            target_recon_embs[unique_indices],
            sim=cfg.sim,
        )

        losses = {
            "code_loss": code_loss,
            "kl_loss": kl_loss,
            "dec_cl_loss": dec_cl_loss,
        }

        if self.train_id:
            unique_targets = targets[unique_indices]
            unique_semantic_embs = self.model_rec.semantic_embedding(unique_targets)
            unique_recon_embs, commit_loss, _, _, _ = self.model_id(unique_semantic_embs)
            recon_loss = (
                F.mse_loss(unique_recon_embs, unique_semantic_embs, reduction="sum")
                / unique_semantic_embs.size(0)
            ) # align with TIGER; /feature_dim in official implementation
            losses["vq_loss"] = recon_loss + cfg.alpha * commit_loss
        else:
            losses["vq_loss"] = torch.zeros((), dtype=code_loss.dtype, device=code_loss.device)

        return losses

    @torch.no_grad()
    def _generated_item_scores(self, data: Dict[freerec.data.fields.Field, torch.Tensor]):
        input_ids, attention_mask = self.code_inputs(data[self.ISeq])
        sequences, sequence_scores = self.model_rec.my_beam_search(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=cfg.code_length + 1,
            num_beams=cfg.num_beams,
            num_return_sequences=cfg.num_beams,
            return_score=True,
        )
        preds = sequences[:, 1:].view(input_ids.size(0), cfg.num_beams, cfg.code_length)
        sequence_scores = sequence_scores.view(input_ids.size(0), cfg.num_beams)
        scores = torch.full(
            (input_ids.size(0), self.Item.count),
            -1.0e23,
            dtype=torch.float,
            device=input_ids.device,
        )
        for row, row_codes in enumerate(preds.detach().cpu().tolist()):
            for rank, code in enumerate(row_codes):
                for item_id in self.code_to_items.get(tuple(code), []):
                    scores[row, item_id] = torch.maximum(
                        scores[row, item_id],
                        sequence_scores[row, rank],
                    )
        return scores

    @torch.no_grad()
    def recommend_from_full(
        self,
        data: Dict[freerec.data.fields.Field, torch.Tensor],
    ) -> torch.Tensor:
        return self._generated_item_scores(data)

    @torch.no_grad()
    def recommend_from_pool(
        self,
        data: Dict[freerec.data.fields.Field, torch.Tensor],
    ) -> torch.Tensor:
        input_ids, attention_mask = self.code_inputs(data[self.ISeq])
        candidates = data[self.IUnseen]
        candidate_count = candidates.size(1)
        labels = self.all_item_code[candidates + self.NUM_PADS].view(-1, cfg.code_length)
        outputs = self.model_rec(
            input_ids=input_ids.repeat_interleave(candidate_count, dim=0),
            attention_mask=attention_mask.repeat_interleave(candidate_count, dim=0),
            labels=labels,
        )
        token_logp = (
            F.log_softmax(outputs.logits, dim=-1)
            .gather(dim=-1, index=labels.unsqueeze(-1))
            .squeeze(-1)
        )
        return token_logp.sum(dim=-1).view(candidates.size(0), candidate_count)


class CoachForETEGRec(freerec.launcher.Coach):
    def set_optimizer(self):
        model = self.get_res_sys_arch()
        self.rec_optimizer = torch.optim.AdamW(
            model.model_rec.parameters(),
            lr=self.cfg.lr_rec,
            weight_decay=self.cfg.weight_decay,
        )
        self.id_optimizer = torch.optim.AdamW(
            model.model_id.parameters(),
            lr=self.cfg.lr_id,
            weight_decay=0.0, # align with TIGER; =self.cfg.weight_decay in official implementation
        )
        self.optimizer = self.rec_optimizer

    def set_other(self):
        self.register_metric("CODE_LOSS", lambda x: x, best_caster=min)
        self.register_metric("VQ_LOSS", lambda x: x, best_caster=min)
        self.register_metric("KL_LOSS", lambda x: x, best_caster=min)
        self.register_metric("DEC_CL_LOSS", lambda x: x, best_caster=min)

    def _set_trainable(self, train_id: bool) -> None:
        model = self.get_res_sys_arch()
        for param in model.model_id.parameters():
            param.requires_grad = train_id
        for name, param in model.model_rec.named_parameters():
            param.requires_grad = (not train_id) and not name.startswith("semantic_embedding")

    def _loss_weights(self, epoch: int, train_id: bool) -> Dict[str, float]:
        warmed = epoch >= self.cfg.warm_epoch
        aux_weight = self.cfg.base_auxiliary_loss
        if train_id:
            return {
                "vq_loss": self.cfg.id_vq_loss,
                "code_loss": self.cfg.id_code_loss if warmed else 0.0,
                "kl_loss": self.cfg.id_kl_loss * aux_weight if warmed else 0.0,
                "dec_cl_loss": self.cfg.id_dec_cl_loss * aux_weight if warmed else 0.0,
            }
        return {
            "vq_loss": self.cfg.rec_vq_loss,
            "code_loss": self.cfg.rec_code_loss,
            "kl_loss": self.cfg.rec_kl_loss * aux_weight if warmed else 0.0,
            "dec_cl_loss": self.cfg.rec_dec_cl_loss * aux_weight if warmed else 0.0,
        }

    def _train_epoch(self, epoch: int, train_id: bool):
        model = self.get_res_sys_arch()
        model.set_train_stage(train_id)
        self._set_trainable(train_id)
        model.model_id.train(train_id)
        model.model_rec.train(not train_id)
        optimizer = self.id_optimizer if train_id else self.rec_optimizer
        weights = self._loss_weights(epoch, train_id)

        optimizer.zero_grad()
        for data in self.dataloader:
            data = self.dict_to_device(data)
            losses = self.model(data)
            loss = sum(losses[name] * weights[name] for name in weights)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.model_id.parameters() if train_id else model.model_rec.parameters(),
                1.0,
            )
            optimizer.step()
            optimizer.zero_grad()

            batch_size = len(data[model.User]) if model.User in data else data[model.ISeq].size(0)
            self.monitor(loss.item(), n=batch_size, mode="train", pool=["LOSS"])
            self.monitor(
                losses["code_loss"].item(),
                n=batch_size,
                mode="train",
                pool=["CODE_LOSS"],
            )
            self.monitor(
                losses["vq_loss"].item(),
                n=batch_size,
                mode="train",
                pool=["VQ_LOSS"],
            )
            self.monitor(
                losses["kl_loss"].item(),
                n=batch_size,
                mode="train",
                pool=["KL_LOSS"],
            )
            self.monitor(
                losses["dec_cl_loss"].item(),
                n=batch_size,
                mode="train",
                pool=["DEC_CL_LOSS"],
            )

        if train_id:
            model.refresh_item_codes(verbose=True)

    def train_per_epoch(self, epoch: int):
        etegrec_epoch = epoch - 1
        self._train_epoch(etegrec_epoch, train_id=(etegrec_epoch % self.cfg.cycle == 0))

    def save(self, filename: str = None) -> None:
        if freerec.ddp.is_main_process():
            filename = self.cfg.SAVED_FILENAME if filename is None else filename
            path = os.path.join(self.cfg.LOG_PATH, filename)
            model = self.get_res_sys_arch()
            torch.save(
                {
                    "model_rec": model.model_rec.state_dict(),
                    "model_id": model.model_id.state_dict(),
                    "all_item_code": model.all_item_code.detach().cpu(),
                },
                path,
            )
            with open(path + ".code.json", "w", encoding="utf-8") as file:
                json.dump(model.all_item_code.detach().cpu().tolist(), file)
        freerec.ddp.synchronize()

    def load(self, path: str, filename: str = None) -> None:
        filename = self.cfg.SAVED_FILENAME if filename is None else filename
        checkpoint = torch.load(os.path.join(path, filename), map_location=self.device)
        model = self.get_res_sys_arch()
        model.model_rec.load_state_dict(checkpoint["model_rec"], strict=False)
        model.model_id.load_state_dict(checkpoint["model_id"], strict=False)
        model.all_item_code.copy_(checkpoint["all_item_code"].to(self.device))
        model.refresh_item_codes(verbose=False)
        freerec.ddp.synchronize()

    def save_checkpoint(self, epoch: int) -> None:
        checkpoint = {
            "epoch": epoch,
            "model": self.model.state_dict(),
            "rec_optimizer": self.rec_optimizer.state_dict(),
            "id_optimizer": self.id_optimizer.state_dict(),
            "monitors": self.monitors.state_dict(),
        }
        path = os.path.join(self.cfg.CHECKPOINT_PATH, self.cfg.CHECKPOINT_FILENAME)
        torch.save(checkpoint, path)
        freerec.ddp.synchronize()

    def load_checkpoint(self) -> int:
        path = os.path.join(self.cfg.CHECKPOINT_PATH, self.cfg.CHECKPOINT_FILENAME)
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model"], strict=False)
        self.rec_optimizer.load_state_dict(checkpoint["rec_optimizer"])
        self.id_optimizer.load_state_dict(checkpoint["id_optimizer"])
        self.monitors.load_state_dict(checkpoint["monitors"])
        self.get_res_sys_arch().refresh_item_codes(verbose=False)
        freerec.ddp.synchronize()
        return checkpoint["epoch"]

    def reset_best_for_finetune(self) -> None:
        self._best = -float("inf") if self.meter4best.caster is max else float("inf")
        self._best_epoch = 0
        self._best_step = -1
        self._stopping_steps = 0
        self._early_stop_patience = 10

    def finetune(self):
        if self.cfg.finetune_epochs <= 0:
            return
        freerec.utils.infoLogger("[Coach] >>> Start ETEGRec code-loss finetuning")
        self.load_best()
        self.reset_best_for_finetune()
        model = self.get_res_sys_arch()
        model.set_train_stage(False)
        self._set_trainable(train_id=False)
        for param in model.model_id.parameters():
            param.requires_grad = False

        self.rec_optimizer = torch.optim.AdamW(
            (p for p in model.model_rec.parameters() if p.requires_grad),
            lr=5e-4,
            weight_decay=self.cfg.weight_decay,
        )

        for epoch in range(self.cfg.finetune_epochs):
            try:
                self.mode = "train"
                self.rec_optimizer.zero_grad()
                for data in self.dataloader:
                    data = self.dict_to_device(data)
                    losses = self.model(data)
                    loss = losses["code_loss"]
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.model_rec.parameters(), 1.0)
                    self.rec_optimizer.step()
                    self.rec_optimizer.zero_grad()
                self.valid(self.cfg.epochs + epoch + 1)
            except freerec.launcher.EarlyStopError:
                freerec.utils.infoLogger(f"[Coach] >>> Finetune Early Stop @Epoch: {epoch}")
                break

    def fit(self):
        start_epoch = self.resume()
        epoch = 0
        try:
            for epoch in range(start_epoch, self.cfg.epochs):
                self.train(epoch + 1)
                if (epoch + 1) % self.cfg.CHECKPOINT_FREQ == 0:
                    self.save_checkpoint(epoch + 1)
                if (epoch + 1) % self.cfg.eval_freq == 0:
                    if self.cfg.eval_valid:
                        self.valid(epoch + 1)
                    if self.cfg.eval_test:
                        self.test(epoch + 1)
        except freerec.launcher.EarlyStopError:
            freerec.utils.infoLogger(f"[Coach] >>> Early Stop @Epoch: {epoch}")
        self.save()
        self.finetune()
        self.save()
        self.test(self.cfg.epochs + self.cfg.finetune_epochs)
        self._stopping_steps = -1
        best = self.summary()
        self.eval_at_best()
        self.easy_record_best(best)
        self.shutdown()


def main():
    dataset: freerec.data.datasets.RecDataSet
    try:
        dataset = getattr(freerec.data.datasets, cfg.dataset)(root=cfg.root)
    except AttributeError:
        dataset = freerec.data.datasets.RecDataSet(
            cfg.root,
            cfg.dataset,
            tasktag=cfg.tasktag,
        )

    model = ETEGRec(dataset)
    trainpipe = model.sure_trainpipe(cfg.maxlen, cfg.batch_size)
    validpipe = model.sure_validpipe(cfg.maxlen, ranking=cfg.ranking, batch_size=16)
    testpipe = model.sure_testpipe(cfg.maxlen, ranking=cfg.ranking, batch_size=16)

    coach = CoachForETEGRec(
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
