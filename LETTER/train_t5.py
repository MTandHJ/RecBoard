import json
from typing import Dict, Iterable, List, Literal, Sequence, Tuple

import freerec
import torch
import torch.nn.functional as F
from converter import SemIDConverter, prefix_allowed_tokens_fn
from transformers import T5Config, T5ForConditionalGeneration, T5Tokenizer
from transformers.generation.stopping_criteria import StoppingCriteriaList

DTYPE = torch.bfloat16
BACKGROUND_SCORE_MAX = 1.0e-3
BEAM_SCORE_BASE = 1.0


freerec.declare(version="1.0.1")

cfg = freerec.parser.Parser()
cfg.add_argument("--maxlen", type=int, default=50, help="maximum item sequence length")
cfg.add_argument("--embedding-dim", type=int, default=128, help="T5 d_model")
cfg.add_argument("--attention-size", type=int, default=64, help="T5 d_kv")
cfg.add_argument("--intermediate-size", type=int, default=64 * 4, help="T5 d_ff")
cfg.add_argument("--num-heads", type=int, default=4, help="number of attention heads")
cfg.add_argument("--num-layers", type=int, default=6, help="number of encoder/decoder layers")
cfg.add_argument("--dropout-rate", type=float, default=0.1, help="T5 dropout rate")
cfg.add_argument(
    "--sid-vocab-file",
    type=str,
    default="sid_vocab.json",
    help="SID vocabulary JSON file",
)
cfg.add_argument("--num-beams", type=int, default=20, help="beam width for full ranking")
cfg.add_argument(
    "--apply-constrained-beam-search",
    type=eval,
    default=True,
    help="whether to apply prefix constraints during full-ranking beam search",
)

cfg.set_defaults(
    description="LETTER-T5",
    root="../../data",
    dataset="Amazon2014Beauty_550_LOU",
    epochs=500,
    batch_size=256,
    optimizer="AdamW",
    lr=1e-3,
    weight_decay=0.1,
    seed=1,
)
cfg.compile()


assert cfg.num_beams > 1, "beam_search requires `num_beams` > 1  ..."


class LETTERT5(freerec.models.SeqRecArch):
    r"""Train and rank with TIGER-style semantic-ID protocol text.

    LETTER uses its own quantizer to build item semantic IDs. Once those IDs are
    exported as ``sid_vocab.json``, the T5 recommender follows the same data
    flow as TIGER: item histories are converted to SID text, T5 predicts the
    next item SID block, and freerec evaluates item-level full or pool ranking.
    """

    def __init__(self, dataset: freerec.data.datasets.RecDataSet) -> None:
        super().__init__(dataset)

        with open(cfg.sid_vocab_file, "r", encoding="utf-8") as file:
            sid_vocab = json.load(file)

        self.tokenizer = T5Tokenizer(vocab=None, extra_ids=0)
        self.converter = SemIDConverter(sid_vocab, self.tokenizer)

        model_config = T5Config(
            vocab_size=len(self.tokenizer),
            d_model=cfg.embedding_dim,
            d_kv=cfg.attention_size,
            d_ff=cfg.intermediate_size,
            num_layers=cfg.num_layers,
            num_decoder_layers=cfg.num_layers,
            num_heads=cfg.num_heads,
            dropout_rate=cfg.dropout_rate,
            output_attentions=False,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            decoder_start_token_id=self.tokenizer.pad_token_id,
        )
        self.t5 = T5ForConditionalGeneration(model_config)

        self.generate_kwargs = {
            "num_beams": cfg.num_beams,
            "num_return_sequences": cfg.num_beams,
            "prefix_allowed_tokens_fn": prefix_allowed_tokens_fn(self.converter),
            "stopping_criteria": StoppingCriteriaList(
                [self.converter.stopping_criteria(num_items=1)]
            ),
            "max_new_tokens": self.converter.max_num_sid_tokens + 1,
            "return_dict_in_generate": True,
            "output_scores": True,
        }
        if not cfg.apply_constrained_beam_search:
            del self.generate_kwargs["prefix_allowed_tokens_fn"]

    def format_item_ids(self, field, item_ids: Iterable[int]) -> List[str]:
        return [self.converter.format(item) for item in item_ids]

    def encode_item_text(self, field, items: Iterable[str]) -> str:
        return self.converter.encode(items)

    def tokenize_text(self, texts: Sequence[str]) -> Dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            texts,
            add_special_tokens=False,
            padding=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].to(self.device),
            "attention_mask": encoded["attention_mask"].to(self.device),
        }

    def sure_trainpipe(self, maxlen: int, batch_size: int):
        return (
            self.dataset.train()
            .shuffled_roll_seqs_source(minlen=2, maxlen=maxlen, keep_at_least_itself=True)
            .seq_train_yielding_pos_(start_idx_for_target=-1, end_idx_for_input=-1)
            # [1, 2, 3] -> ['item_1', 'item_2', 'item_3']
            .map_(self.format_item_ids, modified_fields=(self.ISeq, self.IPos))
            # ['item_1', 'item_2', 'item_3'] -> "<SID> <sid_0_0> ... </SID> <SID> ..."
            .map_(self.encode_item_text, modified_fields=(self.ISeq, self.IPos))
            .batch_(batch_size)
            .tensor_()
        )

    def sure_validpipe(self, maxlen: int, ranking: str = "full", batch_size: int = 512):
        return (
            self.dataset.valid()
            .ordered_user_ids_source()
            .valid_sampling_(ranking)
            .lprune_(maxlen, modified_fields=(self.ISeq,))
            .map_(self.format_item_ids, modified_fields=(self.ISeq,))
            .map_(self.encode_item_text, modified_fields=(self.ISeq,))
            .batch_(batch_size)
            .tensor_()
        )

    def sure_testpipe(self, maxlen: int, ranking: str = "full", batch_size: int = 512):
        return (
            self.dataset.test()
            .ordered_user_ids_source()
            .test_sampling_(ranking)
            .lprune_(maxlen, modified_fields=(self.ISeq,))
            .map_(self.format_item_ids, modified_fields=(self.ISeq,))
            .map_(self.encode_item_text, modified_fields=(self.ISeq,))
            .batch_(batch_size)
            .tensor_()
        )

    def fit(self, data: Dict[freerec.data.fields.Field, str]) -> Dict[str, torch.Tensor]:
        context = self.tokenize_text(data[self.ISeq])
        targets = self.tokenize_text(data[self.IPos])
        labels = targets["input_ids"].masked_fill(
            targets["attention_mask"].eq(0),
            -100,  # -100 is ignored by CrossEntropy loss.
        )
        outputs = self.t5(
            input_ids=context["input_ids"],
            attention_mask=context["attention_mask"],
            labels=labels,
            return_dict=True,
        )
        return {"rec_loss": outputs.loss}

    def prepare_decoder_prefix(self, batch_size: int) -> torch.Tensor:
        """Build the forced ``<SID>`` generation prefix for a batch."""
        prefix = self.tokenizer(
            [SemIDConverter.SID_START_TOKEN] * batch_size,
            add_special_tokens=False,
            return_tensors="pt",
        )
        return prefix.input_ids.to(self.device)

    @torch.no_grad()
    def _generate_full_candidates(
        self, data: Dict[freerec.data.fields.Field, str | torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate constrained next-item candidates and sequence scores."""
        context = self.tokenize_text(data[self.ISeq])
        batch_size = context["input_ids"].size(0)

        outputs = self.t5.generate(
            input_ids=context["input_ids"],
            attention_mask=context["attention_mask"],
            decoder_input_ids=self.prepare_decoder_prefix(batch_size),
            **self.generate_kwargs,
        )
        decoded = self.converter.batch_decode(self.tokenizer.batch_decode(outputs.sequences))
        candidate_ids = torch.tensor(
            # self.Item.count is used as a dummy item for invalid Item SIDs
            [items[0] if len(items) == 1 else self.Item.count for items in decoded],
            dtype=torch.long,
            device=self.device,
        ).view(batch_size, cfg.num_beams)
        return candidate_ids, outputs.sequences_scores.view(batch_size, cfg.num_beams)

    @torch.no_grad()
    def recommend_from_full(
        self, data: Dict[freerec.data.fields.Field, str | torch.Tensor]
    ) -> torch.Tensor:
        candidate_ids, sequence_scores = self._generate_full_candidates(data)
        batch_size = candidate_ids.size(0)
        scores = torch.rand(
            (batch_size, self.Item.count + 1),
            device=self.device,
        ).mul_(BACKGROUND_SCORE_MAX)
        raised_scores = (
            sequence_scores - sequence_scores.min(dim=1, keepdim=True).values + BEAM_SCORE_BASE
        )
        return scores.scatter(dim=1, index=candidate_ids, src=raised_scores)[:, : self.Item.count]

    @torch.no_grad()
    def recommend_from_pool(
        self, data: Dict[freerec.data.fields.Field, str | torch.Tensor]
    ) -> torch.Tensor:
        context = self.tokenize_text(data[self.ISeq])
        candidate_rows = data[self.IUnseen].detach().cpu().tolist()
        candidate_rows = [[int(item) for item in row] for row in candidate_rows]
        candidate_count = len(candidate_rows[0])

        target_texts = [
            self.converter.encode(self.converter.format(item))
            for row in candidate_rows
            for item in row
        ]
        targets = self.tokenize_text(target_texts)
        labels = targets["input_ids"].masked_fill(
            targets["attention_mask"].eq(0),
            -100,
        )
        outputs = self.t5(
            input_ids=context["input_ids"].repeat_interleave(candidate_count, dim=0),
            attention_mask=context["attention_mask"].repeat_interleave(candidate_count, dim=0),
            labels=labels,
            return_dict=True,
        )

        token_logp = (
            F.log_softmax(outputs.logits, dim=-1)
            .gather(
                dim=-1,
                index=targets["input_ids"].unsqueeze(-1),
            )
            .squeeze(-1)
        )
        score_mask = targets["attention_mask"].bool()
        score_mask[:, 0] = False  # ``<SID>`` is a forced prefix in full ranking.
        candidate_scores = token_logp.masked_fill(~score_mask, 0.0).sum(dim=-1)
        return candidate_scores.view(len(candidate_rows), candidate_count)

    def forward(self, data: Dict, ranking: Literal["pool", "full"] = "full"):
        with torch.amp.autocast("cuda", dtype=DTYPE, enabled=self.device.type == "cuda"):
            return super().forward(data, ranking)


class CoachForLETTERT5(freerec.launcher.Coach):
    def train_per_epoch(self, epoch: int):
        for data in self.dataloader:
            data = self.dict_to_device(data)
            loss = self.model(data)["rec_loss"]

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.monitor(
                loss.item(),
                n=len(data[self.User]),
                reduction="mean",
                mode="train",
                pool=["LOSS"],
            )


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

    model = LETTERT5(dataset)
    trainpipe = model.sure_trainpipe(cfg.maxlen, cfg.batch_size)
    validpipe = model.sure_validpipe(cfg.maxlen, ranking=cfg.ranking, batch_size=16)
    testpipe = model.sure_testpipe(cfg.maxlen, ranking=cfg.ranking, batch_size=16)

    coach = CoachForLETTERT5(
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
