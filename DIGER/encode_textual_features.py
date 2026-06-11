import argparse
import html
import os
import re
from typing import Any, List, Optional, Sequence, Tuple

import freerec
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

__all__ = ["TextualFeatureEncoder"]

POOLING_CHOICES = ("mean", "last")


class TextualFeatureEncoder:
    r"""Encode item text with a Hugging Face transformers model.

    Parameters
    ----------
    root : str
        Root directory that contains the processed dataset.
    dataset : str
        Processed dataset name.
    model : str
        Hugging Face model id or local model path.
    pooling : str
        Pooling method over the final hidden states, either ``mean`` or ``last``.
    fields : Sequence[str]
        Item textual fields to encode.

    Workflow
    --------
    1. load item textual fields from ``item.txt``.
    2. encode texts with ``AutoTokenizer`` and ``AutoModel``.
    3. pool final hidden states into one embedding per item.
    4. export a float tensor with shape ``[Item.count, D]``.
    """

    def __init__(
        self,
        root: str,
        dataset: str,
        model: str,
        pooling: str,
        fields: Sequence[str],
        batch_size: int,
        max_length: Optional[int],
        device: Optional[str],
        item_file: str,
        output_file: Optional[str],
        trust_remote_code: bool,
    ) -> None:
        self.root = root
        self.dataset = dataset
        self.model = model
        self.pooling = pooling
        self.fields = tuple(fields)
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.item_file = item_file
        self.output_file = output_file
        self.trust_remote_code = trust_remote_code

    @property
    def dataset_path(self) -> str:
        return os.path.join(self.root, "Processed", self.dataset)

    def run(self) -> Tuple[torch.Tensor, str]:
        item_df = self.load_items()
        fields = self.resolve_fields(item_df)
        texts = self.build_texts(item_df, fields)
        features = self.encode(texts)

        if features.size(0) != len(item_df):
            raise RuntimeError("encoded feature count does not match item count.")

        output_file = self.resolve_output_file(fields)
        freerec.utils.export_pickle(features.float(), output_file)
        return features, output_file

    def load_items(self) -> pd.DataFrame:
        return pd.read_csv(os.path.join(self.dataset_path, self.item_file), sep="\t")

    def resolve_fields(self, item_df: pd.DataFrame) -> Tuple[str, ...]:
        missing = [field for field in self.fields if field not in item_df.columns]
        if missing:
            raise ValueError(f"missing item fields: {missing}.")
        if not self.fields:
            raise ValueError("at least one textual field is required.")
        return self.fields

    def build_texts(self, item_df: pd.DataFrame, fields: Sequence[str]) -> List[str]:
        item_df = item_df.loc[:, list(fields)].fillna("")
        texts = []
        for values in item_df.itertuples(index=False, name=None):
            lines = []
            for field, value in zip(fields, values):
                text = self.clean_text(value)
                if text:
                    text = text if text[-1] in ".!?" else f"{text}."
                    lines.append(f"{field}: {text}")
            texts.append("\n".join(lines))
        return texts

    @staticmethod
    def clean_text(value: Any) -> str:
        r"""Normalize raw item text before field-level concatenation.

        This only removes formatting noise such as HTML tags, quotes, newlines,
        and repeated whitespace. Long-text truncation is left to the tokenizer.
        """
        if value is None:
            return ""
        if isinstance(value, list):
            text = " ".join(str(item) for item in value)
        elif isinstance(value, dict):
            text = " ".join(f"{key}: {val}" for key, val in value.items())
        elif pd.isna(value):
            return ""
        else:
            text = str(value)

        text = html.unescape(text)
        text = re.sub(r"</?\w+[^>]*>", " ", text)
        text = re.sub(r"[\"\n\r]", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    @torch.no_grad()
    def encode(self, texts: Sequence[str]) -> torch.Tensor:
        tokenizer = AutoTokenizer.from_pretrained(
            self.model,
            trust_remote_code=self.trust_remote_code,
        )
        model = AutoModel.from_pretrained(
            self.model,
            trust_remote_code=self.trust_remote_code,
        ).eval()

        if tokenizer.pad_token is None:
            if tokenizer.eos_token is None:
                tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
                model.resize_token_embeddings(len(tokenizer))
            else:
                tokenizer.pad_token = tokenizer.eos_token

        model = model.to(self.device)
        features = []
        batch_starts = range(0, len(texts), self.batch_size)
        num_batches = (len(texts) + self.batch_size - 1) // self.batch_size
        for start in tqdm(batch_starts, total=num_batches, desc="Encoding", unit="batch"):
            batch_texts = list(texts[start : start + self.batch_size])
            tokenizer_kwargs = {
                "return_tensors": "pt",
                "padding": True,
                "truncation": self.max_length is not None,
            }
            if self.max_length is not None:
                tokenizer_kwargs["max_length"] = self.max_length

            inputs = tokenizer(batch_texts, **tokenizer_kwargs)
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            hidden_states = self.encode_batch(model, inputs)
            pooled = self.pool_hidden_states(hidden_states, inputs["attention_mask"])
            features.append(pooled.detach().cpu())

        return torch.cat(features, dim=0).float()

    def encode_batch(self, model: Any, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        if getattr(model.config, "is_encoder_decoder", False):
            outputs = model.get_encoder()(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                output_hidden_states=True,
                return_dict=True,
            )
        else:
            outputs = model(**inputs, output_hidden_states=True, return_dict=True)
        return outputs.hidden_states[-1]

    def pool_hidden_states(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        attention_mask = attention_mask.to(hidden_states.device)
        if self.pooling == "mean":
            mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
            return hidden_states.mul(mask).sum(dim=1).div(mask.sum(dim=1).clamp_min(1.0))
        if self.pooling == "last":
            positions = torch.arange(hidden_states.size(1), device=hidden_states.device)
            positions = positions.unsqueeze(0).expand_as(attention_mask)
            last_indices = attention_mask.long().mul(positions).argmax(dim=1)
            batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
            return hidden_states[batch_indices, last_indices]
        raise ValueError(f"pooling must be one of {POOLING_CHOICES}, but got {self.pooling!r}.")

    def resolve_output_file(self, fields: Sequence[str]) -> str:
        if self.output_file is not None:
            if os.path.isabs(self.output_file):
                return self.output_file
            return os.path.join(self.dataset_path, self.output_file)

        field_part = "_".join(field.lower() for field in fields)
        filename = f"{self.model_name()}_{field_part}_{self.pooling}.pkl"
        return os.path.join(self.dataset_path, filename.lower())

    def model_name(self) -> str:
        return os.path.basename(os.path.normpath(self.model)).replace("\\", "_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Encode item textual features.")
    parser.add_argument("--root", type=str, default="../../data")
    parser.add_argument("--dataset", type=str, default="Amazon2014Beauty_550_LOU")
    parser.add_argument("--item-file", type=str, default="item.txt")
    parser.add_argument("--fields", type=str, nargs="+", default=("TITLE", "CATEGORIES", "BRAND"))
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--pooling", type=str, choices=POOLING_CHOICES, default="mean")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-file", type=str, default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    encoder = TextualFeatureEncoder(
        root=args.root,
        dataset=args.dataset,
        model=args.model,
        pooling=args.pooling,
        fields=args.fields,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=args.device,
        item_file=args.item_file,
        output_file=args.output_file,
        trust_remote_code=args.trust_remote_code,
    )
    features, output_file = encoder.run()
    print(f"Exported textual features with shape {tuple(features.shape)} to {output_file}.")


if __name__ == "__main__":
    main()
