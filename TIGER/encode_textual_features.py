import argparse
import html
import os
import re
from typing import Any, List, Optional, Sequence, Tuple

import freerec
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

__all__ = ["TextualFeatureEncoder"]

DEFAULT_FIELDS = ("TITLE", "CATEGORIES", "BRAND")
DEFAULT_ROOT = "../../data"
DEFAULT_DATASET = "Amazon2014Beauty_550_LOU"
DEFAULT_MODEL = "sentence-t5-xl"


class TextualFeatureEncoder:
    r"""Encode item textual fields into dense feature tensors.

    Parameters
    ----------
    root : str
        Root directory that contains the ``Processed`` dataset folder.
    dataset : str
        Processed dataset name.
    model : str
        Local model name under ``model_dir`` or a direct Hugging Face model id when
        ``model_dir`` is empty.
    model_dir : str
        Directory that stores local pretrained models.
    fields : Sequence[str]
        Item fields used to build textual inputs.

    Workflow
    --------
    1. load ``item.txt`` from the processed dataset directory.
    2. build one text string per item from selected fields.
    3. encode texts with SentenceTransformer's default sentence embedding.
    4. export the resulting tensor with ``freerec.utils.export_pickle``.
    """

    def __init__(
        self,
        root: str = DEFAULT_ROOT,
        dataset: str = DEFAULT_DATASET,
        model: str = DEFAULT_MODEL,
        model_dir: str = "./models",
        fields: Sequence[str] = DEFAULT_FIELDS,
        batch_size: int = 128,
        device: Optional[str] = None,
        item_file: str = "item.txt",
        output_file: Optional[str] = None,
    ) -> None:
        self.root = root
        self.dataset = dataset
        self.model = model
        self.model_dir = model_dir
        self.fields = tuple(fields)
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.item_file = item_file
        self.output_file = output_file

    @property
    def dataset_path(self) -> str:
        return os.path.join(self.root, "Processed", self.dataset)

    @property
    def model_path(self) -> str:
        if self.model_dir:
            return os.path.join(self.model_dir, self.model)
        return self.model

    def run(self) -> Tuple[torch.Tensor, str]:
        item_df = self.load_items()
        fields = self.resolve_fields(item_df)
        texts = self.build_texts(item_df, fields)
        features = self.encode_texts(texts)

        if features.size(0) != len(item_df):
            raise RuntimeError("encoded feature count does not match item count.")

        output_file = self.resolve_output_file(fields)
        freerec.utils.export_pickle(features.float(), output_file)
        return features, output_file

    def load_items(self) -> pd.DataFrame:
        return pd.read_csv(os.path.join(self.dataset_path, self.item_file), sep="\t")

    def resolve_fields(self, item_df: pd.DataFrame) -> Tuple[str, ...]:
        fields = self.fields
        missing = [field for field in fields if field not in item_df.columns]
        if missing:
            raise ValueError(f"missing item fields: {missing}.")
        if not fields:
            raise ValueError("at least one textual field is required.")
        return tuple(fields)

    def build_texts(self, item_df: pd.DataFrame, fields: Sequence[str]) -> List[str]:
        item_df = item_df.loc[:, list(fields)]
        texts = []
        for values in item_df.itertuples(index=False, name=None):
            texts.append("\n".join(f"{field}: {self.clean_text(value)}." for field, value in zip(fields, values)))
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
    def encode_texts(self, texts: Sequence[str]) -> torch.Tensor:
        encoder = SentenceTransformer(self.model_path, device=self.device).eval()
        features = encoder.encode(
            list(texts),
            convert_to_tensor=True,
            batch_size=self.batch_size,
            show_progress_bar=True,
        )
        return features.detach().cpu().float()

    def resolve_output_file(self, fields: Sequence[str]) -> str:
        if self.output_file is not None:
            if os.path.isabs(self.output_file):
                return self.output_file
            return os.path.join(self.dataset_path, self.output_file)

        field_part = "_".join(field.lower() for field in fields)
        filename = f"{self.safe_filename(self.model)}_{field_part}.pkl"
        return os.path.join(self.dataset_path, filename.lower())

    @staticmethod
    def safe_filename(value: str) -> str:
        return value.replace("/", "_").replace("\\", "_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Encode item textual features.")
    parser.add_argument("--root", type=str, default=DEFAULT_ROOT)
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET)
    parser.add_argument("--item-file", type=str, default="item.txt")
    parser.add_argument("--fields", type=str, nargs="+", default=DEFAULT_FIELDS)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--model-dir", type=str, default="./models")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-file", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    encoder = TextualFeatureEncoder(
        root=args.root,
        dataset=args.dataset,
        model=args.model,
        model_dir=args.model_dir,
        fields=args.fields,
        batch_size=args.batch_size,
        device=args.device,
        item_file=args.item_file,
        output_file=args.output_file,
    )
    features, output_file = encoder.run()
    print(f"Exported textual features with shape {tuple(features.shape)} to {output_file}.")


if __name__ == "__main__":
    main()
