import argparse
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import freerec
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer

__all__ = ["TextualFeatureEncoder", "main"]

DEFAULT_FIELDS = ("TITLE", "CATEGORIES", "BRAND")
DEFAULT_ROOT = "../../data"
DEFAULT_DATASET = "Amazon2014Beauty_550_LOU"
DEFAULT_MODEL = "sentence-t5-xl"
BACKENDS = ("sentence-transformer", "transformers")
POOLINGS = ("native", "last", "mean")
RESERVED_COLUMNS = {"ITEM", "ITEMID", "ITEM_ID", "ID"}
DECODER_MODEL_HINTS = (
    "bloom",
    "baichuan",
    "chatglm",
    "falcon",
    "gemma",
    "gpt",
    "gpt2",
    "gptj",
    "gpt_neox",
    "llama",
    "mistral",
    "mpt",
    "opt",
    "phi",
    "qwen",
    "rwkv",
)
DECODER_ARCHITECTURE_HINTS = ("causallm", "gpt", "llama", "mistral", "qwen")


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
    backend : str
        Model execution backend, either ``sentence-transformer`` or ``transformers``.
    pooling : Optional[str]
        Pooling strategy. ``native`` keeps SentenceTransformer's original sentence
        embedding, while ``last`` and ``mean`` pool token hidden states manually.
    fields : Optional[Sequence[str]]
        Item fields used to build textual inputs. If ``None``, common textual
        fields are selected when available.

    Workflow
    --------
    1. load ``item.txt`` from the processed dataset directory.
    2. build one text string per item from selected fields.
    3. encode texts with the selected backend and pooling strategy.
    4. export the resulting tensor with ``freerec.utils.export_pickle``.
    """

    def __init__(
        self,
        root: str = DEFAULT_ROOT,
        dataset: str = DEFAULT_DATASET,
        model: str = DEFAULT_MODEL,
        model_dir: str = "./models",
        backend: str = "sentence-transformer",
        pooling: Optional[str] = None,
        fields: Optional[Sequence[str]] = None,
        batch_size: int = 128,
        device: Optional[str] = None,
        device_map: Optional[str] = None,
        item_file: str = "item.txt",
        output_file: Optional[str] = None,
        trust_remote_code: bool = False,
    ) -> None:
        self.root = root
        self.dataset = dataset
        self.model = model
        self.model_dir = model_dir
        self.backend = backend
        self.model_type: Optional[str] = None
        self.pooling = self.resolve_initial_pooling(backend, pooling)
        self.fields = tuple(fields) if fields is not None else None
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device_map = device_map
        self.item_file = item_file
        self.output_file = output_file
        self.trust_remote_code = trust_remote_code

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

        if self.backend == "sentence-transformer":
            features = self.encode_by_sentence_transformer(texts)
        elif self.backend == "transformers":
            features = self.encode_by_transformers(texts)
        else:
            raise ValueError(f"backend must be one of {BACKENDS}, but got {self.backend!r}.")

        if features.size(0) != len(item_df):
            raise RuntimeError("encoded feature count does not match item count.")

        output_file = self.resolve_output_file(fields)
        freerec.utils.export_pickle(features.float(), output_file)
        return features, output_file

    def load_items(self) -> pd.DataFrame:
        return pd.read_csv(os.path.join(self.dataset_path, self.item_file), sep="\t")

    def resolve_fields(self, item_df: pd.DataFrame) -> Tuple[str, ...]:
        if self.fields is not None:
            fields = self.fields
        else:
            fields = tuple(field for field in DEFAULT_FIELDS if field in item_df.columns)
            if not fields:
                fields = tuple(field for field in item_df.columns if field.upper() not in RESERVED_COLUMNS)

        missing = [field for field in fields if field not in item_df.columns]
        if missing:
            raise ValueError(f"missing item fields: {missing}.")
        if not fields:
            raise ValueError("at least one textual field is required.")
        return tuple(fields)

    def build_texts(self, item_df: pd.DataFrame, fields: Sequence[str]) -> List[str]:
        item_df = item_df.loc[:, list(fields)].fillna("")
        texts = []
        for values in item_df.itertuples(index=False, name=None):
            texts.append("\n".join(f"{field}: {value}." for field, value in zip(fields, values)))
        return texts

    @torch.no_grad()
    def encode_by_sentence_transformer(self, texts: Sequence[str]) -> torch.Tensor:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as err:
            raise ImportError("sentence-transformers is required for --backend=sentence-transformer.") from err

        encoder = SentenceTransformer(self.model_path, device=self.device).eval()
        if self.pooling == "native":
            features = encoder.encode(
                list(texts),
                convert_to_tensor=True,
                batch_size=self.batch_size,
                show_progress_bar=True,
            )
            return features.detach().cpu().float()

        batches = []
        for start in tqdm(range(0, len(texts), self.batch_size), desc="Encoding"):
            batch_texts = list(texts[start : start + self.batch_size])
            inputs = self.move_to_device(encoder.tokenize(batch_texts), self.device)
            outputs = encoder(inputs)
            attention_mask = outputs.get("attention_mask", inputs["attention_mask"])
            batches.append(
                self.pool_hidden_states(outputs["token_embeddings"], attention_mask, self.pooling).detach().cpu()
            )
        return torch.cat(batches, dim=0).float()

    @torch.no_grad()
    def encode_by_transformers(self, texts: Sequence[str]) -> torch.Tensor:
        config = AutoConfig.from_pretrained(
            self.model_path,
            trust_remote_code=self.trust_remote_code,
        )
        self.model_type = self.infer_model_type(config)
        self.pooling = self.resolve_transformer_pooling(self.model_type, self.pooling)

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=self.trust_remote_code,
        )
        tokenizer.padding_side = "left" if self.model_type == "decoder-only" else "right"
        tokenizer_kwargs: Dict[str, Any] = {
            "return_tensors": "pt",
            "padding": True,
            "truncation": True,
        }

        model_kwargs: Dict[str, Any] = {"trust_remote_code": self.trust_remote_code}
        if self.device_map is not None:
            model_kwargs["device_map"] = self.device_map
        model = AutoModel.from_pretrained(self.model_path, config=config, **model_kwargs).eval()

        if tokenizer.pad_token is None:
            if tokenizer.eos_token is None:
                tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
                model.resize_token_embeddings(len(tokenizer))
            else:
                tokenizer.pad_token = tokenizer.eos_token

        if self.device_map is None:
            model = model.to(self.device)

        batches = []
        for start in tqdm(range(0, len(texts), self.batch_size), desc="Encoding"):
            batch_texts = list(texts[start : start + self.batch_size])
            inputs = tokenizer(
                batch_texts,
                **tokenizer_kwargs,
            )
            inputs = self.move_to_device(inputs, self.model_input_device(model))
            outputs = self.encode_transformer_batch(model, inputs)
            batches.append(
                self.pool_hidden_states(outputs.last_hidden_state, inputs["attention_mask"], self.pooling).detach().cpu()
            )
        return torch.cat(batches, dim=0).float()

    def resolve_output_file(self, fields: Sequence[str]) -> str:
        if self.output_file is not None:
            if os.path.isabs(self.output_file):
                return self.output_file
            return os.path.join(self.dataset_path, self.output_file)

        field_part = "_".join(field.lower() for field in fields)
        pooling = self.pooling or "auto"
        pooling_part = "" if pooling == "native" else f"_{pooling}"
        filename = f"{self.safe_filename(self.model)}_{field_part}{pooling_part}.pkl"
        return os.path.join(self.dataset_path, filename.lower())

    @staticmethod
    def resolve_initial_pooling(backend: str, pooling: Optional[str]) -> Optional[str]:
        if pooling is None:
            return "native" if backend == "sentence-transformer" else None
        if pooling not in POOLINGS:
            raise ValueError(f"pooling must be one of {POOLINGS}, but got {pooling!r}.")
        if backend == "transformers" and pooling == "native":
            raise ValueError("pooling='native' is only available for backend='sentence-transformer'.")
        return pooling

    @staticmethod
    def resolve_transformer_pooling(model_type: str, pooling: Optional[str]) -> str:
        if pooling is None:
            return "last" if model_type == "decoder-only" else "mean"
        return pooling

    @classmethod
    def infer_model_type(cls, config: Any) -> str:
        if getattr(config, "is_encoder_decoder", False):
            return "encoder-decoder"
        if getattr(config, "is_decoder", False):
            return "decoder-only"

        model_type = str(getattr(config, "model_type", "")).lower()
        if any(model_type.startswith(hint) for hint in DECODER_MODEL_HINTS):
            return "decoder-only"

        architectures = getattr(config, "architectures", None) or ()
        for architecture in architectures:
            architecture = str(architecture).lower()
            if any(hint in architecture for hint in DECODER_ARCHITECTURE_HINTS):
                return "decoder-only"

        return "encoder-only"

    def encode_transformer_batch(self, model: Any, inputs: Dict[str, torch.Tensor]) -> Any:
        if self.model_type != "encoder-decoder":
            return model(**inputs)

        if hasattr(model, "get_encoder"):
            encoder = model.get_encoder()
        else:
            encoder = model.encoder
        return encoder(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )

    @staticmethod
    def pool_hidden_states(hidden_states: torch.Tensor, attention_mask: torch.Tensor, pooling: str) -> torch.Tensor:
        attention_mask = attention_mask.to(hidden_states.device)
        if pooling == "last":
            positions = torch.arange(hidden_states.size(1), device=hidden_states.device)
            positions = positions.unsqueeze(0).expand_as(attention_mask)
            last_indices = attention_mask.long().mul(positions).argmax(dim=1)
            batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
            return hidden_states[batch_indices, last_indices]
        if pooling == "mean":
            mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
            return hidden_states.mul(mask).sum(dim=1).div(mask.sum(dim=1).clamp_min(1.0))
        raise ValueError("token hidden states support only pooling='last' or pooling='mean'.")

    @staticmethod
    def move_to_device(data: Any, device: str) -> Any:
        if torch.is_tensor(data):
            return data.to(device)
        if isinstance(data, dict):
            return {key: TextualFeatureEncoder.move_to_device(value, device) for key, value in data.items()}
        if isinstance(data, list):
            return [TextualFeatureEncoder.move_to_device(value, device) for value in data]
        if isinstance(data, tuple):
            return tuple(TextualFeatureEncoder.move_to_device(value, device) for value in data)
        return data

    @staticmethod
    def model_input_device(model: Any) -> str:
        if hasattr(model, "device"):
            return str(model.device)
        return str(next(model.parameters()).device)

    @staticmethod
    def safe_filename(value: str) -> str:
        return value.replace("/", "_").replace("\\", "_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Encode item textual features.")
    parser.add_argument("--root", type=str, default=DEFAULT_ROOT)
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET)
    parser.add_argument("--item-file", type=str, default="item.txt")
    parser.add_argument("--fields", type=str, nargs="+", default=None)
    parser.add_argument("--backend", type=str, choices=BACKENDS, default="sentence-transformer")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--model-dir", type=str, default="./models")
    parser.add_argument("--pooling", type=str, choices=POOLINGS, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--device-map", type=str, default=None)
    parser.add_argument("--output-file", type=str, default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    encoder = TextualFeatureEncoder(
        root=args.root,
        dataset=args.dataset,
        model=args.model,
        model_dir=args.model_dir,
        backend=args.backend,
        pooling=args.pooling,
        fields=args.fields,
        batch_size=args.batch_size,
        device=args.device,
        device_map=args.device_map,
        item_file=args.item_file,
        output_file=args.output_file,
        trust_remote_code=args.trust_remote_code,
    )
    features, output_file = encoder.run()
    print(f"Exported textual features with shape {tuple(features.shape)} to {output_file}.")


if __name__ == "__main__":
    main()
