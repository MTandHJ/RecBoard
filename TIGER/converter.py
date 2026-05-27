import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

import torch
from freerec.utils import infoLogger
from transformers.generation.stopping_criteria import StoppingCriteria
from transformers.tokenization_utils_base import AddedToken, PreTrainedTokenizerBase

_Trie = Dict[int, "_Trie"]


class ItemCountStoppingCriteria(StoppingCriteria):
    """Stop after a fixed number of completed SID items are generated.

    Parameters
    ----------
    sid_end_id : int
        Token ID for the ``</SID>`` item boundary.
    num_items : int, default=1
        Number of newly generated item boundaries required to stop.
    generation_start : int, default=0
        Index where newly generated decoder tokens begin. Tokens before this
        position are ignored so an existing SID prefix is not counted.

    Examples
    --------
    >>> criterion = ItemCountStoppingCriteria(9, num_items=2, generation_start=1)
    >>> criterion(torch.tensor([[9, 3, 9, 4]]), None).tolist()
    [False]
    >>> criterion(torch.tensor([[9, 3, 9, 9]]), None).tolist()
    [True]
    """

    def __init__(self, sid_end_id: int, num_items: int = 1, generation_start: int = 0):
        if num_items <= 0:
            raise ValueError("num_items must be positive")
        if generation_start < 0:
            raise ValueError("generation_start must not be negative")
        self.sid_end_id = sid_end_id
        self.num_items = num_items
        self.generation_start = generation_start

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
        **kwargs,
    ) -> torch.BoolTensor:
        """Evaluate completion for each sequence in a generation batch.

        Parameters
        ----------
        input_ids : torch.LongTensor
            Full decoder-side sequences, including any initial decoder prefix.
        scores : torch.FloatTensor
            Current generation scores, accepted for the Transformers stopping
            criteria interface and not used by this criterion.
        **kwargs
            Additional generation callback values.

        Returns
        -------
        torch.BoolTensor
            Boolean stop decision for every sequence in ``input_ids``.
        """
        generated_ids = input_ids[:, self.generation_start :]
        return generated_ids.eq(self.sid_end_id).sum(dim=1).ge(self.num_items)


class SemIDConverter:
    """Convert item keys to semantic-ID protocol text and back.

    Parameters
    ----------
    sid_vocab : Dict[str, Tuple[str, ...]]
        Raw semantic IDs keyed by canonical item keys.
    tokenizer : PreTrainedTokenizerBase
        A Hugging Face style tokenizer supporting ``AddedToken`` registration,
        ``add_tokens()`` and ``convert_tokens_to_ids()``.

    Examples
    --------
    >>> class _Tokenizer:
    ...     def __init__(self):
    ...         self.vocab = {}
    ...     def add_tokens(self, tokens):
    ...         for token in tokens:
    ...             self.vocab.setdefault(str(token), len(self.vocab))
    ...         return len(tokens)
    ...     def convert_tokens_to_ids(self, token):
    ...         return self.vocab[token]
    >>> converter = SemIDConverter({"item_3": ("<sid_0_2>",)}, _Tokenizer())
    [SemIDConverter] >>> Registering 3 tokens to _Tokenizer ...
    >>> converter.encode("item_3")
    '<SID> <sid_0_2> </SID>'
    >>> converter.decode("result: <SID><sid_0_2></SID>")
    [3]
    >>> start = converter.tokenizer.convert_tokens_to_ids("<SID>")
    >>> converter.allowed_tokens([]) == (start,)
    True
    """

    ITEM_FORMAT = "{prefix}_{id}"
    SID_FORMAT = "<sid_{level}_{id}>"
    CHECK_SID_FORMAT = "<sid_c_{id}>"

    ITEM_PATTERN = re.compile(r"^(?P<prefix>[A-Za-z0-9_-]+)_(?P<id>0|[1-9][0-9]*)$")
    SID_PATTERN = re.compile(r"^<sid_[0-9]+_[0-9]+>$")
    SID_CONTENT_PATTERN = re.compile(r"<sid_[0-9]+_[0-9]+>|<sid_c_[0-9]+>")
    SID_BLOCK_PATTERN = re.compile(r"<SID>(?P<body>.*?)</SID>", re.DOTALL)

    SID_START_TOKEN = "<SID>"
    SID_END_TOKEN = "</SID>"

    def __init__(
        self,
        sid_vocab: Dict[str, Tuple[str, ...]],
        tokenizer: PreTrainedTokenizerBase,
    ):
        self.tokenizer = tokenizer

        sid_vocab = self._check_sid_vocab(sid_vocab)
        if self._check_conflicts(sid_vocab):
            sid_vocab = self.resolve_conflicts(sid_vocab)
        self._item_to_sids = sid_vocab
        self._sids_to_item = {sids: item for item, sids in sid_vocab.items()}

        self._register_protocol_tokens(self._item_to_sids)
        self._sid_start_id = self.tokenizer.convert_tokens_to_ids(self.SID_START_TOKEN)
        self._sid_end_id = self.tokenizer.convert_tokens_to_ids(self.SID_END_TOKEN)
        self._tries_by_prefix = self._build_tries()

    @classmethod
    def format(cls, item_id: int, prefix: str = "item") -> str:
        """Format an item identifier using the converter key convention.

        Parameters
        ----------
        item_id : int
            Integer item identifier.
        prefix : str, default="item"
            Namespace for the item identifier.

        Returns
        -------
        str
            Formatted item key.

        Examples
        --------
        >>> SemIDConverter.format(12, prefix="book")
        'book_12'
        """
        item = cls.ITEM_FORMAT.format(prefix=prefix, id=item_id)
        return item

    @classmethod
    def parse(cls, item: str) -> Tuple[str, int]:
        """Parse a canonical item key into its prefix and numeric identifier.

        Parameters
        ----------
        item : str
            Item key in ``{prefix}_{id}`` format.

        Returns
        -------
        Tuple[str, int]
            Namespace prefix and integer item identifier.

        Raises
        ------
        ValueError
            If ``item`` does not follow the canonical key format.

        Examples
        --------
        >>> SemIDConverter.parse("book_12")
        ('book', 12)
        >>> SemIDConverter.parse("book_001")
        Traceback (most recent call last):
        ...
        ValueError: invalid item key: 'book_001'
        """
        match = cls.ITEM_PATTERN.fullmatch(item)
        if match is None:
            raise ValueError(f"invalid item key: {item!r}")
        return match.group("prefix"), int(match.group("id"))

    @property
    def sid_vocab(self) -> Dict[str, Tuple[str, ...]]:
        """Return the SID vocabulary used by encoding and decoding."""
        return self._item_to_sids

    def encode(self, items: Union[str, List[str]]) -> str:
        """Encode one or more item keys as wrapped SID protocol text.

        Parameters
        ----------
        items : Union[str, List[str]]
            An item key or an ordered list of item keys.

        Returns
        -------
        str
            Space-separated ``<SID> ... </SID>`` blocks.

        Raises
        ------
        ValueError
            If any requested item has not been registered.

        Examples
        --------
        >>> class _Tokenizer(dict):
        ...     def add_tokens(self, tokens):
        ...         for token in tokens:
        ...             self.setdefault(str(token), len(self))
        ...     def convert_tokens_to_ids(self, token):
        ...         return self[token]
        >>> converter = SemIDConverter({"item_2": ("<sid_0_1>", "<sid_1_3>"), "item_8": ("<sid_0_4>",)}, _Tokenizer())
        [SemIDConverter >>>] Registering 5 tokens to _Tokenizer ...
        >>> converter.encode("item_2")
        '<SID> <sid_0_1> <sid_1_3> </SID>'
        >>> converter.encode(["item_2", "item_8"])
        '<SID> <sid_0_1> <sid_1_3> </SID> <SID> <sid_0_4> </SID>'
        >>> converter.encode("item_9")
        Traceback (most recent call last):
        ...
        ValueError: item 'item_9' is not registered
        """
        items = [items] if isinstance(items, str) else items

        encoded: List[str] = []
        for item in items:
            if item not in self._item_to_sids:
                raise ValueError(f"item '{item}' is not registered")
            encoded.append(
                " ".join(
                    (
                        self.SID_START_TOKEN,
                        *self._item_to_sids[item],
                        self.SID_END_TOKEN,
                    )
                )
            )
        return " ".join(encoded)

    def batch_encode(self, batch_items: List[List[str]]) -> List[str]:
        """Encode multiple ordered item sequences.

        Parameters
        ----------
        batch_items : List[List[str]]
            Batch of item-key sequences.

        Returns
        -------
        List[str]
            Encoded SID protocol text for each sequence.
        """
        return [self.encode(items) for items in batch_items]

    def decode(self, text: str, *, prefix: str = "item") -> List[int]:
        """Extract registered item identifiers from SID protocol blocks.

        Parameters
        ----------
        text : str
            Text containing zero or more complete ``<SID> ... </SID>`` blocks.
        prefix : str, default="item"
            Namespace of item identifiers to return.

        Returns
        -------
        List[int]
            Matching item identifiers in textual order. Invalid or unknown
            blocks are ignored.

        Examples
        --------
        >>> class _Tokenizer(dict):
        ...     def add_tokens(self, tokens):
        ...         for token in tokens:
        ...             self.setdefault(str(token), len(self))
        ...     def convert_tokens_to_ids(self, token):
        ...         return self[token]
        >>> converter = SemIDConverter({"item_2": ("<sid_0_1>", "<sid_1_3>"), "book_8": ("<sid_0_4>",)}, _Tokenizer())
        [SemIDConverter] >>> Registering 5 tokens to _Tokenizer ...
        >>> converter.decode("chosen: <SID> <sid_0_1> <sid_1_3> </SID>; <SID><sid_0_1><sid_1_3></SID>")
        [2, 2]
        >>> converter.decode("<SID><sid_0_4></SID>", prefix="book")
        [8]
        >>> converter.decode("<SID><sid_9_9></SID> <SID><sid_0_1>")
        []
        """
        decoded: List[int] = []
        for match in self.SID_BLOCK_PATTERN.finditer(text):
            key = self._match_wrapped_sid(match.group("body"))
            if key is None:
                continue

            item_prefix, item_id = self.parse(key)
            if item_prefix == prefix:
                decoded.append(item_id)
        return decoded

    def batch_decode(
        self, texts: List[str], *, prefix: str = "item"
    ) -> List[List[int]]:
        """Extract item identifiers from a batch of protocol strings.

        Parameters
        ----------
        texts : List[str]
            Texts to decode.
        prefix : str, default="item"
            Namespace of item identifiers to return.

        Returns
        -------
        List[List[int]]
            Decoded item identifier sequences.
        """
        return [self.decode(text, prefix=prefix) for text in texts]

    def allowed_tokens(
        self,
        generated_ids: Union[List[int], Tuple[int, ...]],
        *,
        prefix: str = "item",
    ) -> Tuple[int, ...]:
        """Return next-token IDs that preserve validity of an SID item.

        Parameters
        ----------
        generated_ids : Union[List[int], Tuple[int, ...]]
            SID protocol token IDs generated after the decoder prefix.
        prefix : str, default="item"
            Namespace whose item trie constrains generation.

        Returns
        -------
        Tuple[int, ...]
            Token IDs permitted for the next generation step.

        Notes
        -----
        Completed items are followed by a new ``<SID>`` token. A separate
        stopping criterion decides when the required number of items has been
        completed.
        """
        root = self._tries_by_prefix[prefix]

        if not generated_ids or generated_ids[-1] == self._sid_end_id:
            return (self._sid_start_id,)

        suffix: List[int] = []
        for token_id in reversed(generated_ids):
            if token_id == self._sid_start_id:
                break
            suffix.append(token_id)
        else:
            raise ValueError("expected <SID> token")

        node = root
        for token_id in reversed(suffix):
            node = node[token_id]
        return tuple(node)

    def stopping_criteria(
        self, num_items: int = 1, *, generation_start: int = 0
    ) -> StoppingCriteria:
        """Build a fixed-item-count generation stopping criterion.

        Parameters
        ----------
        num_items : int, default=1
            Number of newly generated SID blocks to produce.
        generation_start : int, default=0
            Decoder sequence index at which newly generated tokens start.

        Returns
        -------
        StoppingCriteria
            Criterion suitable for a Transformers ``generate`` call.
        """
        return ItemCountStoppingCriteria(self._sid_end_id, num_items, generation_start)

    @classmethod
    def _check_sid_vocab(
        cls, sid_vocab: Dict[str, Tuple[str, ...]]
    ) -> Dict[str, Tuple[str, ...]]:
        for item, sids in sid_vocab.items():
            if cls.ITEM_PATTERN.fullmatch(item) is None:
                raise ValueError(f"invalid item for {item}")
            for sid in sids:
                if cls.SID_PATTERN.fullmatch(sid) is None:
                    raise ValueError(f"invalid SID token for {item}: {sid}")
        return {item: tuple(sids) for item, sids in sid_vocab.items()}

    @classmethod
    def _check_conflicts(
        cls,
        sid_vocab: Dict[str, Tuple[str, ...]],
    ) -> bool:
        seen = set()
        for _, sids in sid_vocab.items():
            if sids in seen:
                return True
            seen.add(sids)
        return False

    @classmethod
    def resolve_conflicts(
        cls,
        sid_vocab: Dict[str, Tuple[str, ...]],
    ) -> Dict[str, Tuple[str]]:
        """Append deterministic check SID tokens to resolve duplicate paths.

        Parameters
        ----------
        sid_vocab : Dict[str, Tuple[str, ...]]
            SID paths grouped by canonical item key.

        Returns
        -------
        Dict[str, Tuple[str, ...]]
            Deterministically disambiguated SID paths.

        Examples
        --------
        >>> SemIDConverter.resolve_conflicts(
        ...     {"item_0": ("<sid_0_2>",), "item_1": ("<sid_0_2>",)}
        ... )
        [SemIDConverter] >>> Additional 2 tokens for resolving conflicts ...
        {'item_0': ('<sid_0_2>', '<sid_c_0>'), 'item_1': ('<sid_0_2>', '<sid_c_1>')}
        """
        groups = defaultdict(list)
        for item, sids in sid_vocab.items():
            groups[sids].append(item)

        resolved: Dict[str, Tuple[str]] = {}
        max_check_tokens = 0
        for sids, items in groups.items():
            for check_id, item in enumerate(sorted(items)):
                resolved[item] = sids + (cls.CHECK_SID_FORMAT.format(id=check_id),)
            max_check_tokens = max(max_check_tokens, len(items))
        infoLogger(
            f"[{cls.__name__}] >>> Additional {max_check_tokens} tokens for resolving conflicts ..."
        )
        return {item: resolved[item] for item in sorted(resolved)}

    def _register_protocol_tokens(self, sid_vocab: Dict[str, Tuple[str, ...]]) -> None:
        unique_sid_tokens = {token for sid in sid_vocab.values() for token in sid}
        protocol_tokens = [
            self.SID_START_TOKEN,
            self.SID_END_TOKEN,
            *sorted(unique_sid_tokens),
        ]
        added_tokens = [
            AddedToken(
                token,
                lstrip=True,
                rstrip=False,
                single_word=False,
                normalized=False,
                special=False,
            )
            for token in protocol_tokens
        ]
        self.tokenizer.add_tokens(added_tokens)
        infoLogger(
            f"[{type(self).__name__}] >>> Registering {len(added_tokens)} tokens "
            f"to {type(self.tokenizer).__qualname__} ..."
        )

    def _build_tries(self) -> Dict[str, _Trie]:
        tries: Dict[str, _Trie] = {}
        for item, sids in self._item_to_sids.items():
            prefix, _ = self.parse(item)
            root = tries.setdefault(prefix, {})
            node = root
            for token in (*sids, self.SID_END_TOKEN):
                token_id = self.tokenizer.convert_tokens_to_ids(token)
                node = node.setdefault(token_id, {})
        return tries

    def _match_wrapped_sid(self, body: str) -> Optional[str]:
        tokens = tuple(self.SID_CONTENT_PATTERN.findall(body))
        if self.SID_CONTENT_PATTERN.sub("", body).strip():
            return None
        return self._sids_to_item.get(tokens)


def prefix_allowed_tokens_fn(converter: SemIDConverter, prefix: str = "item"):
    """Build a Transformers prefix-token callback for an SID converter.

    Parameters
    ----------
    converter : SemIDConverter
        Converter that evaluates allowed SID token IDs.
    prefix : str, default="item"
        Namespace whose item trie constrains generation.

    Returns
    -------
    Callable
        Callback compatible with ``prefix_allowed_tokens_fn`` in
        Transformers generation.

    Examples
    --------
    >>> class _Constraint:
    ...     def allowed_tokens(self, ids, *, prefix="item"):
    ...         return (7,) if ids == [3] and prefix == "item" else ()
    >>> callback = prefix_allowed_tokens_fn(_Constraint())
    >>> callback(0, torch.tensor([3]))
    [7]
    """

    def prefix_allowed_tokens(batch_id: int, sentence: torch.Tensor):
        return list(converter.allowed_tokens(sentence.tolist(), prefix=prefix))

    return prefix_allowed_tokens
