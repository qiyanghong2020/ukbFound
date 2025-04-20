import json
import pickle
from pathlib import Path
from collections import Counter, OrderedDict
from typing import Dict, Iterable, List, Optional, Tuple, Union
from typing_extensions import Self

import numpy as np
import pandas as pd
import torch
import torchtext.vocab as torch_vocab
from torchtext.vocab import Vocab

from .. import logger


class ValueVocab(Vocab):
    """
    Vocabulary for traits.
    """

    def __init__(
        self,
        trait_list_or_vocab: Union[List[str], Vocab],
        specials: Optional[List[str]] = None,
        special_first: bool = True,
        default_token: Optional[str] = "<pad>",
    ) -> None:
        """
        Initialize the vocabulary.
        Note: add specials only works when init from a trait list.

        Args:
            trait_list_or_vocab (List[str] or Vocab): List of trait names or a
                Vocab object.
            specials (List[str]): List of special tokens.
            special_first (bool): Whether to add special tokens to the beginning
                of the vocabulary.
            default_token (str): Default token, by default will set to "<pad>",
                if "<pad>" is in the vocabulary.
        """
        if isinstance(trait_list_or_vocab, Vocab):
            _vocab = trait_list_or_vocab
            if specials is not None:
                raise ValueError(
                    "receive non-empty specials when init from a Vocab object."
                )
        elif isinstance(trait_list_or_vocab, list):
            _vocab = self._build_vocab_from_iterator(
                trait_list_or_vocab,
                specials=specials,
                special_first=special_first,
            )
        else:
            raise ValueError(
                "trait_list_or_vocab must be a list of trait names or a Vocab object."
            )
        super().__init__(_vocab.vocab)
        if default_token is not None and default_token in self:
            self.set_default_token(default_token)

    @classmethod
    def from_file(cls, file_path: Union[Path, str]) -> Self:
        """
        Load the vocabulary from a file. The file should be either a pickle or a
        json file of token to index mapping.
        """
        if isinstance(file_path, str):
            file_path = Path(file_path)
        if file_path.suffix == ".pkl":
            with file_path.open("rb") as f:
                vocab = pickle.load(f)
                return cls(vocab)
        elif file_path.suffix == ".json":
            with file_path.open("r") as f:
                token2idx = json.load(f)
                return cls.from_dict(token2idx)
        else:
            raise ValueError(
                f"{file_path} is not a valid file type. "
                "Only .pkl and .json are supported."
            )

    @classmethod
    def from_dict(
        cls,
        token2idx: Dict[str, int],
        default_token: Optional[str] = "<pad>",
    ) -> Self:
        """
        Load the vocabulary from a dictionary.

        Args:
            token2idx (Dict[str, int]): Dictionary mapping tokens to indices.
        """
        # initiate an empty vocabulary first
        _vocab = cls([])

        # add the tokens to the vocabulary, ValueVocab requires consecutive indices
        for t, i in sorted(token2idx.items(), key=lambda x: x[1]):
            _vocab.insert_token(t, i)

        if default_token is not None and default_token in _vocab:
            _vocab.set_default_token(default_token)

        return _vocab

    def _build_vocab_from_iterator(
        self,
        iterator: Iterable,
        min_freq: int = 1,
        specials: Optional[List[str]] = None,
        special_first: bool = True,
    ) -> Vocab:
        """
        Build a Vocab from an iterator. This function is modified from
        torchtext.vocab.build_vocab_from_iterator. The original function always
        splits tokens into characters, which is not what we want.

        Args:
            iterator (Iterable): Iterator used to build Vocab. Must yield list
                or iterator of tokens.
            min_freq (int): The minimum frequency needed to include a token in
                the vocabulary.
            specials (List[str]): Special symbols to add. The order of supplied
                tokens will be preserved.
            special_first (bool): Whether to add special tokens to the beginning

        Returns:
            torchtext.vocab.Vocab: A `Vocab` object
        """

        counter = Counter()
        counter.update(iterator)

        if specials is not None:
            for tok in specials:
                del counter[tok]

        sorted_by_freq_tuples = sorted(counter.items(), key=lambda x: x[0])
        sorted_by_freq_tuples.sort(key=lambda x: x[1], reverse=True)
        ordered_dict = OrderedDict(sorted_by_freq_tuples)

        if specials is not None:
            if special_first:
                specials = specials[::-1]
            for symbol in specials:
                ordered_dict.update({symbol: min_freq})
                ordered_dict.move_to_end(symbol, last=not special_first)

        word_vocab = torch_vocab.vocab(ordered_dict, min_freq=min_freq)
        return word_vocab

    @property
    def pad_token(self) -> Optional[str]:
        """
        Get the pad token.
        """
        if getattr(self, "_pad_token", None) is None:
            self._pad_token = None
        return self._pad_token

    @pad_token.setter
    def pad_token(self, pad_token: str) -> None:
        """
        Set the pad token. Will not add the pad token to the vocabulary.

        Args:
            pad_token (str): Pad token, should be in the vocabulary.
        """
        if pad_token not in self:
            raise ValueError(f"{pad_token} is not in the vocabulary.")
        self._pad_token = pad_token

    def save_json(self, file_path: Union[Path, str]) -> None:
        """
        Save the vocabulary to a json file.
        """
        if isinstance(file_path, str):
            file_path = Path(file_path)
        with file_path.open("w") as f:
            json.dump(self.get_stoi(), f, indent=2)

    def set_default_token(self, default_token: str) -> None:
        """
        Set the default token.

        Args:
            default_token (str): Default token.
        """
        if default_token not in self:
            raise ValueError(f"{default_token} is not in the vocabulary.")
        self.set_default_index(self[default_token])


def get_default_value_vocab() -> ValueVocab:
    """
    Get the default value vocabulary, consisting of value symbols and ids.
    """
    vocab_file = Path(__file__).parent / "default_value_vocab.json"
    if not vocab_file.exists():
        logger.info(
            f"No existing default vocab, will build one and save to {vocab_file}"
        )
        return _build_default_value_vocab(save_vocab_to=vocab_file)
    logger.info(f"Loading value vocabulary from {vocab_file}")
    return ValueVocab.from_file(vocab_file)


def _build_default_value_vocab(
    download_source_to: str = "/tmp",
    save_vocab_to: Union[Path, str, None] = None,
) -> ValueVocab:
    """
    Build the default value vocabulary from HGNC value symbols.

    Args:
        download_source_to (str): Directory to download the source data.
        save_vocab_to (Path or str): Path to save the vocabulary. If None,
            the vocabulary will not be saved. Default to None.
    """
    value_collection_file = (
        Path(download_source_to) / "human.value_name_symbol.from_valuenames.org.tsv"
    )

    logger.info(f"Building value vocabulary from {value_collection_file}")
    df = pd.read_csv(value_collection_file, sep="\t")
    value_list = df["Approved symbol"].dropna().unique().tolist()
    value_vocab = ValueVocab(value_list)  # no special tokens set in default vocab
    if save_vocab_to is not None:
        value_vocab.save_json(Path(save_vocab_to))
    return value_vocab


def tokenize_batch(
    data: np.ndarray,
    value_ids: np.ndarray,
    return_pt: bool = True,
    append_cls: bool = True,
    include_zero_value: bool = False,
    cls_id: int = "<cls>",
    mod_type: np.ndarray = None,
    cls_id_mod_type: int = None,
) -> List[Tuple[Union[torch.Tensor, np.ndarray]]]:
    """
    Tokenize a batch of data. Returns a list of tuple (trait_id, count).

    Args:
        data (array-like): A batch of data, with shape (batch_size, n_features).
            n_features equals the number of all traits.
        value_ids (array-like): A batch of trait ids, with shape (n_features,).
        return_pt (bool): Whether to return torch tensors of value_ids and counts,
            default to True.

    Returns:
        list: A list of tuple (trait_id, count) of non zero trait expressions.
    """
    if data.shape[0] != value_ids.shape[0]:
        raise ValueError(
            f"Number of features in data ({data.shape[1]}) does not match "
            f"number of value_ids ({len(value_ids)})."
        )
    if mod_type is not None and data.shape[1] != len(mod_type):
        raise ValueError(
            f"Number of features in data ({data.shape[1]}) does not match "
            f"number of mod_type ({len(mod_type)})."
        )

    tokenized_data = []
    for i in range(len(data)):
        row = data[i]
        trait_id = value_ids[i]
        mod_types = None
        if include_zero_value:
            values = row
            traits = trait_id
            if mod_type is not None:
                mod_types = mod_type

        if append_cls:
            traits = np.insert(traits, 0, cls_id)
            values = np.insert(values, 0, 0)
            if mod_type is not None:
                mod_types = np.insert(mod_types, 0, cls_id_mod_type)
        if return_pt:
            if traits.dtype.type is np.str_:
                traits = traits.tolist()
            elif traits.dtype.type is np.int_:
                traits = torch.from_numpy(traits).long()
            else:
                raise ValueError("Unsupported dtype in traits array")
            values = torch.from_numpy(values).float()
            if mod_type is not None:
                mod_types = torch.from_numpy(mod_types).long()
        tokenized_data.append((traits, values, mod_types))
    return tokenized_data


def pad_batch(
    batch: List[Tuple],
    max_len: int,
    vocab: Vocab,
    pad_token: str = "<pad>",
    pad_value: int = 0,
    cls_appended: bool = True,
    vocab_mod: Vocab = None,
) -> Dict[str, torch.Tensor]:
    """
    Pad a batch of data. Returns a list of Dict[trait_id, count].

    Args:
        batch (list): A list of tuple (trait_id, count).
        max_len (int): The maximum length of the batch.
        vocab (Vocab): The vocabulary containing the pad token.
        pad_token (str): The token to pad with.

    Returns:
        Dict[str, torch.Tensor]: A dictionary of trait_id and count.
    """
    max_ori_len = max(len(batch[i][0]) for i in range(len(batch)))
    max_len = min(max_ori_len, max_len)

    pad_id = vocab[pad_token]
    if vocab_mod is not None:
        mod_pad_id = vocab_mod[pad_token]
    trait_ids_list = []
    values_list = []
    mod_types_list = []

    for i in range(len(batch)):
        trait_ids, values, mod_types = batch[i]

        if len(trait_ids) > max_len:
            # sample max_len traits
            if not cls_appended:
                idx = np.random.choice(len(trait_ids), max_len, replace=False)
            else:
                idx = np.random.choice(len(trait_ids) - 1, max_len - 1, replace=False)
                idx = idx + 1
                idx = np.insert(idx, 0, 0)
            trait_ids = trait_ids[idx]
            values = values[idx]
            if mod_types is not None:
                mod_types = mod_types[idx]
        if len(trait_ids) < max_len:
            trait_ids = torch.cat(
                [
                    trait_ids,
                    torch.full(
                        (max_len - len(trait_ids),), pad_id, dtype=trait_ids.dtype
                    ),
                ]
            )
            values = torch.cat(
                [
                    values,
                    torch.full((max_len - len(values),), pad_value, dtype=values.dtype),
                ]
            )
            if mod_types is not None:
                mod_types = torch.cat(
                    [
                        mod_types,
                        torch.full(
                            (max_len - len(mod_types),),
                            mod_pad_id,
                            dtype=mod_types.dtype,
                        ),
                    ]
                )

        trait_ids_list.append(trait_ids)
        values_list.append(values)
        if mod_types is not None:
            mod_types_list.append(mod_types)

    batch_padded = {
        "traits": torch.stack(trait_ids_list, dim=0),
        "values": torch.stack(values_list, dim=0),
    }
    if mod_types is not None:
        batch_padded["mod_types"] = torch.stack(mod_types_list, dim=0)
    return batch_padded


def tokenize_and_pad_batch(
    data: np.ndarray,
    value_ids: np.ndarray,
    max_len: int,
    vocab: Vocab,
    pad_token: str,
    pad_value: int,
    append_cls: bool = True,
    include_zero_value: bool = False,
    cls_token: str = "<cls>",
    return_pt: bool = True,
    mod_type: np.ndarray = None,
    vocab_mod: Vocab = None,
) -> Dict[str, torch.Tensor]:
    """
    Tokenize and pad a batch of data. Returns a list of tuple (trait_id, count).
    """
    cls_id = vocab[cls_token]
    if mod_type is not None:
        cls_id_mod_type = vocab_mod[cls_token]
    tokenized_data = tokenize_batch(
        data,
        value_ids,
        return_pt=return_pt,
        append_cls=append_cls,
        include_zero_value=include_zero_value,
        cls_id=cls_id,
        mod_type=mod_type,
        cls_id_mod_type=cls_id_mod_type if mod_type is not None else None,
    )

    batch_padded = pad_batch(
        tokenized_data,
        max_len,
        vocab,
        pad_token,
        pad_value,
        cls_appended=append_cls,
        vocab_mod=vocab_mod,
    )
    return batch_padded


def random_mask_value(
    values: Union[torch.Tensor, np.ndarray],
    mask_ratio: float = 0.15,
    mask_value: int = -1,
    pad_value: int = 0,
) -> torch.Tensor:
    """
    Randomly mask a batch of data.

    Args:
        values (array-like):
            A batch of tokenized data, with shape (batch_size, n_features).
        mask_ratio (float): The ratio of traits to mask, default to 0.15.
        mask_value (int): The value to mask with, default to -1.
        pad_value (int): The value of padding in the values, will be kept unchanged.

    Returns:
        torch.Tensor: A tensor of masked data.
    """
    if isinstance(values, torch.Tensor):
        # it is crutial to clone the tensor, otherwise it changes the original tensor
        values = values.clone().detach().numpy()
    else:
        values = values.copy()

    for i in range(len(values)):
        row = values[i]
        non_padding_idx = np.nonzero(row - pad_value)[0]
        n_mask = int(len(non_padding_idx) * mask_ratio)
        mask_idx = np.random.choice(non_padding_idx, n_mask, replace=False)
        row[mask_idx] = mask_value
        row[(1302 <= row) & (row <= 2353)  & (row%2==0)] = mask_value  # hqy 20240820 for 20002, force include diseases
    return torch.from_numpy(values).float()
