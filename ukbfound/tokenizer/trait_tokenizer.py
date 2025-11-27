import json
import pickle
from pathlib import Path
from collections import Counter, OrderedDict
from typing import Dict, Iterable, List, Optional, Tuple, Union
from typing_extensions import Self

import numpy as np
import pandas as pd
import torch

from .. import logger


class ValueVocab:
    """
    轻量级 Vocabulary，用纯 Python 实现，去掉对 torchtext 的依赖。

    功能目标：
    - 支持从列表或字典构建 token -> index 映射
    - 支持从文件加载（.pkl / .json）
    - 支持 __getitem__、__contains__、__len__
    - 支持 set_default_index / get_default_index
    - 支持 set_default_token / pad_token 属性
    - 支持 get_stoi() / get_itos() 以兼容原有接口
    """

    def __init__(
        self,
        trait_list_or_vocab: Optional[Iterable[str]] = None,
        specials: Optional[List[str]] = None,
        special_first: bool = True,
        default_token: Optional[str] = "<pad>",
    ) -> None:
        """
        Initialize the vocabulary.

        Args:
            trait_list_or_vocab: 可迭代的 token 列表；如果为 None，则创建空 vocab。
            specials: 特殊 token 列表（如 ["<pad>", "<mask>", "<cls>"]）。
            special_first: 特殊 token 是否放在前面。
            default_token: 默认 token 名称，用于缺失时回退。
        """
        if trait_list_or_vocab is None:
            tokens = []
        else:
            tokens = list(trait_list_or_vocab)

        # 根据频次和 specials 生成有序 token 列表
        tokens = self._build_token_list_from_iterator(
            tokens,
            specials=specials,
            special_first=special_first,
        )

        self.stoi: "OrderedDict[str, int]" = OrderedDict()
        self.itos: List[str] = []
        self._default_index: Optional[int] = None
        self._pad_token: Optional[str] = None

        for tok in tokens:
            if tok not in self.stoi:
                self.stoi[tok] = len(self.itos)
                self.itos.append(tok)

        if default_token is not None and default_token in self:
            self.set_default_token(default_token)

    # ------------------------------------------------------------------
    # 构建 vocab 的内部工具函数（不依赖 torchtext）
    # ------------------------------------------------------------------
    def _build_token_list_from_iterator(
        self,
        iterator: Iterable[str],
        min_freq: int = 1,
        specials: Optional[List[str]] = None,
        special_first: bool = True,
    ) -> List[str]:
        """
        根据频次和 specials 构建有序的 token 列表。
        逻辑与原来的 _build_vocab_from_iterator 类似，只是改为返回 List[str]。
        """
        counter = Counter()
        counter.update(iterator)

        # 先从 counter 中移除 specials，避免重复计数
        if specials is not None:
            for tok in specials:
                if tok in counter:
                    del counter[tok]

        # 先按 token 字典序，再按频次排序（频次从高到低）
        sorted_by_freq_tuples = sorted(counter.items(), key=lambda x: x[0])
        sorted_by_freq_tuples.sort(key=lambda x: x[1], reverse=True)

        ordered_dict = OrderedDict(
            (tok, freq) for tok, freq in sorted_by_freq_tuples if freq >= min_freq
        )

        # 插入 specials
        if specials is not None:
            if special_first:
                specials_iter = specials[::-1]
            else:
                specials_iter = specials
            for symbol in specials_iter:
                # 频次设为 min_freq，仅用于占位
                ordered_dict.update({symbol: min_freq})
                ordered_dict.move_to_end(symbol, last=not special_first)

        # 返回最终的 token 顺序
        return list(ordered_dict.keys())

    # ------------------------------------------------------------------
    # 基本容器行为
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.itos)

    def __contains__(self, token: str) -> bool:
        return token in self.stoi

    def __getitem__(self, token: str) -> int:
        """
        vocab[token] -> index

        若 token 不存在且设置了 default_index，则返回 default_index；
        否则抛出 KeyError。
        """
        if token in self.stoi:
            return self.stoi[token]
        if self._default_index is not None:
            return self._default_index
        raise KeyError(f"Token {token!r} is not in the vocabulary.")

    # ------------------------------------------------------------------
    # 与原 torchtext.Vocab 接口兼容的辅助方法
    # ------------------------------------------------------------------
    def insert_token(self, token: str, index: int) -> None:
        """
        插入一个 token 到指定 index。若 token 已存在，则忽略。
        用于 from_dict 等场景，尽量模拟 torchtext 的行为。
        """
        if token in self.stoi:
            return
        index = int(index)
        if index < 0:
            index = 0
        if index > len(self.itos):
            index = len(self.itos)

        self.itos.insert(index, token)
        # 重新构建 stoi
        self.stoi = OrderedDict((tok, i) for i, tok in enumerate(self.itos))

    def get_stoi(self) -> Dict[str, int]:
        return dict(self.stoi)

    def get_itos(self) -> List[str]:
        return list(self.itos)

    def set_default_index(self, index: Optional[int]) -> None:
        self._default_index = index

    def get_default_index(self) -> Optional[int]:
        return self._default_index

    # ------------------------------------------------------------------
    # pad token / 默认 token 相关
    # ------------------------------------------------------------------
    @property
    def pad_token(self) -> Optional[str]:
        """
        获取 pad token。
        """
        if getattr(self, "_pad_token", None) is None:
            self._pad_token = None
        return self._pad_token

    @pad_token.setter
    def pad_token(self, pad_token: str) -> None:
        """
        设置 pad token（不会自动添加到 vocab 中）。
        """
        if pad_token not in self:
            raise ValueError(f"{pad_token} is not in the vocabulary.")
        self._pad_token = pad_token

    def set_default_token(self, default_token: str) -> None:
        """
        设置默认 token，即 default_index 对应的 token。
        """
        if default_token not in self:
            raise ValueError(f"{default_token} is not in the vocabulary.")
        self.set_default_index(self[default_token])

    # ------------------------------------------------------------------
    # 文件读写 / 构造方法
    # ------------------------------------------------------------------
    @classmethod
    def from_file(cls, file_path: Union[Path, str]) -> Self:
        """
        从文件加载词表。支持：
        - .pkl: 直接存储的 ValueVocab 或兼容结构
        - .json: token -> index 映射
        """
        if isinstance(file_path, str):
            file_path = Path(file_path)
        if file_path.suffix == ".pkl":
            with file_path.open("rb") as f:
                vocab = pickle.load(f)
                # 如果已经是 ValueVocab，就直接返回
                if isinstance(vocab, cls):
                    return vocab
                # 否则认为是 token2idx 字典
                return cls.from_dict(vocab)
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
        根据 token -> index 字典构建 ValueVocab。
        ValueVocab 要求 index 连续，我们按 index 排序后重建。
        """
        # 按 index 排序后的 token 列表
        sorted_tokens = [t for t, i in sorted(token2idx.items(), key=lambda x: x[1])]
        vocab = cls(
            sorted_tokens,
            specials=None,
            special_first=True,
            default_token=default_token,
        )
        return vocab

    def save_json(self, file_path: Union[Path, str]) -> None:
        """
        将词表保存为 JSON（token -> index 映射）。
        """
        if isinstance(file_path, str):
            file_path = Path(file_path)
        with file_path.open("w") as f:
            json.dump(self.get_stoi(), f, indent=2)


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
    # 默认词表不设置 specials
    value_vocab = ValueVocab(value_list)
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
    vocab: ValueVocab,
    pad_token: str = "<pad>",
    pad_value: int = 0,
    cls_appended: bool = True,
    vocab_mod: Optional[ValueVocab] = None,
) -> Dict[str, torch.Tensor]:
    """
    Pad a batch of data. Returns a list of Dict[trait_id, count].

    Args:
        batch (list): A list of tuple (trait_id, count).
        max_len (int): The maximum length of the batch.
        vocab (ValueVocab): The vocabulary containing the pad token.
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
    vocab: ValueVocab,
    pad_token: str,
    pad_value: int,
    append_cls: bool = True,
    include_zero_value: bool = False,
    cls_token: str = "<cls>",
    return_pt: bool = True,
    mod_type: np.ndarray = None,
    vocab_mod: Optional[ValueVocab] = None,
) -> Dict[str, torch.Tensor]:
    """
    Tokenize and pad a batch of data. Returns a list of tuple (trait_id, count).
    """
    cls_id = vocab[cls_token]
    if mod_type is not None:
        if vocab_mod is None:
            raise ValueError("vocab_mod must be provided when mod_type is not None.")
        cls_id_mod_type = vocab_mod[cls_token]
    else:
        cls_id_mod_type = None

    tokenized_data = tokenize_batch(
        data,
        value_ids,
        return_pt=return_pt,
        append_cls=append_cls,
        include_zero_value=include_zero_value,
        cls_id=cls_id,
        mod_type=mod_type,
        cls_id_mod_type=cls_id_mod_type,
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
        # it is crucial to clone the tensor, otherwise it changes the original tensor
        values = values.clone().detach().numpy()
    else:
        values = values.copy()

    for i in range(len(values)):
        row = values[i]
        non_padding_idx = np.nonzero(row - pad_value)[0]
        n_mask = int(len(non_padding_idx) * mask_ratio)
        if n_mask > 0:
            mask_idx = np.random.choice(non_padding_idx, n_mask, replace=False)
            row[mask_idx] = mask_value
        # hqy 20240820 for 20002, force include diseases
        row[(1302 <= row) & (row <= 2353) & (row % 2 == 0)] = mask_value
    return torch.from_numpy(values).float()
