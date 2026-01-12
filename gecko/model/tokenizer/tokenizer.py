import os
from dataclasses import dataclass
from logging import getLogger
from typing import List, Optional

logger = getLogger()

@dataclass
class TokenizerConf:
    # config for processing different types of data
    type: str = 'llama2'
    path: Optional[str] = None
    num_reserved_special_tokens: int = 0  # used for placeholder tokens
    data_tokenized: bool = False  # read already tokenized data

    @property
    def tokenizer_path(self) -> str:
        assert self.path is not None
        # gptneox tokenizer stored in a folder not a file
        assert os.path.isfile(self.path) or os.path.isdir(self.path), self.path
        return self.path


class Tokenizer:
    def __init__(self, tokenizer_cfg: TokenizerConf):
        # BOS / EOS token IDs
        self._bos_id: Optional[int] = None
        self._eos_id: Optional[int] = None
        self._pad_id: Optional[int] = None

    def encode(self, s: [str, List[int]], bos: bool, eos: bool) -> List[int]:
        raise NotImplemented

    def decode(self, tokens: List[int], cut_at_eos: bool = True) -> str:
        raise NotImplemented

    @property
    def bos_id(self) -> int:
        raise NotImplemented

    @property
    def eos_id(self) -> int:
        raise NotImplemented

    @property
    def pad_id(self) -> int:
        raise NotImplemented
