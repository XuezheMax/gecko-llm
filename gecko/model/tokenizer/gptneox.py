from logging import getLogger
from typing import List, Optional
import os

from .tokenizer import Tokenizer, TokenizerConf

logger = getLogger()


class GPTNeoXTokenizer(Tokenizer):
    """
    Tokenizing and encoding/decoding text using the AutoTokenizer for GPT-NeoX style models.
    """

    def __init__(self, tokenizer_cfg: TokenizerConf):
        super().__init__(tokenizer_cfg)
        try:
            from transformers import AutoTokenizer
        except ImportError:
            raise EnvironmentError(
                f"The transformers library must be installed to use GPTNeo tokenizer"
            )

        # reload tokenizer
        model_path = tokenizer_cfg.tokenizer_path
        assert os.path.isfile(model_path) or os.path.isdir(model_path), model_path

        # Load the tokenizer using AutoTokenizer
        self.model = AutoTokenizer.from_pretrained(model_path)
        logger.info(f"Reloaded AutoTokenizer from {model_path}")

        self.already_tokenized_data = tokenizer_cfg.data_tokenized

        # BOS / EOS token IDs
        self._bos_id: Optional[int] = None
        self._eos_id: Optional[int] = None
        self._pad_id: Optional[int] = None

        # Set up special tokens based on the config
        # EOS token: <|endoftext|>
        # PAD token: <|padding|>
        # No BOS token specified in the config (bos_token: null)
        
        # Get token IDs for special tokens using the tokenizer's built-in properties
        # This handles cases where tokens might be in special_mappings.json
        # BOS token not defined.
        self._eos_id = self.model.eos_token_id
        self._pad_id = self.model.pad_token_id

        if self.model.bos_token_id is None:
            self.model.add_special_tokens({'bos_token': '<|beginoftext|>'})

        self._bos_id = self.model.bos_token_id
        self.n_words: int = len(self.model)

        logger.info(
            f"#words: {self.n_words} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id} - PAD ID: {self.pad_id}"
        )

    def encode(self, s: [str, List[int]], bos: bool, eos: bool) -> List[int]:
        if self.already_tokenized_data:
            t = s
        else:
            assert type(s) is str
            # Use the tokenizer to encode the text
            # Note: The config shows add_bos_token: false and add_eos_token: false
            # So we handle BOS/EOS manually in our encode method
            encoded = self.model.encode(s, add_special_tokens=False)
            t = encoded

        if bos:
            t.insert(0, self.bos_id)
        if eos:
            t.append(self.eos_id)

        return t

    def decode(self, tokens: List[int], cut_at_eos: bool = True) -> str:
        if cut_at_eos and self.eos_id is not None:
            for k, t in enumerate(tokens):
                if t == self.eos_id:
                    tokens = tokens[:k + 1]
                    break

        return self.model.decode(tokens)

    @property
    def bos_id(self) -> int:
        return self._bos_id # should return same id as bos_id

    @property
    def eos_id(self) -> int:
        return self._eos_id

    @property
    def pad_id(self) -> int:
        return self._pad_id
