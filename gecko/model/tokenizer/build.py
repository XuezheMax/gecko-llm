from .tokenizer import Tokenizer, TokenizerConf
from .sentencepiece import SentencePieceTokenizer
from .llama3 import Llama3Tokenizer
from .gptneox import GPTNeoXTokenizer


def build_tokenizer(tokenizer_cfg: TokenizerConf) -> Tokenizer:
    if tokenizer_cfg.type in ['llama2', 'sentencepiece']:
        return SentencePieceTokenizer(tokenizer_cfg)
    elif tokenizer_cfg.type == 'llama3':
        return Llama3Tokenizer(tokenizer_cfg)
    elif tokenizer_cfg.type == 'gptneox':
        return GPTNeoXTokenizer(tokenizer_cfg)
    else:
        raise ValueError(f"Unknown Tokenizer type: {tokenizer_cfg.type}")
