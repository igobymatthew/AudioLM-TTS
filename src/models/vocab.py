from dataclasses import dataclass

from transformers import AutoTokenizer


class SimpleTokenizer:
    def __init__(self):
        self.vocab = {'<PAD>': 0, '<EOS>': 1, '<UNK>': 2}
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.pad_token = '<PAD>'
        self.eos_token = '<EOS>'

    def add_special_tokens(self, specials):
        for tok in specials.get('additional_special_tokens', []):
            if tok not in self.vocab:
                self.vocab[tok] = len(self.vocab)

    def encode(self, text: str, add_special_tokens: bool = True):
        ids = []
        for w in text.strip().split():
            if w not in self.vocab:
                self.vocab[w] = len(self.vocab)
            ids.append(self.vocab[w])
        if add_special_tokens:
            ids.append(self.eos_token_id)
        return ids

    def __len__(self):
        return len(self.vocab)


@dataclass
class CombinedVocab:
    text_tokenizer_name: str = 'gpt2'
    audio_vocab_size: int = 8192

    def __post_init__(self):
        try:
            self.text_tokenizer = AutoTokenizer.from_pretrained(self.text_tokenizer_name, local_files_only=True)
        except Exception:
            self.text_tokenizer = SimpleTokenizer()

        specials = {'additional_special_tokens': ['<AUDIO_START>', '<AUDIO_END>', '<SPK>']}
        self.text_tokenizer.add_special_tokens(specials)
        if getattr(self.text_tokenizer, 'pad_token', None) is None:
            self.text_tokenizer.pad_token = self.text_tokenizer.eos_token

        self.text_vocab_size = len(self.text_tokenizer)
        self.audio_offset = self.text_vocab_size
        self.total_vocab_size = self.text_vocab_size + self.audio_vocab_size

    def encode_text(self, text: str) -> list[int]:
        return self.text_tokenizer.encode(text, add_special_tokens=True)

    def encode_audio(self, audio_tokens) -> list[int]:
        return [int(t) + self.audio_offset for t in audio_tokens]

    def decode_audio(self, model_tokens) -> list[int]:
        out = []
        for t in model_tokens:
            v = int(t) - self.audio_offset
            if 0 <= v < self.audio_vocab_size:
                out.append(v)
        return out
