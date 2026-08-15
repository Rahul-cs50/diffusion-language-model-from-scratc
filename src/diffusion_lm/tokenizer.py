import os

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.normalizers import NFKC
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast

SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[BOS]", "[EOS]", "[MASK]", "<|user|>", "<|assistant|>", "<|system|>", "<|end|>"]


def format_story(story_text: str) -> str:
    story_text = story_text.strip()
    return "<|user|>\nWrite a short story.\n<|assistant|>\n" + story_text + "\n<|end|>\n"


def train_tokenizer(dataset, n_examples: int, vocab_size: int, output_dir: str):
    def iterator():
        for i in range(min(n_examples, len(dataset))):
            yield format_story(dataset[i]["text"])

    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.normalizer = NFKC()
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    trainer = BpeTrainer(vocab_size=vocab_size, min_frequency=2, special_tokens=SPECIAL_TOKENS)
    tokenizer.train_from_iterator(iterator(), trainer=trainer)
    bos_id = tokenizer.token_to_id("[BOS]")
    eos_id = tokenizer.token_to_id("[EOS]")
    tokenizer.post_processor = TemplateProcessing(single="[BOS] $A [EOS]", special_tokens=[("[BOS]", bos_id), ("[EOS]", eos_id)])
    tokenizer.decoder = ByteLevelDecoder()
    os.makedirs(output_dir, exist_ok=True)
    tokenizer.save(os.path.join(output_dir, "tokenizer.json"))
    return tokenizer


def load_tokenizer(tokenizer_file: str) -> PreTrainedTokenizerFast:
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_file)
    tokenizer.pad_token = "[PAD]"
    tokenizer.unk_token = "[UNK]"
    tokenizer.bos_token = "[BOS]"
    tokenizer.eos_token = "[EOS]"
    tokenizer.mask_token = "[MASK]"
    tokenizer.add_special_tokens({"additional_special_tokens": ["<|user|>", "<|assistant|>", "<|system|>", "<|end|>"]})
    return tokenizer
