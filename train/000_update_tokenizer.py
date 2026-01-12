from tokenizers import Tokenizer, models, processors

bpe_model = models.BPE.from_file(
    "../model/tokenizer/vocab.json",
    "../model/tokenizer/merges.txt"
)

tokenizer = Tokenizer(bpe_model)

new_special_tokens = ["[CONTEXT]", "[INPUT]", "[OUTPUT]"]
tokenizer.add_special_tokens(new_special_tokens)

tokenizer.save("../model/tokenizer_extended/tokenizer.json")
