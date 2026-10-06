"""Train a byte-level BPE (vocab 2048) on R_gen and encode all splits to uint16 token streams.
Each document is followed by the <|eod|> token id. Outputs data/tok.json, data/<split>.bin."""
import argparse, json, os
import numpy as np
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

EOD = "<|eod|>"

def read_docs(path):
    with open(path, encoding="utf-8") as f:
        return [d.strip("\n") for d in f.read().split("\n" + EOD + "\n") if d.strip()]

def train_tok(docs, vocab):
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tr = trainers.BpeTrainer(vocab_size=vocab, special_tokens=[EOD], show_progress=False,
                             initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator(docs, trainer=tr)
    return tok

def encode_docs(tok, docs):
    eod = tok.token_to_id(EOD)
    out = []
    for enc in tok.encode_batch(docs):
        out.extend(enc.ids); out.append(eod)
    return np.asarray(out, dtype=np.uint16)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--vocab", type=int, default=2048)
    a = ap.parse_args()
    tok = train_tok(read_docs(os.path.join(a.data, "R_gen.txt")), a.vocab)
    tok.save(os.path.join(a.data, "tok.json"))
    stats = {"vocab": tok.get_vocab_size()}
    for s in ["R_gen", "R_train", "R_dev", "R_val"]:
        docs = read_docs(os.path.join(a.data, f"{s}.txt"))
        ids = encode_docs(tok, docs)
        ids.tofile(os.path.join(a.data, f"{s}.bin"))
        stats[s] = {"docs": len(docs), "tokens": int(ids.size), "chars_per_tok": round(sum(map(len, docs)) / ids.size, 3)}
    print(json.dumps(stats))
    with open(os.path.join(a.data, "tok_stats.json"), "w") as f:
        json.dump(stats, f, indent=1)

if __name__ == "__main__":
    main()
