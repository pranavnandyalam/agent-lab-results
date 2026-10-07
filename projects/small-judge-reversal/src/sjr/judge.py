import os, time
from . import config as C  # must precede transformers: sets HF_HOME
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from .prompts import chat_input

def set_threads(n=4):
    os.environ.setdefault("OMP_NUM_THREADS", str(n)); os.environ.setdefault("MKL_NUM_THREADS", str(n))
    torch.set_num_threads(n)

def load_tokenizer(name):
    repo, rev = C.MODELS[name]
    return AutoTokenizer.from_pretrained(repo, revision=rev, trust_remote_code=False)

class Judge:
    def __init__(self, name):
        repo, rev = C.MODELS[name]
        self.name, self.is_qwen3 = name, name.startswith("Qwen3")
        self.tok = load_tokenizer(name)
        self.model = AutoModelForCausalLM.from_pretrained(repo, revision=rev, trust_remote_code=False,
                                                          torch_dtype=getattr(torch, C.DTYPES.get(name, "float32")), use_safetensors=True)
        self.model.eval()
        self.ids = {}
        for t in ["A", "B", " A", " B"]:
            enc = self.tok.encode(t, add_special_tokens=False)
            assert len(enc) == 1, f"{name}: {t!r} is not a single token: {enc}"
            self.ids[t] = enc[0]

    @torch.inference_mode()
    def judge(self, prompt, a, b, criterion):
        text = chat_input(self.tok, prompt, a, b, criterion, self.is_qwen3)
        enc = self.tok(text, return_tensors="pt", add_special_tokens=False)
        t0 = time.perf_counter()
        logits = self.model(**enc, logits_to_keep=1).logits[0, -1].float()
        dt = time.perf_counter() - t0
        lp = torch.log_softmax(logits, -1)
        la, lb = logits[self.ids["A"]].item(), logits[self.ids["B"]].item()
        mass = sum(lp[i].exp().item() for i in self.ids.values())
        top = int(torch.argmax(logits))
        return {"score": la - lb, "letter": "A" if la > lb else "B", "mass_AB": mass,
                "logit_sp_A_minus_B": (logits[self.ids[" A"]] - logits[self.ids[" B"]]).item(),
                "top_token": self.tok.decode([top]), "n_tokens": int(enc["input_ids"].shape[1]), "sec": dt}
