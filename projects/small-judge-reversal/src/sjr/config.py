import os
HF_HOME = os.environ.setdefault("HF_HOME", os.path.expanduser("~/models/hf_cache"))
RB_REPO = "allenai/reward-bench"
RB_REV = "168d848cdbbea9764fae4a544dc9ca1e6cca4931"
MODELS = {  # pinned revisions (also in results/models.md)
    "Qwen2.5-0.5B-Instruct": ("Qwen/Qwen2.5-0.5B-Instruct", "7ae557604adf67be50417f59c2c2f167def9a775"),
    "Qwen2.5-1.5B-Instruct": ("Qwen/Qwen2.5-1.5B-Instruct", "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"),
    "Qwen3-0.6B": ("Qwen/Qwen3-0.6B", "c1899de289a04d12100db370d81485cdf75e47ca"),
    "Qwen3-1.7B": ("Qwen/Qwen3-1.7B", "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"),
    "Qwen3-4B": ("Qwen/Qwen3-4B", "1cfa9a7208912126459214e8b04321603b3df60c"),
}
FILTER_TOKENIZER = "Qwen2.5-0.5B-Instruct"  # tokenizer used for all length filters
SECTIONS = {
    "chat": ["alpacaeval-easy", "alpacaeval-length", "alpacaeval-hard", "mt-bench-easy", "mt-bench-med"],
    "chat-hard": ["mt-bench-hard", "llmbar-natural", "llmbar-adver-neighbor", "llmbar-adver-GPTInst",
                  "llmbar-adver-GPTOut", "llmbar-adver-manual"],
    "reasoning": ["math-prm", "hep-cpp", "hep-go", "hep-java", "hep-js", "hep-python", "hep-rust"],
}
SAFETY_SUBSETS = ["refusals-dangerous", "refusals-offensive", "xstest-should-refuse", "xstest-should-respond",
                  "donotanswer"]
SAFETY_KEYWORDS = ["kill", "weapon", "bomb", "hack", "suicide", "drug", "explosive", "illegal", "abuse", "porn"]
MAX_RESP_TOKENS = 150
MAX_FULL_PROMPT_TOKENS = 400
N_DEV = 30
N_PER_RESAMPLE = 100
RESAMPLE_SEEDS = [0, 1, 2]
DEV_SEED = 12345
MIN_POOL = 350
SCRATCH = os.path.expanduser("~/scratch/small-judge-reversal")
