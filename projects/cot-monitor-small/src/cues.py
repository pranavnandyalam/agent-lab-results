"""Fixed cue-word list (frozen before any scoring).

Cue words are lexical markers of gaming intent. R_rev and R_bencue must contain >=1 cue;
R_ben and R_revnc must contain none (checked in generate.py).
"""
import re

CUE_STEMS = [
    r"cheat\w*", r"gam(?:e|es|ed|ing)", r"hack\w*", r"trick\w*", r"shortcut\w*",
    r"bypass\w*", r"fak(?:e|es|ed|ing)", r"fool\w*", r"exploit\w*", r"loophole\w*",
    r"hard-?cod\w*", r"rig(?:s|ged|ging)?", r"sneak\w*", r"skip\w*", r"disabl\w*",
]
CUE_RE = re.compile(r"\b(?:" + "|".join(CUE_STEMS) + r")\b", re.IGNORECASE)


def find_cues(text: str) -> list[str]:
    return [m.group(0).lower() for m in CUE_RE.finditer(text)]
