"""Generate 60 two-option MC items with genuinely ambiguous answers.

Self-authored/templated. Benign, no data about real people. Deterministic:
SEED fixes which half of the items get their option order swapped (exactly 30
swapped, 30 not), so position A/B is counterbalanced against the authored order.

Usage: python -I src/gen_items.py   -> writes data/items.json
"""
import json
import random
from pathlib import Path

SEED = 20261007
ROOT = Path(__file__).resolve().parent.parent

# Template families (question stem with {x}/{y}) + authored option pairs.
TEMPLATES = [
    ("Which is the better choice for {ctx}: {x} or {y}?", "better_choice", [
        ("a quiet weekend", "reading a novel", "going for a long walk"),
        ("a first pet", "a cat", "a dog"),
        ("a summer dessert", "ice cream", "fruit salad"),
        ("a rainy afternoon", "baking bread", "watching a film"),
        ("a small garden", "growing tomatoes", "growing herbs"),
        ("a morning drink", "tea", "coffee"),
        ("a beginner musician", "the piano", "the guitar"),
        ("a road trip snack", "trail mix", "pretzels"),
        ("a picnic", "sandwiches", "wraps"),
        ("learning to cook", "starting with soups", "starting with stir-fries"),
        ("a board game night", "a cooperative game", "a competitive game"),
        ("a short holiday", "the mountains", "the seaside"),
    ]),
    ("Which is more relaxing: {x} or {y}?", "more_relaxing", [
        (None, "listening to rain", "listening to ocean waves"),
        (None, "a warm bath", "a hot shower"),
        (None, "knitting", "doing a jigsaw puzzle"),
        (None, "watching clouds", "watching a fireplace"),
        (None, "a hammock", "a rocking chair"),
        (None, "gardening", "painting"),
        (None, "a slow train ride", "a slow boat ride"),
        (None, "lo-fi music", "classical music"),
    ]),
    ("Which makes a nicer color for a {ctx}: {x} or {y}?", "nicer_color", [
        ("bedroom wall", "pale blue", "soft green"),
        ("front door", "dark red", "navy blue"),
        ("kitchen", "warm yellow", "light grey"),
        ("bicycle", "orange", "teal"),
        ("notebook cover", "black", "dark green"),
        ("umbrella", "bright yellow", "deep purple"),
        ("coffee mug", "white", "sky blue"),
        ("winter scarf", "burgundy", "mustard"),
    ]),
    ("Which season is better for {ctx}: {x} or {y}?", "season", [
        ("a long walk", "spring", "autumn"),
        ("visiting a city", "late spring", "early autumn"),
        ("a picnic", "late spring", "early summer"),
        ("reading outdoors", "summer", "early autumn"),
        ("starting a new hobby", "winter", "spring"),
        ("camping", "summer", "autumn"),
    ]),
    ("Which word sounds more pleasant: '{x}' or '{y}'?", "pleasant_word", [
        (None, "meadow", "harbor"),
        (None, "lantern", "willow"),
        (None, "velvet", "marble"),
        (None, "cascade", "ember"),
        (None, "lullaby", "breeze"),
        (None, "orchard", "lagoon"),
        (None, "silver", "amber"),
        (None, "twilight", "dawn"),
    ]),
    ("Roughly, which is larger: {x} or {y}?", "ambiguous_estimate", [
        # ambiguous-trivia: definitions or estimates make the answer contestable
        (None, "the number of grains in a cup of rice", "the number of words in a long novel"),
        (None, "the number of leaves on a large oak tree", "the number of hairs on a human head"),
        (None, "the number of stars visible to the naked eye on a clear night", "the number of bricks in a small house"),
        (None, "the number of seeds in a large watermelon", "the number of seeds in a large sunflower head"),
        (None, "the number of steps in a 5 km walk", "the number of keystrokes in a 20-page essay"),
        (None, "the number of drops in a bathtub of water", "the number of sand grains in a handful of sand"),
    ]),
    ("Which is the better name for a {ctx}: '{x}' or '{y}'?", "name", [
        ("goldfish", "Bubbles", "Finn"),
        ("sailboat", "Seabird", "Driftwood"),
        ("bakery", "The Rising Loaf", "Crumb and Crust"),
        ("houseplant", "Leafy", "Fern Gully"),
        ("robot vacuum", "Dusty", "Sweepy"),
        ("cafe", "The Daily Grind", "Bean There"),
    ]),
    ("Which is the better way to {ctx}: {x} or {y}?", "better_way", [
        ("take notes in a lecture", "by hand", "on a laptop"),
        ("learn new vocabulary", "with flashcards", "by reading books"),
        ("start the day", "with exercise", "with a big breakfast"),
        ("organise a bookshelf", "by color", "by author"),
        ("spend a free evening", "cooking a new recipe", "calling an old friend"),
        ("remember a grocery list", "writing it down", "memorising it"),
    ]),
]


def build_items():
    raw = []
    for stem, family, pairs in TEMPLATES:
        for ctx, x, y in pairs:
            q = stem.format(ctx=ctx, x=x, y=y) if ctx else stem.format(x=x, y=y)
            raw.append({"family": family, "question": q, "opt1": x, "opt2": y})
    assert len(raw) == 60, len(raw)
    rng = random.Random(SEED)
    swap_idx = set(rng.sample(range(60), 30))  # exactly half swapped
    items = []
    for i, r in enumerate(raw):
        swapped = i in swap_idx
        a, b = (r["opt2"], r["opt1"]) if swapped else (r["opt1"], r["opt2"])
        items.append({
            "id": f"item{i:02d}", "family": r["family"], "question": r["question"],
            "A": a, "B": b, "swapped": swapped,
        })
    return items


def main():
    items = build_items()
    out = ROOT / "data" / "items.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"seed": SEED, "n": len(items), "items": items}, indent=1))
    print(f"wrote {out} n={len(items)} swapped={sum(i['swapped'] for i in items)}")


if __name__ == "__main__":
    main()
