"""
Deterministic scene generator — produces N diverse scene-fact JSONs covering
the 8 content categories the hackathon evaluates on:
    nature · urban · animals · people · sports · food · weather · technology

Usage:
    python finetune/generate_scenes.py --n 200 --out finetune/scenes.jsonl
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
from pathlib import Path

# ---------------------------------------------------------------------------
# Building blocks — every combination produces a plausible short-clip scene.
# ---------------------------------------------------------------------------

# category → list of (summary_template, setting, subjects, actions, mood, audio, tech)
CATEGORIES = {
    "nature": [
        ("A river winds through a forest of pine trees at {tod}.", "forest", ["river","pines"], ["flowing"], "calm", "birds and water", False),
        ("Waves crash against black volcanic rocks on a coast at {tod}.", "rocky coast", ["waves","rocks"], ["crashing"], "energetic", "ocean surf", False),
        ("A field of wildflowers sways in the wind under a blue sky.", "meadow", ["wildflowers","sky"], ["swaying"], "cheerful", "wind", False),
        ("Snow falls quietly on pine branches in a mountain forest.", "snowy forest", ["snow","pines"], ["falling"], "melancholic", "silence", False),
        ("A waterfall drops from a mossy cliff into a green pool.", "gorge", ["waterfall","moss","pool"], ["falling"], "energetic", "waterfall roar", False),
        ("Northern lights ripple over a frozen lake.", "arctic lake", ["aurora","ice"], ["glowing"], "cinematic", "silence", False),
        ("A desert dune shifts under gusts of wind at {tod}.", "desert dunes", ["sand","wind"], ["shifting"], "melancholic", "wind", False),
        ("A hummingbird hovers next to a red flower.", "garden", ["hummingbird","flower"], ["hovering"], "energetic", "wings", False),
        ("Autumn leaves drift down through a stand of birch trees.", "birch grove", ["leaves","birches"], ["falling"], "calm", "wind and leaves", False),
        ("A misty valley reveals itself as fog lifts at sunrise.", "valley", ["mist","hills"], ["clearing"], "cinematic", "silence", False),
    ],
    "urban": [
        ("Yellow taxis crawl through a rainy intersection at night.", "city intersection", ["taxis","rain","neon"], ["driving"], "melancholic", "rain and traffic", False),
        ("Commuters cross a wide zebra crossing during {tod}.", "city street", ["pedestrians","crossing"], ["walking"], "energetic", "traffic", False),
        ("A subway train pulls into a busy underground platform.", "subway platform", ["train","commuters"], ["arriving"], "tense", "train brakes", False),
        ("Lit shop windows reflect on a wet cobblestone street.", "old-town street", ["shops","cobblestone"], ["glowing"], "cozy", "distant chatter", False),
        ("A cyclist weaves through evening traffic on a bike lane.", "avenue", ["cyclist","cars"], ["cycling"], "energetic", "traffic", False),
        ("Steam rises from a manhole cover next to a taxi stand.", "downtown", ["steam","taxis"], ["rising"], "melancholic", "traffic", False),
        ("Skyscrapers reflect a golden sunset onto a river.", "riverfront", ["skyscrapers","river","sun"], ["reflecting"], "cinematic", "wind", False),
        ("A vendor arranges fruit crates outside a corner market at dawn.", "corner market", ["vendor","crates","fruit"], ["arranging"], "calm", "distant traffic", False),
        ("A red double-decker bus passes an old stone bridge.", "bridge", ["bus","bridge","river"], ["driving"], "neutral", "engine", False),
        ("Rooftop antennas rise above a dense residential block.", "rooftops", ["antennas","buildings"], ["standing"], "neutral", "wind", False),
    ],
    "animals": [
        ("A tabby kitten stalks a butterfly through a garden.", "garden", ["kitten","butterfly"], ["stalking"], "cheerful", "outdoor ambience", False),
        ("A pack of huskies pulls a sled across a snowfield.", "snowfield", ["huskies","sled"], ["running"], "energetic", "runners on snow", False),
        ("A pod of dolphins leaps beside a fishing boat.", "ocean", ["dolphins","boat"], ["leaping"], "cheerful", "sea and engine", False),
        ("A golden retriever chases a frisbee across a park.", "park", ["retriever","frisbee"], ["running","catching"], "cheerful", "outdoor ambience", False),
        ("A curious raccoon peers out from a garbage bin at night.", "alley", ["raccoon","bin"], ["peering"], "neutral", "distant traffic", False),
        ("A herd of horses gallops across an open plain.", "plain", ["horses","grass"], ["galloping"], "energetic", "hooves", False),
        ("A parrot repeats a phrase from its perch in a living room.", "living room", ["parrot","perch"], ["speaking"], "cheerful", "voice indoors", False),
        ("A flock of pigeons scatters as a jogger passes.", "plaza", ["pigeons","jogger"], ["flying","running"], "energetic", "wings", False),
        ("A koala clings to a eucalyptus branch, chewing slowly.", "eucalyptus tree", ["koala","branch"], ["clinging","chewing"], "calm", "leaves", False),
        ("A dachshund puppy tries and fails to climb a stair.", "hallway", ["puppy","stairs"], ["climbing"], "cheerful", "indoor ambience", False),
    ],
    "people": [
        ("A grandmother knits by a fireplace in a wooden cabin.", "cabin", ["grandmother","yarn","fireplace"], ["knitting"], "cozy", "fire crackle", False),
        ("Two friends laugh over drinks on a rooftop terrace.", "rooftop", ["two friends","drinks"], ["laughing"], "cheerful", "chatter", False),
        ("A father teaches his daughter to ride a bicycle in a park.", "park", ["father","daughter","bicycle"], ["riding","teaching"], "cheerful", "outdoor ambience", False),
        ("A busker plays a violin in a marble subway station.", "subway", ["busker","violin"], ["playing"], "melancholic", "violin and echoes", False),
        ("A couple shares an umbrella walking down a rainy street.", "street", ["couple","umbrella"], ["walking"], "cozy", "rain", False),
        ("A yoga class holds a pose facing the ocean at sunrise.", "beach", ["yogis","ocean"], ["holding pose"], "calm", "waves", False),
        ("A librarian reshelves books in a quiet reading room.", "library", ["librarian","books"], ["reshelving"], "calm", "silence", False),
        ("A crowd cheers as a runner crosses a marathon finish line.", "finish line", ["runner","crowd","banner"], ["running","cheering"], "energetic", "cheers", False),
        ("A barber gives a careful cut to an elderly man in a small shop.", "barbershop", ["barber","client"], ["cutting"], "calm", "clippers", False),
        ("A child blows out candles on a homemade birthday cake.", "kitchen", ["child","cake","candles"], ["blowing"], "cheerful", "singing", False),
    ],
    "sports": [
        ("A skateboarder lands a kickflip on a concrete ramp.", "skatepark", ["skater","ramp"], ["jumping","landing"], "energetic", "wheels", False),
        ("A soccer striker curves a free kick into the top corner.", "stadium", ["striker","ball","goal"], ["kicking"], "tense", "crowd", False),
        ("A climber grips a chalked hold on an indoor bouldering wall.", "gym", ["climber","wall"], ["gripping"], "tense", "gym ambience", False),
        ("A surfer carves along the face of a barrel wave.", "ocean", ["surfer","wave"], ["surfing"], "energetic", "surf", False),
        ("Two boxers spar in a ring under bright arena lights.", "boxing ring", ["boxers","ring"], ["sparring"], "tense", "crowd", False),
        ("A cyclist attacks a steep mountain road at dawn.", "mountain road", ["cyclist","road"], ["climbing"], "tense", "wind", False),
        ("A gymnast holds a still position on a balance beam.", "gym", ["gymnast","beam"], ["holding"], "tense", "silence", False),
        ("A rally car drifts through a gravel corner in a forest stage.", "forest track", ["car","gravel"], ["drifting"], "energetic", "engine and gravel", False),
        ("A basketball player takes a fadeaway jumper.", "court", ["player","hoop"], ["jumping","shooting"], "tense", "sneaker squeaks", False),
        ("A tennis player serves an ace on a red clay court.", "clay court", ["player","ball"], ["serving"], "energetic", "impact and cheers", False),
    ],
    "food": [
        ("A chef sprinkles herbs onto a plated pasta dish.", "restaurant kitchen", ["chef","pasta","herbs"], ["plating"], "focused", "kitchen ambience", False),
        ("Hot coffee pours slowly into a ceramic mug on a wooden table.", "kitchen", ["coffee","mug"], ["pouring"], "calm", "pouring", False),
        ("A pizza slides out of a wood-fired oven.", "pizzeria", ["pizza","oven"], ["baking"], "energetic", "fire crackle", False),
        ("Sushi is arranged on a black slate board with pickled ginger.", "sushi bar", ["sushi","board"], ["arranging"], "focused", "silence", False),
        ("A market vendor fries dumplings in a large steel pan.", "street market", ["vendor","dumplings","pan"], ["frying"], "energetic", "sizzling", False),
        ("Melted chocolate drizzles onto a stack of pancakes.", "cafe", ["pancakes","chocolate"], ["drizzling"], "cheerful", "cafe ambience", False),
        ("A barista pours a rosetta into a flat white.", "coffee shop", ["barista","coffee"], ["pouring"], "focused", "espresso machine", False),
        ("Steam rises from a bowl of ramen with a soft-boiled egg.", "ramen shop", ["ramen","egg"], ["steaming"], "cozy", "chopsticks", False),
        ("A baker dusts flour onto a fresh loaf of sourdough.", "bakery", ["baker","bread"], ["dusting"], "calm", "kitchen ambience", False),
        ("A stack of tacos is topped with lime and cilantro.", "food truck", ["tacos","lime","cilantro"], ["topping"], "cheerful", "street ambience", False),
    ],
    "weather": [
        ("Lightning strikes over a distant plain during a summer storm.", "plain", ["lightning","clouds"], ["striking"], "tense", "thunder", False),
        ("Heavy rain pours onto a tin roof under a grey sky.", "porch", ["rain","roof"], ["pouring"], "melancholic", "rain on tin", False),
        ("Fog rolls into a coastal village at dawn.", "village", ["fog","houses"], ["rolling in"], "melancholic", "silence", False),
        ("A tornado forms in the distance across an open prairie.", "prairie", ["tornado","clouds"], ["forming"], "tense", "wind", False),
        ("Snow drifts pile up against a wooden fence in a whiteout.", "field", ["snow","fence"], ["drifting"], "melancholic", "wind", False),
        ("A rainbow arcs over a wet countryside road after a shower.", "country road", ["rainbow","road"], ["arcing"], "cheerful", "silence", False),
        ("Hailstones bounce off a car hood in a supermarket parking lot.", "parking lot", ["hail","car"], ["bouncing"], "tense", "impacts", False),
        ("A dust storm sweeps across a desert highway.", "highway", ["dust","road"], ["sweeping"], "tense", "wind", False),
        ("Ice crystals form on a window pane during a cold night.", "window", ["ice","glass"], ["forming"], "calm", "silence", False),
        ("A double rainbow arches over a rain-soaked mountain lake.", "mountain lake", ["rainbow","lake"], ["arching"], "cheerful", "silence", False),
    ],
    "technology": [
        ("A robotic arm assembles a circuit board in a bright factory.", "factory floor", ["robot arm","board"], ["assembling"], "focused", "servos", True),
        ("A developer types on a mechanical keyboard in a home office at night.", "home office", ["developer","keyboard"], ["typing"], "focused", "keys", True),
        ("A drone lifts off from a rooftop launch pad.", "rooftop", ["drone","pad"], ["lifting"], "energetic", "propellers", True),
        ("A 3D printer extrudes a green prototype layer by layer.", "workshop", ["printer","prototype"], ["extruding"], "focused", "printer stepper", True),
        ("Server rack LEDs blink rapidly in a cool data center aisle.", "data center", ["servers","LEDs"], ["blinking"], "tense", "fan hum", True),
        ("A VR user reaches out at objects only they can see, in a living room.", "living room", ["VR user","headset"], ["reaching"], "cheerful", "indoor ambience", True),
        ("A rocket engine ignites on a test stand, shooting flames into a trench.", "test stand", ["engine","flames"], ["igniting"], "tense", "roar", True),
        ("A humanoid robot pours water from a bottle into a glass.", "lab", ["robot","bottle","glass"], ["pouring"], "focused", "servos", True),
        ("A solar farm's panels tilt slowly to track the setting sun.", "solar farm", ["panels","sun"], ["tilting"], "calm", "wind", True),
        ("A holographic projection of a molecule spins above a conference table.", "conference room", ["hologram","molecule"], ["spinning"], "focused", "chatter", True),
    ],
}

TIMES_OF_DAY = ["sunrise", "midday", "golden hour", "dusk", "night"]


def build_scene(rng: random.Random, category: str, template) -> dict:
    (summary_tmpl, setting, subjects, actions, mood, audio, tech) = template
    tod = rng.choice(TIMES_OF_DAY)
    summary = summary_tmpl.replace("{tod}", tod)
    return {
        "category": category,
        "summary": summary,
        "setting": setting,
        "subjects": subjects,
        "actions": actions,
        "mood": mood,
        "audio_hint": audio,
        "tech_visible": tech,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--out", type=Path, default=Path("finetune/scenes.jsonl"))
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rng = random.Random(args.seed)
    # Deterministic, balanced draw across the 8 categories (guide list).
    per_cat = args.n // len(CATEGORIES)
    remainder = args.n - per_cat * len(CATEGORIES)
    scenes: list[dict] = []
    for i, (cat, templates) in enumerate(CATEGORIES.items()):
        # Cycle through templates so identical templates never adjoin.
        picks = list(itertools.islice(itertools.cycle(templates), per_cat + (1 if i < remainder else 0)))
        rng.shuffle(picks)
        for t in picks:
            scenes.append(build_scene(rng, cat, t))

    rng.shuffle(scenes)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "\n".join(json.dumps(s, ensure_ascii=False) for s in scenes),
        encoding="utf-8",
    )
    print(f"Wrote {len(scenes)} scenes -> {args.out}")
    # Print a quick category breakdown so you can sanity-check.
    from collections import Counter
    print(dict(Counter(s["category"] for s in scenes)))


if __name__ == "__main__":
    main()
