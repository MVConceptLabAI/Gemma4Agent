from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.models import (
    REQUIRED_STYLES,
    caption_passes_style_filter,
    normalize_captions,
    parse_tasks,
    validate_results,
)
from app.pipeline import _model_candidates, _provider_order


def main() -> None:
    tasks = parse_tasks(
        [
            {
                "task_id": "demo",
                "video_url": "https://example.com/demo.mp4",
                "styles": ["formal"],
            }
        ]
    )
    assert tasks[0]["styles"] == list(REQUIRED_STYLES)

    facts = {"summary": "A chef plates pasta under warm restaurant lighting."}
    captions = normalize_captions(
        {
            "formal": '"You can see a chef plates pasta under warm restaurant lighting!"',
            "sarcastic": "A chef ships dinner to production, because apparently plates need CI now.",
            "humorous_tech": "The plating service ships a garnish patch to production.",
            "humorous_non_tech": "The kitchen API deploys dinner with a database garnish.",
        },
        tasks[0]["styles"],
        facts,
    )

    assert set(captions) == set(REQUIRED_STYLES)
    assert all(captions[style].strip() for style in REQUIRED_STYLES)
    assert captions["formal"] == facts["summary"]
    assert "production" not in captions["sarcastic"].lower()
    assert caption_passes_style_filter("humorous_non_tech", captions["humorous_non_tech"])
    assert len(normalize_captions({"formal": "x" * 400}, ["formal"])["formal"]) <= 300
    assert "docker" not in normalize_captions(
        {"humorous_non_tech": "Docker deploys dinner."},
        ["humorous_non_tech"],
        {"summary": "A chef debugs a Docker deployment."},
    )["humorous_non_tech"].lower()
    assert "afro" not in normalize_captions(
        {"humorous_tech": "The office worker's afro is running a recursive algorithm."},
        ["humorous_tech"],
        {"summary": "A person types at a computer in an office."},
    )["humorous_tech"].lower()
    # Literal animals must SURVIVE the filters (the hidden set has an animals
    # category; the old low-taste list nuked captions mentioning squirrels).
    assert "squirrel" in normalize_captions(
        {"humorous_non_tech": "A squirrel sprints along the fence like it is late for a very important meeting."},
        ["humorous_non_tech"],
        {"summary": "A squirrel runs along a garden fence."},
    )["humorous_non_tech"].lower()
    assert "décrit" not in normalize_captions(
        {"formal": "Cette vidéo décrit une scène calme près de la rivière."},
        ["formal"],
    )["formal"].lower()
    assert _model_candidates("a", "b,a,c") == ["a", "b", "c"]
    assert _provider_order()
    assert _provider_order("describe")
    assert _provider_order("style")

    validated = validate_results([{"task_id": "demo", "captions": captions}])
    assert validated[0]["task_id"] == "demo"
    print("contract_test_ok")


if __name__ == "__main__":
    main()
