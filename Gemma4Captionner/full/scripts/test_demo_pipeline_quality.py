"""Offline regressions for the Gemma demo caption quality guard."""

from app.demo_pipeline import _has_caption_quality_risk, _has_lexical_corruption


def main() -> None:
    evidence = (
        "The digital display counts down from 08.40 to 00.00. "
        "The display briefly shows 'BUKD3' and then changes to 'ERROR' in red text."
    )

    assert not _has_lexical_corruption(
        "The countdown reaches 00.00 and the display changes to ERROR.", evidence
    )
    assert not _has_lexical_corruption(
        "The display briefly shows BUKD3 before changing to ERROR.", evidence
    )
    assert not _has_caption_quality_risk(
        evidence, "The countdown reaches zero and concludes with a red ERROR message."
    )

    assert _has_lexical_corruption("The scene contains LLBSSLSB orange text.", evidence)
    assert _has_lexical_corruption("The video is aL a montage.", evidence)
    assert _has_lexical_corruption("The result causes a totalest collapse.", evidence)

    print("demo pipeline quality regressions: ok")


if __name__ == "__main__":
    main()
