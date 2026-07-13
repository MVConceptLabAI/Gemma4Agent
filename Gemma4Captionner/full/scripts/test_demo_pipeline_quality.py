"""Offline regressions for the Gemma demo caption quality guard."""

from app.demo_pipeline import (
    _has_caption_quality_risk,
    _has_generic_humour,
    _has_lexical_corruption,
    _has_speed_inversion,
    _safe_caption,
)


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

    traffic_evidence = "- Vehicles travel along a multi-lane road and enter a concrete tunnel."
    assert _has_generic_humour(
        "humorous_non_tech",
        "The visible sequence arrives like a family photo album, giving each grounded moment an entrance.",
    )
    assert not _has_generic_humour(
        "humorous_non_tech",
        "Vehicles enter the tunnel like a family squeezing every suitcase into one car.",
    )
    assert "Vehicles travel along a multi-lane road" in _safe_caption("humorous_non_tech", traffic_evidence)

    running_evidence = "- People run quickly along a paved race route toward the camera."
    assert _has_speed_inversion(
        "humorous_tech",
        running_evidence,
        "The runners move with the processing speed of a dial-up modem loading an image.",
    )
    assert not _has_speed_inversion(
        "humorous_tech",
        running_evidence,
        "The runners move like a low-latency network racing packets toward the finish.",
    )
    assert not _has_speed_inversion(
        "humorous_tech",
        running_evidence,
        "The runners move faster than a dial-up modem could load the starting line.",
    )
    assert _has_caption_quality_risk(
        running_evidence,
        "The runners move with the processing speed of a dial-up modem.",
        "humorous_tech",
    )

    print("demo pipeline quality regressions: ok")


if __name__ == "__main__":
    main()
