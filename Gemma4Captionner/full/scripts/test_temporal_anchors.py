"""Offline checks for the low-cost endpoint-aware frame sampler."""

from app.pipeline import _sample_frame_timestamps


def test_anchor_sampler_keeps_frame_budget_and_reaches_both_ends() -> None:
    timestamps = _sample_frame_timestamps(120.0, 16, anchored=True)
    assert len(timestamps) == 16
    assert timestamps == sorted(timestamps)
    assert timestamps[0] == 0.5
    assert timestamps[-1] == 119.5


def test_legacy_uniform_schedule_stays_available_for_ablation() -> None:
    timestamps = _sample_frame_timestamps(120.0, 4, anchored=False)
    assert timestamps == [24.0, 48.0, 72.0, 96.0]


def main() -> None:
    test_anchor_sampler_keeps_frame_budget_and_reaches_both_ends()
    test_legacy_uniform_schedule_stays_available_for_ablation()
    print("temporal_anchors_ok")


if __name__ == "__main__":
    main()
