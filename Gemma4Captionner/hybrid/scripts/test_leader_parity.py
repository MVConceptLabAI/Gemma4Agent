"""Offline assertion tests for leader-parity frame sampling helpers.

Run with::

    PYTHONPATH=. python scripts/test_leader_parity.py
"""

from __future__ import annotations

import math
import shutil
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from app.leader_parity import (
    FACTUAL_REVIEW_STATUS,
    REQUESTED_STYLES,
    caption_violations,
    normalize_requested_captions,
    parse_json_object,
    validate_caption_set,
)
import app.pipeline as pipeline
from app.pipeline import (
    FRAME_PROFILES,
    _extract_frames_at_timestamps,
    _extract_official_repo_frames,
    _extract_ratio_frames,
    _ffmpeg_fps_extract,
    _official_repo_indices,
    _ratio_timestamps,
)


def _assert_raises(error_type: type[BaseException], operation, message: str) -> BaseException:
    try:
        operation()
    except error_type as error:
        return error
    raise AssertionError(message)


def test_public_timestamp_profiles_are_exact() -> None:
    assert set(FRAME_PROFILES) == {"describex_oci_hypothesis", "endpoint_aware"}
    assert _ratio_timestamps(100.0, "describex_oci_hypothesis") == [
        5.0,
        16.25,
        27.5,
        38.75,
        50.0,
        61.25,
        72.5,
        83.75,
    ]
    assert _ratio_timestamps(100.0, "endpoint_aware") == [
        5.0,
        17.857,
        30.714,
        43.571,
        56.429,
        69.286,
        82.143,
        95.0,
    ]


def test_thirty_second_profiles_are_ordered_and_bounded() -> None:
    for profile in FRAME_PROFILES:
        timestamps = _ratio_timestamps(30.0, profile)
        assert len(timestamps) == 8
        assert timestamps == sorted(timestamps)
        assert len(set(timestamps)) == 8
        assert all(0.0 < timestamp < 30.0 for timestamp in timestamps)
        assert all(timestamp == round(timestamp, 3) for timestamp in timestamps)

    assert _ratio_timestamps(30.0, "describex_oci_hypothesis") == [
        1.5,
        4.875,
        8.25,
        11.625,
        15.0,
        18.375,
        21.75,
        25.125,
    ]
    assert _ratio_timestamps(30.0, "endpoint_aware") == [
        1.5,
        5.357,
        9.214,
        13.071,
        16.929,
        20.786,
        24.643,
        28.5,
    ]


def test_ratio_timestamp_validation() -> None:
    for duration in (0.0, -1.0, math.inf, -math.inf, math.nan):
        _assert_raises(
            ValueError,
            lambda duration=duration: _ratio_timestamps(
                duration, "describex_oci_hypothesis"
            ),
            f"invalid duration was accepted: {duration!r}",
        )
    _assert_raises(
        ValueError,
        lambda: _ratio_timestamps(30.0, "unknown"),
        "unknown frame profile was accepted",
    )


def _run_fixture_ffmpeg(arguments: list[str]) -> None:
    executable = shutil.which("ffmpeg")
    assert executable is not None, "real-media sampling tests require ffmpeg"
    subprocess.run(
        [executable, "-hide_banner", "-loglevel", "error", "-y", *arguments],
        check=True,
        capture_output=True,
        timeout=10.0,
    )


def _center_rgb(path: Path) -> tuple[int, int, int]:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        return rgb.getpixel((rgb.width // 2, rgb.height // 2))


def test_real_single_frame_clip_accepts_zero_pts() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        video = root / "single_frame.mp4"
        workdir = root / "frames"
        workdir.mkdir()
        _run_fixture_ffmpeg(
            [
                "-f",
                "lavfi",
                "-i",
                "color=c=green:s=64x64:r=1:d=1",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(video),
            ]
        )

        assert pipeline._ffprobe_video_last_pts(video, timeout_s=3.0) == 0.0
        frames = _extract_ratio_frames(
            video, workdir, "endpoint_aware", max_edge=64
        )
        assert len(frames) == 8
        assert all(_center_rgb(frame)[1] > 100 for frame in frames)


def test_real_low_fps_clip_represents_beginning_and_end() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        video = root / "low_fps.mp4"
        workdir = root / "frames"
        workdir.mkdir()
        _run_fixture_ffmpeg(
            [
                "-f",
                "lavfi",
                "-i",
                "color=c=red:s=64x64:r=1:d=1",
                "-f",
                "lavfi",
                "-i",
                "color=c=blue:s=64x64:r=1:d=1",
                "-filter_complex",
                "[0:v][1:v]concat=n=2:v=1:a=0",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(video),
            ]
        )

        assert pipeline._ffprobe_duration(video) == 2.0
        assert pipeline._ffprobe_video_last_pts(video, timeout_s=3.0) == 1.0
        frames = _extract_ratio_frames(
            video, workdir, "endpoint_aware", max_edge=64
        )
        assert len(frames) == 8
        assert all(frame.stat().st_size > 0 for frame in frames)
        first = _center_rgb(frames[0])
        last = _center_rgb(frames[-1])
        assert first[0] > 200 and first[2] < 40, first
        assert last[2] > 200 and last[0] < 40, last
        assert len({_center_rgb(frame) for frame in frames}) >= 2


def test_real_audio_longer_than_video_uses_video_timeline() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        video = root / "audio_long.mp4"
        workdir = root / "frames"
        workdir.mkdir()
        _run_fixture_ffmpeg(
            [
                "-f",
                "lavfi",
                "-i",
                "color=c=blue:s=64x64:r=5:d=1",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=440:duration=2",
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                str(video),
            ]
        )

        assert pipeline._ffprobe_duration(video) >= 1.9
        assert pipeline._ffprobe_video_last_pts(video, timeout_s=3.0) == 0.8
        frames = _extract_ratio_frames(
            video, workdir, "endpoint_aware", max_edge=64
        )
        assert len(frames) == 8
        assert all(frame.stat().st_size > 0 for frame in frames)


def test_official_repo_indices_match_reference_and_edges() -> None:
    assert _official_repo_indices(60) == [
        0,
        3,
        7,
        11,
        15,
        18,
        22,
        26,
        30,
        33,
        37,
        41,
        45,
        48,
        52,
        59,
    ]
    assert _official_repo_indices(1) == [0]
    assert _official_repo_indices(3) == [0, 1, 2]
    assert _official_repo_indices(16) == list(range(16))
    assert _official_repo_indices(17) == [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        16,
    ]
    _assert_raises(
        ValueError,
        lambda: _official_repo_indices(60, target=1),
        "target=1 cannot preserve both the first and last frame",
    )
    for total_frames in (0, -1):
        _assert_raises(
            ValueError,
            lambda total_frames=total_frames: _official_repo_indices(total_frames),
            f"invalid frame count was accepted: {total_frames}",
        )


def test_ffmpeg_fps_extract_command_and_nonempty_sorted_output() -> None:
    with tempfile.TemporaryDirectory() as directory:
        workdir = Path(directory)
        video = workdir / "clip.mp4"

        def create_frames(command: list[str], **_: object) -> None:
            output_dir = Path(command[-1]).parent
            (output_dir / "frame_002.jpg").write_bytes(b"second")
            (output_dir / "frame_001.jpg").write_bytes(b"first")
            (output_dir / "frame_003.jpg").touch()

        with patch("app.pipeline.subprocess.run", side_effect=create_frames) as run:
            frames = _ffmpeg_fps_extract(
                video=video,
                workdir=workdir,
                fps=2.5,
                qscale=2,
                total_timeout_s=12.0,
            )

        expected_pattern = workdir / "official_repo" / "frame_%03d.jpg"
        run.assert_called_once_with(
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(video),
                "-vf",
                "fps=2.50000000",
                "-q:v",
                "2",
                str(expected_pattern),
            ],
            check=True,
            timeout=12.0,
        )
        assert [frame.name for frame in frames] == ["frame_001.jpg", "frame_002.jpg"]
        assert all(frame.stat().st_size > 0 for frame in frames)


def test_ffmpeg_fps_extract_rejects_empty_output_and_invalid_controls() -> None:
    with tempfile.TemporaryDirectory() as directory:
        workdir = Path(directory)
        video = workdir / "clip.mp4"
        with patch("app.pipeline.subprocess.run"):
            _assert_raises(
                RuntimeError,
                lambda: _ffmpeg_fps_extract(video, workdir, 1.0, 2, 12.0),
                "empty FFmpeg output was accepted",
            )

        for fps in (0.0, -1.0, math.inf, math.nan):
            _assert_raises(
                ValueError,
                lambda fps=fps: _ffmpeg_fps_extract(video, workdir, fps, 2, 12.0),
                f"invalid extraction fps was accepted: {fps!r}",
            )
        for timeout in (0.0, -1.0, math.inf, math.nan):
            _assert_raises(
                ValueError,
                lambda timeout=timeout: _ffmpeg_fps_extract(
                    video, workdir, 1.0, 2, timeout
                ),
                f"invalid extraction timeout was accepted: {timeout!r}",
            )


def test_ffmpeg_fps_extract_propagates_timeout() -> None:
    with tempfile.TemporaryDirectory() as directory:
        workdir = Path(directory)
        video = workdir / "clip.mp4"
        timeout = subprocess.TimeoutExpired(["ffmpeg"], 12.0)
        with patch("app.pipeline.subprocess.run", side_effect=timeout) as run:
            caught = _assert_raises(
                subprocess.TimeoutExpired,
                lambda: _ffmpeg_fps_extract(video, workdir, 1.0, 2, 12.0),
                "FFmpeg timeout was swallowed",
            )
        assert caught is timeout
        assert run.call_args.kwargs["check"] is True
        assert run.call_args.kwargs["timeout"] == 12.0


def test_official_repo_extraction_uses_minimum_fps_and_reference_sampling() -> None:
    with tempfile.TemporaryDirectory() as directory:
        workdir = Path(directory)
        video = workdir / "clip.mp4"
        extracted = [workdir / f"source_{index:03d}.jpg" for index in range(60)]

        with patch(
            "app.pipeline._ffmpeg_fps_extract", return_value=extracted
        ) as fps_extract, patch(
            "app.pipeline._extract_scene_frames",
            side_effect=AssertionError("scene detection must not be used"),
        ) as scene_extract:
            selected = _extract_official_repo_frames(
                video=video,
                workdir=workdir,
                video_fps=24.0,
                duration=30.0,
            )

        fps_extract.assert_called_once_with(
            video=video,
            workdir=workdir,
            fps=2.0,
            qscale=2,
            total_timeout_s=12.0,
        )
        scene_extract.assert_not_called()
        expected_indices = _official_repo_indices(60)
        assert selected == [extracted[index] for index in expected_indices]

        with patch(
            "app.pipeline._ffmpeg_fps_extract", return_value=extracted[:10]
        ) as fps_extract:
            selected = _extract_official_repo_frames(video, workdir, 1.5, 10.0)
        assert fps_extract.call_args.kwargs["fps"] == 1.5
        assert selected == extracted[:10]


def test_official_repo_extraction_rejects_invalid_media_metadata() -> None:
    with tempfile.TemporaryDirectory() as directory:
        workdir = Path(directory)
        video = workdir / "clip.mp4"
        for video_fps, duration in (
            (0.0, 30.0),
            (-1.0, 30.0),
            (math.inf, 30.0),
            (math.nan, 30.0),
            (24.0, 0.0),
            (24.0, -1.0),
            (24.0, math.inf),
            (24.0, math.nan),
        ):
            _assert_raises(
                ValueError,
                lambda video_fps=video_fps, duration=duration: (
                    _extract_official_repo_frames(
                        video, workdir, video_fps, duration
                    )
                ),
                f"invalid metadata was accepted: fps={video_fps!r}, duration={duration!r}",
            )


def test_timestamp_extractor_command_order_and_nonempty_outputs() -> None:
    with tempfile.TemporaryDirectory() as directory:
        workdir = Path(directory)
        video = workdir / "clip.mp4"
        timestamps = [1.25, 2.5]

        def create_frame(command: list[str], **_: object) -> None:
            Path(command[-1]).write_bytes(command[6].encode("ascii"))

        with patch("app.pipeline.subprocess.run", side_effect=create_frame) as run:
            frames = _extract_frames_at_timestamps(
                video=video,
                workdir=workdir,
                timestamps=timestamps,
                max_edge=768,
                jpeg_quality=85,
            )

        assert frames == [workdir / "leader_01.jpg", workdir / "leader_02.jpg"]
        assert [frame.read_text(encoding="ascii") for frame in frames] == [
            "1.250",
            "2.500",
        ]
        expected_scale = (
            "scale='min(768,iw)':'min(768,ih)':force_original_aspect_ratio=decrease"
        )
        expected_commands = [
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                f"{timestamp:.3f}",
                "-i",
                str(video),
                "-frames:v",
                "1",
                "-vf",
                expected_scale,
                "-pix_fmt",
                "yuvj420p",
                "-q:v",
                "6",
                str(workdir / f"leader_{index:02d}.jpg"),
            ]
            for index, timestamp in enumerate(timestamps, start=1)
        ]
        assert [entry.args[0] for entry in run.call_args_list] == expected_commands
        assert all(entry.kwargs["check"] is True for entry in run.call_args_list)
        assert all(0.0 < entry.kwargs["timeout"] <= 3.0 for entry in run.call_args_list)


def test_timestamp_extractor_rejects_one_missing_or_empty_frame() -> None:
    with tempfile.TemporaryDirectory() as directory:
        workdir = Path(directory)
        video = workdir / "clip.mp4"
        calls = 0

        def omit_second_frame(command: list[str], **_: object) -> None:
            nonlocal calls
            calls += 1
            if calls == 1:
                Path(command[-1]).write_bytes(b"frame")
            else:
                Path(command[-1]).touch()

        with patch("app.pipeline.subprocess.run", side_effect=omit_second_frame):
            error = _assert_raises(
                RuntimeError,
                lambda: _extract_frames_at_timestamps(
                    video, workdir, [1.0, 2.0], 768, 85
                ),
                "partial frame extraction was accepted",
            )
        assert "1" in str(error) and "2" in str(error)
        assert "8" not in str(error), "error text hardcodes an eight-frame request"


def test_timestamp_extractor_rejects_empty_timestamps() -> None:
    with tempfile.TemporaryDirectory() as directory:
        workdir = Path(directory)
        video = workdir / "clip.mp4"
        with patch("app.pipeline.subprocess.run") as run:
            _assert_raises(
                ValueError,
                lambda: _extract_frames_at_timestamps(video, workdir, [], 768, 85),
                "an empty timestamp request was accepted",
            )
        run.assert_not_called()


def test_timestamp_extractor_enforces_deadline_and_timestamp_validation() -> None:
    with tempfile.TemporaryDirectory() as directory:
        workdir = Path(directory)
        video = workdir / "clip.mp4"
        with patch("app.pipeline.time.monotonic", side_effect=[10.0, 22.0]), patch(
            "app.pipeline.subprocess.run"
        ) as run:
            _assert_raises(
                TimeoutError,
                lambda: _extract_frames_at_timestamps(video, workdir, [1.0], 768, 85),
                "expired total extraction deadline was ignored",
            )
        run.assert_not_called()

        for timestamps in (
            [0.0],
            [-1.0],
            [math.inf],
            [math.nan],
            [2.0, 1.0],
            [1.0, 1.0],
        ):
            with patch("app.pipeline.subprocess.run") as run:
                _assert_raises(
                    ValueError,
                    lambda timestamps=timestamps: _extract_frames_at_timestamps(
                        video, workdir, timestamps, 768, 85
                    ),
                    f"invalid timestamps were accepted: {timestamps!r}",
                )
            run.assert_not_called()


def test_ratio_extractor_uses_ffprobe_and_never_scene_detection() -> None:
    with tempfile.TemporaryDirectory() as directory:
        workdir = Path(directory)
        video = workdir / "clip.mp4"
        expected_frames = [workdir / f"leader_{index:02d}.jpg" for index in range(1, 9)]
        expected_timestamps = [0.0, 0.0, 0.0, 0.0, 30.0, 30.0, 30.0, 30.0]
        with patch("app.pipeline.time.monotonic", side_effect=[100.0, 100.0]), patch(
            "app.pipeline._ffprobe_video_timestamps", return_value=[0.0, 30.0]
        ) as ffprobe, patch(
            "app.pipeline._extract_frames_at_timestamps", return_value=expected_frames
        ) as extract, patch(
            "app.pipeline._extract_scene_frames",
            side_effect=AssertionError("scene detection must not be used"),
        ) as scene_extract:
            frames = _extract_ratio_frames(
                video=video,
                workdir=workdir,
                profile="endpoint_aware",
            )

        assert frames == expected_frames
        ffprobe.assert_called_once_with(video, timeout_s=3.0)
        extract.assert_called_once_with(
            video=video,
            workdir=workdir,
            timestamps=expected_timestamps,
            max_edge=768,
            jpeg_quality=85,
            deadline=112.0,
            allow_repeated=True,
        )
        scene_extract.assert_not_called()


def test_ratio_extractor_bounds_ffprobe_timeout_and_propagates_it() -> None:
    with tempfile.TemporaryDirectory() as directory:
        workdir = Path(directory)
        video = workdir / "clip.mp4"
        timeout = subprocess.TimeoutExpired(["ffprobe"], 3.0)
        with patch(
            "app.pipeline.subprocess.check_output", side_effect=timeout
        ) as ffprobe, patch("app.pipeline._extract_frames_at_timestamps") as extract:
            caught = _assert_raises(
                subprocess.TimeoutExpired,
                lambda: _extract_ratio_frames(video, workdir, "endpoint_aware"),
                "ffprobe timeout was swallowed",
            )

        assert caught is timeout
        command = ffprobe.call_args.args[0]
        assert command[command.index("-select_streams") + 1] == "v:0"
        assert command[command.index("-show_entries") + 1] == "packet=pts_time"
        assert 0.0 < ffprobe.call_args.kwargs["timeout"] <= 3.0
        extract.assert_not_called()


def test_ratio_extractor_shares_one_twelve_second_deadline() -> None:
    with tempfile.TemporaryDirectory() as directory:
        workdir = Path(directory)
        video = workdir / "clip.mp4"
        expected_frames = [workdir / f"leader_{index:02d}.jpg" for index in range(1, 9)]
        with patch("app.pipeline.time.monotonic", side_effect=[100.0, 100.0]), patch(
            "app.pipeline.subprocess.check_output", return_value="10.0\n"
        ) as ffprobe, patch(
            "app.pipeline._extract_frames_at_timestamps", return_value=expected_frames
        ) as extract:
            frames = _extract_ratio_frames(video, workdir, "endpoint_aware")

        assert frames == expected_frames
        assert ffprobe.call_args.kwargs["timeout"] == 3.0
        assert extract.call_args.kwargs["deadline"] == 112.0


def test_invalid_ffprobe_output_is_rejected_before_extraction() -> None:
    with tempfile.TemporaryDirectory() as directory:
        workdir = Path(directory)
        video = workdir / "clip.mp4"
        with patch(
            "app.pipeline.subprocess.check_output", return_value="not-a-duration"
        ), patch("app.pipeline._extract_frames_at_timestamps") as extract:
            _assert_raises(
                ValueError,
                lambda: _extract_ratio_frames(
                    video, workdir, "describex_oci_hypothesis"
                ),
                "invalid ffprobe output was accepted",
            )
        extract.assert_not_called()


def _counted_caption(
    sentence_count: int,
    word_count: int,
    *,
    required_words: tuple[str, ...] = (),
    prefix: str = "detail",
) -> str:
    assert sentence_count > 0
    assert word_count >= sentence_count
    assert len(required_words) <= word_count
    words = [*required_words]
    token_prefix = "".join(
        character for character in prefix if character.isalpha()
    )[:2] or "w"
    words.extend(
        f"{token_prefix}{index}"
        for index in range(word_count - len(required_words))
    )
    base_size, extra = divmod(word_count, sentence_count)
    sentences: list[str] = []
    offset = 0
    for index in range(sentence_count):
        size = base_size + (1 if index < extra else 0)
        sentence_words = words[offset:offset + size]
        sentences.append(" ".join(sentence_words) + ".")
        offset += size
    return " ".join(sentences)


def _caption_with_exact_length(length: int) -> str:
    caption = _counted_caption(3, 80, prefix="lengthword")
    padding = length - len(caption)
    assert padding >= 0
    first_word, remainder = caption.split(" ", 1)
    exact = first_word + ("x" * padding) + " " + remainder
    assert len(" ".join(exact.split())) == length
    return exact


def test_parse_json_object_accepts_fences_prose_and_nested_objects() -> None:
    expected = {
        "formal": "A caption.",
        "metadata": {"source": "panel", "literal": "brace { text }"},
    }
    encoded = (
        '{"formal":"A caption.","metadata":'
        '{"source":"panel","literal":"brace { text }"}}'
    )
    assert parse_json_object(f"```json\n{encoded}\n```") == expected
    assert parse_json_object(f"Writer output follows:\n{encoded}\nEnd output.") == expected
    assert parse_json_object(f"Prose cites [2026] before {encoded}") == expected


def test_parse_json_object_rejects_zero_multiple_and_root_list() -> None:
    invalid_inputs = (
        "No structured output was returned.",
        '{"formal":"first"}\n{"formal":"second"}',
        '[{"formal":"inside a list"}]',
        'Broken wrapper {"meta":{"formal":"salvaged"}',
    )
    for text in invalid_inputs:
        _assert_raises(
            ValueError,
            lambda text=text: parse_json_object(text),
            f"invalid JSON envelope was accepted: {text!r}",
        )


def test_parse_json_object_rejects_duplicate_keys() -> None:
    _assert_raises(
        ValueError,
        lambda: parse_json_object('{"formal":"first","formal":"second"}'),
        "duplicate JSON keys were silently overwritten",
    )


def test_normalize_requested_captions_maps_only_requested_aliases() -> None:
    normalized = normalize_requested_captions(
        {
            "humorous-tech": "  Tech caption.  ",
            "humorous-non-tech": "  Everyday caption.  ",
        },
        ["humorous_tech", "humorous_non_tech"],
    )
    assert normalized == {
        "humorous_tech": "Tech caption.",
        "humorous_non_tech": "Everyday caption.",
    }
    for alias in ("humorous-tech", "humorous-non-tech"):
        _assert_raises(
            ValueError,
            lambda alias=alias: normalize_requested_captions(
                {alias: "Creative caption."}, ["formal"]
            ),
            f"an unrequested alias was mapped: {alias}",
        )


def test_normalize_requested_captions_rejects_collisions_and_bad_values() -> None:
    for alias, canonical in (
        ("humorous-tech", "humorous_tech"),
        ("humorous-non-tech", "humorous_non_tech"),
    ):
        _assert_raises(
            ValueError,
            lambda alias=alias, canonical=canonical: normalize_requested_captions(
                {canonical: "Canonical.", alias: "Alias."}, [canonical]
            ),
            f"caption alias collision was accepted: {alias}",
        )

    invalid_payloads = (
        {"formal": "Formal only."},
        {"formal": "Formal.", "sarcastic": "Sarcastic.", "extra": "No."},
        {"formal": "   ", "sarcastic": "Sarcastic."},
        {"formal": 123, "sarcastic": "Sarcastic."},
    )
    requested = ["formal", "sarcastic"]
    for payload in invalid_payloads:
        _assert_raises(
            ValueError,
            lambda payload=payload: normalize_requested_captions(payload, requested),
            f"invalid caption payload was accepted: {payload!r}",
        )


def test_formal_sentence_and_word_bands_have_inclusive_boundaries() -> None:
    for sentence_count, word_count in ((3, 80), (9, 220)):
        assert caption_violations(
            "formal", _counted_caption(sentence_count, word_count)
        ) == []

    for sentence_count in (2, 10):
        assert "sentence_count" in caption_violations(
            "formal", _counted_caption(sentence_count, 80)
        )
    for word_count in (79, 221):
        assert "word_count" in caption_violations(
            "formal", _counted_caption(3, word_count)
        )

    complete_sentences = _counted_caption(3, 159, prefix="finished")
    with_trailing_fragment = complete_sentences + " unfinished"
    assert "sentence_count" in caption_violations(
        "formal", with_trailing_fragment
    )


def test_creative_sentence_and_word_bands_have_inclusive_boundaries() -> None:
    creative_styles = (
        "sarcastic",
        "humorous_tech",
        "humorous_non_tech",
    )
    for style in creative_styles:
        required = ("JavaScript",) if style == "humorous_tech" else ()
        for sentence_count, word_count in ((2, 40), (6, 150)):
            assert caption_violations(
                style,
                _counted_caption(
                    sentence_count,
                    word_count,
                    required_words=required,
                    prefix=style.replace("_", ""),
                ),
            ) == []
        for sentence_count in (1, 7):
            assert "sentence_count" in caption_violations(
                style,
                _counted_caption(
                    sentence_count,
                    40,
                    required_words=required,
                    prefix=style.replace("_", ""),
                ),
            )
        for word_count in (39, 151):
            assert "word_count" in caption_violations(
                style,
                _counted_caption(
                    2,
                    word_count,
                    required_words=required,
                    prefix=style.replace("_", ""),
                ),
            )
        complete_sentences = _counted_caption(
            2,
            39,
            required_words=required,
            prefix=style.replace("_", ""),
        )
        assert "sentence_count" in caption_violations(
            style, complete_sentences + " unfinished"
        )


def test_character_limit_is_1600_normalized_characters_inclusive() -> None:
    at_limit = _caption_with_exact_length(1600)
    over_limit = _caption_with_exact_length(1601)
    raw_whitespace_over_limit = at_limit.replace(" ", "   ")
    assert len(raw_whitespace_over_limit) > 1600
    for style in REQUESTED_STYLES:
        assert "character_count" not in caption_violations(style, at_limit)
        assert "character_count" in caption_violations(style, over_limit)
        assert "character_count" not in caption_violations(
            style, raw_whitespace_over_limit
        )


def test_jargon_and_tech_markers_use_whole_words() -> None:
    substring_traps = ("appearance", "happily", "apple", "within", "details")
    non_tech = _counted_caption(
        2, 40, required_words=substring_traps, prefix="ordinary"
    )
    for style in ("sarcastic", "humorous_non_tech"):
        assert "technical_jargon" not in caption_violations(style, non_tech)

    ambiguous_senses = _counted_caption(
        2,
        40,
        required_words=(
            "cloud",
            "model",
            "bug",
            "stack",
            "program",
            "server",
            "cache",
            "python",
        ),
        prefix="ordinary",
    )
    for style in ("sarcastic", "humorous_non_tech"):
        assert "technical_jargon" not in caption_violations(
            style, ambiguous_senses
        )
    assert "missing_tech_reference" in caption_violations(
        "humorous_tech", ambiguous_senses
    )

    actual_jargon = _counted_caption(
        2, 40, required_words=("JavaScript",), prefix="ordinary"
    )
    for style in ("sarcastic", "humorous_non_tech"):
        assert "technical_jargon" in caption_violations(style, actual_jargon)

    whole_word_app = _counted_caption(
        2, 40, required_words=("app",), prefix="ordinary"
    )
    for style in ("sarcastic", "humorous_non_tech"):
        assert "technical_jargon" in caption_violations(style, whole_word_app)
    assert "missing_tech_reference" not in caption_violations(
        "humorous_tech", whole_word_app
    )

    compound_jargon = _counted_caption(
        2, 40, required_words=("AI-powered",), prefix="ordinary"
    )
    for style in ("sarcastic", "humorous_non_tech"):
        assert "technical_jargon" in caption_violations(style, compound_jargon)

    assert "missing_tech_reference" in caption_violations(
        "humorous_tech", non_tech
    )
    natural_tech = _counted_caption(
        2, 40, required_words=("JavaScript",), prefix="techscene"
    )
    assert caption_violations("humorous_tech", natural_tech) == []
    for reference in ("server-side", "laptop", "Python script"):
        contextual_tech = _counted_caption(
            2, 40, required_words=tuple(reference.split()), prefix="techscene"
        )
        assert "missing_tech_reference" not in caption_violations(
            "humorous_tech", contextual_tech
        )

    laptop_scene = _counted_caption(
        2, 40, required_words=("laptop",), prefix="ordinary"
    )
    for style in ("sarcastic", "humorous_non_tech"):
        assert "technical_jargon" not in caption_violations(style, laptop_scene)

    for reference in ("server-side", "Python script"):
        contextual_jargon = _counted_caption(
            2, 40, required_words=tuple(reference.split()), prefix="ordinary"
        )
        for style in ("sarcastic", "humorous_non_tech"):
            assert "technical_jargon" in caption_violations(
                style, contextual_jargon
            )


def test_leaked_analysis_is_detected() -> None:
    leaked = _counted_caption(
        2,
        40,
        required_words=("Let's", "analyze", "the", "image", "step", "by", "step"),
        prefix="leakword",
    )
    assert "leaked_reasoning" in caption_violations("sarcastic", leaked)

    grounded = _counted_caption(
        2,
        40,
        required_words=("A", "hiker", "moves", "step", "by", "step"),
        prefix="grounded",
    )
    assert "leaked_reasoning" not in caption_violations("sarcastic", grounded)


def test_repeated_substantial_fragment_is_detected() -> None:
    fragment = ["bright", "birds", "circle", "above", "the", "quiet", "green", "field"]
    repeated_words = fragment * 2
    repeated_words.extend(f"unique{index}" for index in range(24))
    repeated = (
        " ".join(repeated_words[:20]) + ". "
        + " ".join(repeated_words[20:]) + "."
    )
    assert "repeated_fragment" in caption_violations("sarcastic", repeated)


def test_substantially_non_english_latin_text_is_detected() -> None:
    spanish_words = [
        "el", "gato", "camina", "lentamente", "por", "el", "bosque", "mientras",
        "la", "luz", "suave", "atraviesa", "las", "ramas", "y", "las", "hojas",
        "verdes", "cubren", "el", "sendero", "con", "sombras", "largas", "sobre",
        "la", "tierra", "humeda", "mientras", "una", "brisa", "mueve", "los",
        "arbustos", "cerca", "del", "camino", "bajo", "el", "cielo",
    ]
    assert len(spanish_words) == 40
    non_english = (
        " ".join(spanish_words[:20]) + ". "
        + " ".join(spanish_words[20:]) + "."
    )
    assert "non_english" in caption_violations("humorous_non_tech", non_english)

    non_latin = "猫 在 绿色 森林 里 安静 地 行走。" * 8
    assert "non_english" in caption_violations("humorous_non_tech", non_latin)


def test_validate_caption_set_flags_cross_style_near_duplicates() -> None:
    caption = (
        " ".join(f"word{index}" for index in range(20))
        + ". "
        + " ".join(f"word{index}" for index in range(20, 40))
        + "."
    )
    variant = caption.replace("word0", "changedword", 1).replace(" ", "   ")
    assert len(caption) > 200
    assert variant != caption
    failures = validate_caption_set(
        {"sarcastic": caption, "humorous_non_tech": variant},
        ["sarcastic", "humorous_non_tech"],
    )
    assert failures == {"humorous_non_tech": ["near_duplicate:sarcastic"]}


def test_invented_claims_require_independent_factual_review() -> None:
    factual_status = FACTUAL_REVIEW_STATUS
    assert factual_status == "requires_independent_factual_review"
    claims = ("unicorn", "forest", "mountain-lake")
    cases = {
        "formal": (
            _counted_caption(
                3, 80, required_words=claims, prefix="formalclaim"
            ),
            factual_status,
        ),
        "sarcastic": (
            _counted_caption(
                2, 40, required_words=claims, prefix="sarcasticclaim"
            ),
            factual_status,
        ),
        "humorous_tech": (
            _counted_caption(
                2,
                40,
                required_words=(*claims, "JavaScript"),
                prefix="techclaim",
            ),
            factual_status,
        ),
        "humorous_non_tech": (
            _counted_caption(
                2, 40, required_words=claims, prefix="everydayclaim"
            ),
            factual_status,
        ),
    }
    captions = {style: case[0] for style, case in cases.items()}
    assert set(captions) == set(REQUESTED_STYLES)
    assert validate_caption_set(captions, list(REQUESTED_STYLES)) == {}
    assert {case[1] for case in cases.values()} == {factual_status}
    for caption in captions.values():
        assert all(claim in caption for claim in claims)


def main() -> None:
    test_public_timestamp_profiles_are_exact()
    test_thirty_second_profiles_are_ordered_and_bounded()
    test_ratio_timestamp_validation()
    test_real_single_frame_clip_accepts_zero_pts()
    test_real_low_fps_clip_represents_beginning_and_end()
    test_real_audio_longer_than_video_uses_video_timeline()
    test_official_repo_indices_match_reference_and_edges()
    test_ffmpeg_fps_extract_command_and_nonempty_sorted_output()
    test_ffmpeg_fps_extract_rejects_empty_output_and_invalid_controls()
    test_ffmpeg_fps_extract_propagates_timeout()
    test_official_repo_extraction_uses_minimum_fps_and_reference_sampling()
    test_official_repo_extraction_rejects_invalid_media_metadata()
    test_timestamp_extractor_command_order_and_nonempty_outputs()
    test_timestamp_extractor_rejects_one_missing_or_empty_frame()
    test_timestamp_extractor_rejects_empty_timestamps()
    test_timestamp_extractor_enforces_deadline_and_timestamp_validation()
    test_ratio_extractor_uses_ffprobe_and_never_scene_detection()
    test_ratio_extractor_bounds_ffprobe_timeout_and_propagates_it()
    test_ratio_extractor_shares_one_twelve_second_deadline()
    test_invalid_ffprobe_output_is_rejected_before_extraction()
    test_parse_json_object_accepts_fences_prose_and_nested_objects()
    test_parse_json_object_rejects_zero_multiple_and_root_list()
    test_parse_json_object_rejects_duplicate_keys()
    test_normalize_requested_captions_maps_only_requested_aliases()
    test_normalize_requested_captions_rejects_collisions_and_bad_values()
    test_formal_sentence_and_word_bands_have_inclusive_boundaries()
    test_creative_sentence_and_word_bands_have_inclusive_boundaries()
    test_character_limit_is_1600_normalized_characters_inclusive()
    test_jargon_and_tech_markers_use_whole_words()
    test_leaked_analysis_is_detected()
    test_repeated_substantial_fragment_is_detected()
    test_substantially_non_english_latin_text_is_detected()
    test_validate_caption_set_flags_cross_style_near_duplicates()
    test_invented_claims_require_independent_factual_review()
    print("leader_parity_sampling_ok")


if __name__ == "__main__":
    main()
