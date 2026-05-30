"""End-to-end orchestrator.

Two modes:
  * mode="api"   (default) — MuAPI does download / transcribe / LLM / autocrop.
                              Fast, no local deps, pay-per-call.
  * mode="local"            — yt-dlp + faster-whisper + OpenAI or Gemini + ffmpeg/opencv.
                              Self-hosted, LLM_PROVIDER selects OpenAI or Gemini.
"""
from typing import Dict, List, Optional, Tuple, Union

from .clipper import crop_highlights
from .downloader import download_youtube
from .highlights import call_muapi_llm, get_highlights
from .transcriber import transcribe


ClipDurationInput = Union[None, float, int, str, Tuple[float, float]]
ClipDurationRange = Optional[Tuple[float, float]]


def _normalize_clip_duration(value: ClipDurationInput) -> ClipDurationRange:
    """Coerce CLI/python-API inputs into a (min, max) tuple.

    Accepts:
      - None                         → None
      - 30 / 30.0                    → (30.0, 30.0)
      - "30"                         → (30.0, 30.0)
      - "30-45"                      → (30.0, 45.0)
      - (30, 45) / [30, 45]          → (30.0, 45.0)
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        v = float(value)
        if v <= 0:
            raise ValueError(f"clip_duration must be > 0, got {value!r}")
        return (v, v)
    if isinstance(value, str):
        s = value.strip()
        if "-" in s:
            lo_str, hi_str = s.split("-", 1)
            lo, hi = float(lo_str), float(hi_str)
        else:
            lo = hi = float(s)
    elif isinstance(value, (tuple, list)) and len(value) == 2:
        lo, hi = float(value[0]), float(value[1])
    else:
        raise TypeError(
            f"clip_duration must be a number, '30-45' string, or (min,max) "
            f"tuple — got {value!r}"
        )
    if lo <= 0 or hi <= 0:
        raise ValueError(f"clip_duration values must be > 0, got {value!r}")
    if lo > hi:
        raise ValueError(f"clip_duration min must be <= max, got {value!r}")
    return (lo, hi)


def _run_local(
    youtube_url: str,
    num_clips: int,
    aspect_ratio: str,
    download_format: str,
    language: Optional[str],
    clip_duration: ClipDurationRange,
) -> Dict:
    from .local.clipper import crop_highlights_local
    from .local.downloader import download_youtube_local
    from .local.llm import call_local_llm
    from .local.transcriber import transcribe_local

    source_path = download_youtube_local(youtube_url, fmt=download_format)

    transcript = transcribe_local(source_path, language=language)
    if not transcript["segments"]:
        raise RuntimeError(
            "Whisper produced no segments. The video may have no detectable speech."
        )

    highlights_result = get_highlights(
        transcript,
        num_clips=num_clips,
        llm_fn=call_local_llm,
        clip_duration=clip_duration,
    )
    all_highlights: List[Dict] = highlights_result.get("highlights", [])
    if not all_highlights:
        raise RuntimeError("Highlight generator returned zero clips.")

    top = sorted(all_highlights, key=lambda h: int(h.get("score", 0)), reverse=True)[:num_clips]
    print(f"[pipeline/local] cropping {len(top)} of {len(all_highlights)} candidates", flush=True)

    shorts = crop_highlights_local(source_path, top, aspect_ratio=aspect_ratio)

    return {
        "mode": "local",
        "source_video_url": source_path,
        "transcript": transcript,
        "highlights": all_highlights,
        "shorts": shorts,
    }


def _run_api(
    youtube_url: str,
    num_clips: int,
    aspect_ratio: str,
    download_format: str,
    language: Optional[str],
    clip_duration: ClipDurationRange,
) -> Dict:
    source_url = download_youtube(youtube_url, fmt=download_format)

    transcript = transcribe(source_url, language=language)
    if not transcript["segments"]:
        raise RuntimeError(
            "Whisper produced no segments. The video may have no detectable speech."
        )

    highlights_result = get_highlights(
        transcript,
        num_clips=num_clips,
        llm_fn=call_muapi_llm,
        clip_duration=clip_duration,
    )
    all_highlights: List[Dict] = highlights_result.get("highlights", [])
    if not all_highlights:
        raise RuntimeError("Highlight generator returned zero clips.")

    top = sorted(all_highlights, key=lambda h: int(h.get("score", 0)), reverse=True)[:num_clips]
    print(f"[pipeline] cropping {len(top)} of {len(all_highlights)} candidates", flush=True)

    shorts = crop_highlights(source_url, top, aspect_ratio=aspect_ratio)

    return {
        "mode": "api",
        "source_video_url": source_url,
        "transcript": transcript,
        "highlights": all_highlights,
        "shorts": shorts,
    }


def generate_shorts(
    youtube_url: str,
    num_clips: int = 3,
    aspect_ratio: str = "9:16",
    download_format: str = "720",
    language: Optional[str] = None,
    mode: str = "api",
    clip_duration: ClipDurationInput = None,
) -> Dict:
    """Run the full pipeline and return a structured result.

    Args:
        youtube_url: source URL.
        num_clips: how many shorts to render.
        aspect_ratio: e.g. "9:16", "1:1".
        download_format: source resolution ("360" / "480" / "720" / "1080").
        language: ISO-639-1 to force Whisper language detection.
        mode: "api" (default, MuAPI) or "local" (yt-dlp + faster-whisper +
            OpenAI or Gemini + ffmpeg).
        clip_duration: target clip length. Pass `30` (or `(30, 30)`) for a
            fixed length, or `(30, 45)` / `"30-45"` for a range — clips
            inside the range are kept as-is and clips outside are clamped to
            the nearest edge. Leave as None for the default 45-90s sweet spot.

    Returns:
        {
          "mode": "api" | "local",
          "source_video_url": str,   # hosted URL (api) or local path (local)
          "transcript": {...},
          "highlights": [...],       # all candidates ranked
          "shorts": [...],           # top `num_clips` with clip_url / local path
        }
    """
    clip_range = _normalize_clip_duration(clip_duration)
    mode = (mode or "api").lower()
    if mode == "local":
        return _run_local(youtube_url, num_clips, aspect_ratio, download_format, language, clip_range)
    if mode == "api":
        return _run_api(youtube_url, num_clips, aspect_ratio, download_format, language, clip_range)
    raise ValueError(f"Unknown mode: {mode!r}. Use 'api' or 'local'.")
