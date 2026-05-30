"""Find the most viral-worthy highlights in a transcript.

Logic ported from ViralVadoo's transcript_analysis/highlight_generator.py:
  - content-type / density detection
  - chunking for long videos with overlap
  - virality-criteria prompt
  - score-based dedupe with overlap suppression

The LLM call is pluggable via the `llm_fn` argument so the same prompts can
drive either MuAPI (default, --mode api) or a direct local LLM client
(--mode local).
"""
import json
import re
from typing import Callable, Dict, List, Optional, Tuple

from . import muapi


LLMFn = Callable[[str], str]
ClipDurationRange = Optional[Tuple[float, float]]


CONTENT_TYPE_PROMPT = """Analyze this video transcript sample and classify the content type.
Choose one: podcast, interview, tutorial, lecture, commentary, debate, vlog, other.
Also estimate content density: low (mostly filler/chit-chat), medium, or high (dense info/stories).
Respond with JSON only: {"content_type": "...", "density": "..."}"""


VIRALITY_CRITERIA = """
Virality signals to prioritize (ranked by impact):
1. HOOK MOMENTS — statements that create immediate curiosity ("The secret is...", "Nobody talks about...", "I was completely wrong about...")
2. EMOTIONAL PEAKS — genuine surprise, laughter, anger, vulnerability, excitement; raw unscripted reactions
3. OPINION BOMBS — strong, polarizing or counter-intuitive statements that trigger agree/disagree
4. REVELATION MOMENTS — surprising facts, stats, or confessions that reframe how the viewer thinks
5. CONFLICT/TENSION — disagreement, pushback, or a problem being confronted head-on
6. QUOTABLE ONE-LINERS — a sentence that works as a standalone quote card
7. STORY PEAKS — the climax or twist of an anecdote; the payoff moment
8. PRACTICAL VALUE — a concrete tip, hack, or insight the viewer can immediately apply
"""


HIGHLIGHT_SYSTEM_PROMPT = """You are an elite short-form video editor who has studied thousands of viral clips on TikTok, Instagram Reels, and YouTube Shorts. You know exactly what makes viewers stop scrolling, watch to the end, and share.

{virality_criteria}

Content type: {content_type} | Density: {density}

Your task: identify the most viral-worthy highlights from the transcript.

Rules:
- Every highlight must open with a strong HOOK — a line that grabs attention within the first 3 seconds
- {duration_rules}
- Never cut mid-sentence or mid-thought — each clip must feel complete and self-contained
- Clips must not overlap significantly with each other
- Score 0-100 on viral potential (not general quality)
- {num_clips_instruction}
- For each highlight, identify the single best "hook_sentence" — the opening line that would make someone stop scrolling
- Explain in one sentence why this clip is viral ("virality_reason")

Respond ONLY with valid JSON (no markdown, no explanation):
{{"highlights":[{{"title":"string","start_time":float,"end_time":float,"score":int,"hook_sentence":"string","virality_reason":"string"}}]}}"""


DEFAULT_DURATION_RULES = (
    "Duration sweet spot: 45-90 seconds. Go shorter (20-44s) only for a perfect "
    "standalone one-liner. Go longer (91-180s) only when a story arc needs full "
    "context to land"
)


def _duration_rules(clip_duration: ClipDurationRange) -> str:
    """Build the duration-rules paragraph injected into the highlight prompt.

    `clip_duration` is None or a (min, max) tuple. When set we also clamp
    clips to the range after the LLM step — but it still helps to nudge the
    LLM to pick start points that read well at the target length.
    """
    if not clip_duration:
        return DEFAULT_DURATION_RULES
    lo, hi = clip_duration
    if lo == hi:
        target = lo
        return (
            f"Target duration: exactly {target:.0f} seconds per clip. "
            f"Each clip will be trimmed to {target:.0f} seconds afterwards, "
            f"so pick a start_time where the hook lands in the first 3 "
            f"seconds and the next {target:.0f} seconds stay self-contained"
        )
    return (
        f"Target duration: {lo:.0f}-{hi:.0f} seconds per clip. Each clip "
        f"will be clamped to fall inside {lo:.0f}-{hi:.0f} seconds afterwards "
        f"(in-range picks are kept as-is, shorter clips are extended to "
        f"{lo:.0f}s, longer clips are trimmed to {hi:.0f}s), so pick a "
        f"start_time where the hook lands in the first 3 seconds and the "
        f"next {lo:.0f}-{hi:.0f} seconds stay self-contained"
    )


CHUNK_SIZE_SECONDS = 1200       # 20-min chunks for long videos
LONG_VIDEO_THRESHOLD = 1800     # chunk videos longer than 30 min
CHUNK_OVERLAP_SECONDS = 60
GPT_CALL_TIMEOUT_SECONDS = 300  # cap LLM polls at 5 min — a wedged call should fail fast


def call_muapi_llm(prompt: str) -> str:
    """Default LLM backend: MuAPI gpt-5-mini."""
    result = muapi.run(
        "gpt-5-mini",
        {"prompt": prompt},
        label="gpt-5-mini",
        timeout=GPT_CALL_TIMEOUT_SECONDS,
    )

    outputs = result.get("outputs")
    if isinstance(outputs, list) and outputs and isinstance(outputs[0], str) and outputs[0].strip():
        return outputs[0]

    for key in ("output", "text", "response", "result", "content"):
        v = result.get(key)
        if isinstance(v, str) and v.strip():
            return v
        if isinstance(v, dict):
            inner = v.get("text") or v.get("content")
            if isinstance(inner, str) and inner.strip():
                return inner
        if isinstance(v, list) and v and isinstance(v[0], str):
            return v[0]

    raise RuntimeError(f"Could not extract gpt-5-mini text from response: {result}")


def _parse_json_loose(raw: str) -> Dict:
    """gpt-5-4 sometimes wraps JSON in markdown fences — strip and parse."""
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            return json.loads(text[start:end + 1])
        raise


def detect_content_type(transcript: Dict, llm_fn: LLMFn = call_muapi_llm) -> Dict[str, str]:
    segments = transcript.get("segments", [])
    sample = " ".join(s["text"] for s in segments[:25])[:3000]
    prompt = f"{CONTENT_TYPE_PROMPT}\n\nTranscript sample:\n{sample}"
    try:
        raw = llm_fn(prompt)
        return _parse_json_loose(raw)
    except Exception:
        return {"content_type": "other", "density": "medium"}


def build_transcript_text(transcript: Dict) -> str:
    segments = transcript.get("segments", [])
    return "\n".join(f"[{s['start']:.1f}s] {s['text'].strip()}" for s in segments)


def chunk_transcript(transcript: Dict) -> List[Dict]:
    segments = transcript.get("segments", [])
    duration = transcript.get("duration", segments[-1]["end"] if segments else 0)
    chunks = []
    start = 0
    while start < duration:
        end = min(start + CHUNK_SIZE_SECONDS, duration)
        chunk_segs = [
            s for s in segments
            if s["start"] >= start and s["end"] <= end + CHUNK_OVERLAP_SECONDS
        ]
        if chunk_segs:
            chunk = dict(transcript)
            chunk["segments"] = chunk_segs
            chunk["duration"] = end - start
            chunk["_offset"] = start
            chunks.append(chunk)
        start += CHUNK_SIZE_SECONDS - CHUNK_OVERLAP_SECONDS
    return chunks


def call_highlight_api(
    transcript_text: str,
    content_info: Dict,
    duration: float,
    num_clips: int,
    is_chunk: bool = False,
    llm_fn: LLMFn = call_muapi_llm,
    clip_duration: ClipDurationRange = None,
) -> Dict:
    # Ask for ~2× the user's target so dedupe has headroom, but cap so the model
    # doesn't have to generate a huge JSON payload (which times out gpt-5-mini).
    target = max(num_clips * 2, 5)
    natural_max = max(2 if is_chunk else 3, int(duration / 90))
    min_clips = min(target, natural_max, 8)
    system = HIGHLIGHT_SYSTEM_PROMPT.format(
        virality_criteria=VIRALITY_CRITERIA,
        content_type=content_info.get("content_type", "other"),
        density=content_info.get("density", "medium"),
        num_clips_instruction=f"Generate at least {min_clips} highlights",
        duration_rules=_duration_rules(clip_duration),
    )
    full_prompt = f"{system}\n\nTranscript:\n{transcript_text}"
    raw = llm_fn(full_prompt)
    return _parse_json_loose(raw)


def dedupe_highlights(highlights: List[Dict]) -> List[Dict]:
    """Drop a highlight if it overlaps >50% with a higher-scoring one already kept."""
    highlights = sorted(highlights, key=lambda x: int(x.get("score", 0)), reverse=True)
    kept: List[Dict] = []
    for h in highlights:
        h_start = float(h["start_time"])
        h_end = float(h["end_time"])
        h_dur = h_end - h_start
        overlapping = False
        for k in kept:
            latest_start = max(h_start, float(k["start_time"]))
            earliest_end = min(h_end, float(k["end_time"]))
            overlap = earliest_end - latest_start
            if overlap > 0 and overlap > 0.5 * h_dur:
                overlapping = True
                break
        if not overlapping:
            kept.append(h)
    return kept


def snap_highlights_to_duration(
    highlights: List[Dict],
    clip_duration: ClipDurationRange,
    transcript_duration: Optional[float] = None,
) -> List[Dict]:
    """Clamp every highlight's length into the requested range.

    `clip_duration` is None (no-op) or a (min, max) tuple in seconds:
      - In-range LLM picks are left untouched.
      - Highlights shorter than `min` are extended (anchored at start_time)
        to exactly `min`.
      - Highlights longer than `max` are trimmed (anchored at start_time)
        to exactly `max`.
      - If the resulting window runs past the source video end, it shifts
        backward so it ends at `transcript_duration` while keeping a length
        inside [min, max] when possible.

    Passing a fixed length (e.g. (30, 30)) reproduces the old behavior of
    forcing every short to exactly 30 seconds.
    """
    if not clip_duration:
        return highlights
    lo, hi = clip_duration
    snapped: List[Dict] = []
    for h in highlights:
        start = float(h["start_time"])
        end = float(h["end_time"])
        dur = end - start

        if dur < lo:
            end = start + lo
        elif dur > hi:
            end = start + hi
        # else: in range — keep the LLM's pick as-is.

        # Keep the window inside the source video.
        if transcript_duration is not None and end > transcript_duration:
            end = float(transcript_duration)
            target_dur = max(lo, min(hi, end - start))
            start = max(0.0, end - target_dur)
        if start < 0:
            start = 0.0
            end = start + lo

        snapped.append({**h, "start_time": start, "end_time": end})
    return snapped


def get_highlights(
    transcript: Dict,
    num_clips: int = 3,
    llm_fn: Optional[LLMFn] = None,
    clip_duration: ClipDurationRange = None,
) -> Dict:
    """Main entry point — returns {highlights: [...]} sorted by score.

    `llm_fn` swaps the underlying LLM. Defaults to MuAPI gpt-5-mini; local
    mode passes in a local LLM-backed callable.

    `clip_duration` is None or a (min_seconds, max_seconds) tuple. When set,
    we both nudge the LLM toward that range and clamp every returned
    highlight into it (in-range picks pass through untouched).
    """
    llm_fn = llm_fn or call_muapi_llm
    duration = transcript.get("duration", 0)
    content_info = detect_content_type(transcript, llm_fn=llm_fn)
    print(f"[highlights] content={content_info.get('content_type')} density={content_info.get('density')} duration={duration:.0f}s", flush=True)
    if clip_duration:
        lo, hi = clip_duration
        label = f"{lo:.0f}s" if lo == hi else f"{lo:.0f}-{hi:.0f}s"
        print(f"[highlights] target clip duration: {label} (will clamp after LLM)", flush=True)

    if duration >= LONG_VIDEO_THRESHOLD:
        chunks = chunk_transcript(transcript)
        print(f"[highlights] long video — splitting into {len(chunks)} chunks", flush=True)
        all_highlights: List[Dict] = []
        for i, chunk in enumerate(chunks):
            offset = chunk.get("_offset", 0)
            text = build_transcript_text(chunk)
            print(f"[highlights] chunk {i + 1}/{len(chunks)} (offset {offset:.0f}s)", flush=True)
            result = call_highlight_api(
                text,
                content_info,
                chunk["duration"],
                num_clips=num_clips,
                is_chunk=True,
                llm_fn=llm_fn,
                clip_duration=clip_duration,
            )
            for h in result.get("highlights", []):
                h["start_time"] = float(h["start_time"]) + offset
                h["end_time"] = float(h["end_time"]) + offset
                all_highlights.append(h)
        highlights = dedupe_highlights(all_highlights)
    else:
        text = build_transcript_text(transcript)
        result = call_highlight_api(
            text,
            content_info,
            duration,
            num_clips=num_clips,
            llm_fn=llm_fn,
            clip_duration=clip_duration,
        )
        highlights = dedupe_highlights(result.get("highlights", []))

    if clip_duration:
        highlights = snap_highlights_to_duration(
            highlights, clip_duration, transcript_duration=duration or None
        )

    return {"highlights": highlights}
