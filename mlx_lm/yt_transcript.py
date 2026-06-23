"""YouTube transcript extractor."""

import re
from typing import Optional

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter


_API = YouTubeTranscriptApi()


def get_transcript(video_id: str, languages: Optional[list[str]] = None) -> Optional[str]:
    """Get the transcript of a YouTube video as plain text."""
    vid = _extract_id(video_id)
    if not vid:
        return None
    try:
        transcript = _API.fetch(vid, languages=languages or ["en"])
    except Exception:
        return None
    formatter = TextFormatter()
    return formatter.format_transcript(transcript)


def get_transcript_with_timestamps(video_id: str, languages: Optional[list[str]] = None) -> Optional[str]:
    """Get transcript with timestamps like [0:15] text."""
    vid = _extract_id(video_id)
    if not vid:
        return None
    try:
        transcript = _API.fetch(vid, languages=languages or ["en"])
    except Exception:
        return None
    lines = []
    for entry in transcript:
        secs = int(entry.duration)
        mins, secs = divmod(secs, 60)
        ts = f"[{mins}:{secs:02d}]"
        lines.append(f"{ts} {entry.text}")
    return "\n".join(lines)


def _extract_id(video_id: str) -> Optional[str]:
    """Extract YouTube video ID from URL or return as-is if already an ID."""
    m = re.search(r"(?:v=|youtu\.be/|/shorts/|/embed/|v/|watch\?v=)([a-zA-Z0-9_-]{11})", video_id)
    if m:
        return m.group(1)
    if re.match(r"^[a-zA-Z0-9_-]{11}$", video_id):
        return video_id
    return None
