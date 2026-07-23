# ai-tools-
"""
Clipping tool backend - MVP

Flow:
  1. POST /process with {"youtube_url": "..."}
  2. Download the video with yt-dlp
  3. Transcribe it with faster-whisper (runs locally, free, no API key)
  4. Score segments with highlight_detector to find the best moments
  5. Cut each highlight with ffmpeg, reframe to vertical (9:16), burn in
     captions from the whisper transcript
  6. Return a list of clip file paths (served via /clips/<filename>)

Everything here is free / open-source: yt-dlp, faster-whisper, ffmpeg.
No paid API keys required. This is an MVP meant to run on one server
(e.g. a free Render/Railway instance) - it processes one video at a time.
"""

import os
import re
import uuid
import subprocess

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from faster_whisper import WhisperModel

from highlight_detector import find_highlight_windows

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOWNLOAD_DIR = os.path.join(BASE_DIR, "downloads")
CLIPS_DIR = os.path.join(BASE_DIR, "clips")
os.makedirs(DOWNLOAD_DIR, exist_ok=True)
os.makedirs(CLIPS_DIR, exist_ok=True)

app = Flask(__name__)
CORS(app)  # allow requests from your frontend domain

# "base" is the smallest good-quality model; "small"/"medium" are better
# but slower and need more RAM. Start with "base" on a free server.
print("Loading Whisper model (first run downloads it, then it's cached)...")
whisper_model = WhisperModel("base", device="cpu", compute_type="int8")


def download_video(youtube_url: str) -> str:
    """Downloads video+audio muxed as mp4, returns local file path."""
    out_id = str(uuid.uuid4())
    out_path = os.path.join(DOWNLOAD_DIR, f"{out_id}.mp4")
    cmd = [
        "yt-dlp",
        "-f", "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "--merge-output-format", "mp4",
        "-o", out_path,
        youtube_url,
    ]
    subprocess.run(cmd, check=True)
    return out_path


def transcribe(video_path: str):
    """Returns list of {"start", "end", "text"} segments."""
    segments_out = []
    segments, _info = whisper_model.transcribe(video_path, beam_size=5)
    for seg in segments:
        segments_out.append({
            "start": seg.start,
            "end": seg.end,
            "text": seg.text.strip(),
            "words": seg.words if hasattr(seg, "words") else None,
        })
    return segments_out


def make_srt(segments, start_offset, end_offset, srt_path):
    """Writes an SRT file containing only the segments inside [start_offset, end_offset],
    with timestamps shifted so 0 = start_offset (matches the cut clip)."""
    def fmt(t):
        h = int(t // 3600)
        m = int((t % 3600) // 60)
        s = t % 60
        return f"{h:02}:{m:02}:{s:06.3f}".replace(".", ",")

    lines = []
    idx = 1
    for seg in segments:
        if seg["end"] < start_offset or seg["start"] > end_offset:
            continue
        s = max(seg["start"], start_offset) - start_offset
        e = min(seg["end"], end_offset) - start_offset
        if e <= s:
            continue
        lines.append(str(idx))
        lines.append(f"{fmt(s)} --> {fmt(e)}")
        lines.append(seg["text"])
        lines.append("")
        idx += 1

    with open(srt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def cut_and_caption(video_path, start, end, segments, out_path):
    """Cuts [start, end], reframes to vertical 9:16 with blurred background,
    and burns in captions from the transcript."""
    srt_path = out_path.replace(".mp4", ".srt")
    make_srt(segments, start, end, srt_path)

    duration = end - start

    # Vertical reframe: blurred full-frame background + centered sharp video on top,
    # then burn subtitles. This is a classic free ffmpeg recipe for shorts/reels.
    filter_complex = (
        "[0:v]scale=1080:1920:force_original_aspect_ratio=increase,"
        "crop=1080:1920,boxblur=20:5[bg];"
        "[0:v]scale=1080:-2[fg];"
        "[bg][fg]overlay=(W-w)/2:(H-h)/2[merged];"
        f"[merged]subtitles={srt_path}:force_style="
        "'FontSize=16,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BorderStyle=3'"
        "[final]"
    )

    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start), "-t", str(duration),
        "-i", video_path,
        "-filter_complex", filter_complex,
        "-map", "[final]", "-map", "0:a?",
        "-c:v", "libx264", "-c:a", "aac",
        out_path,
    ]
    subprocess.run(cmd, check=True)


@app.route("/process", methods=["POST"])
def process():
    data = request.get_json(force=True)
    youtube_url = data.get("youtube_url")
    top_n = int(data.get("num_clips", 5))
    clip_length = int(data.get("clip_length", 45))

    if not youtube_url:
        return jsonify({"error": "youtube_url is required"}), 400

    try:
        video_path = download_video(youtube_url)
        segments = transcribe(video_path)
        highlights = find_highlight_windows(
            segments, clip_length=clip_length, top_n=top_n
        )

        clip_urls = []
        for i, h in enumerate(highlights):
            clip_id = str(uuid.uuid4())
            out_path = os.path.join(CLIPS_DIR, f"{clip_id}.mp4")
            cut_and_caption(video_path, h["start"], h["end"], segments, out_path)
            clip_urls.append({
                "url": f"/clips/{clip_id}.mp4",
                "start": h["start"],
                "end": h["end"],
                "score": h["score"],
                "preview_text": h["text"],
            })

        return jsonify({"clips": clip_urls})

    except subprocess.CalledProcessError as e:
        return jsonify({"error": f"Processing failed: {e}"}), 500


@app.route("/clips/<path:filename>")
def serve_clip(filename):
    return send_from_directory(CLIPS_DIR, filename)


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
