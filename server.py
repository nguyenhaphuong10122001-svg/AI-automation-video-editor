"""An external FFmpeg renderer for n8n Cloud. Run as a private-key-protected web service.

POST /jobs         JSON storyboard -> asynchronous job
GET  /jobs/{id}    authenticated polling
GET  /videos/{id}/{name} unguessable output URL, no directory listings
"""
import base64
import hmac
import json
import os
import re
import subprocess
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

import requests
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from PIL import Image, ImageDraw, ImageFont

DATA = Path(os.environ.get("JOB_DIR", "/tmp/ffmpeg-video-jobs"))
DATA.mkdir(parents=True, exist_ok=True)
API_KEY = os.environ.get("OPENAI_API_KEY", "")
TOKEN = os.environ.get("VIDEO_RENDER_TOKEN", "")
DEMO = os.environ.get("DEMO_MODE", "false").lower() == "true"
MAX_WORKERS = 1
POOL = ThreadPoolExecutor(max_workers=MAX_WORKERS)
LOCK = threading.Lock()
JOBS: dict[str, dict] = {}
app = FastAPI(title="n8n Cloud external FFmpeg video engine", version="1.0.0")


class Scene(BaseModel):
    visual_prompt: str = Field(min_length=5, max_length=1800)
    narration: str = Field(min_length=4, max_length=1200)
    caption: str = Field(min_length=1, max_length=140)
    seconds: int = Field(ge=4, le=25)


class RenderRequest(BaseModel):
    title: str = Field(min_length=1, max_length=160)
    language: str = Field(default="ru", max_length=12)
    ratio: Literal["9:16", "16:9", "1:1"] = "9:16"
    target_seconds: int = Field(default=30, ge=12, le=75)
    scenes: list[Scene] = Field(min_length=2, max_length=6)


def requires_auth(request: Request):
    if not TOKEN:
        raise HTTPException(503, "Server VIDEO_RENDER_TOKEN not configured")
    if not hmac.compare_digest(request.headers.get("authorization", ""), f"Bearer {TOKEN}"):
        raise HTTPException(401, "Invalid renderer token")


def run_cmd(argv: list[str], timeout: int = 180):
    cp = subprocess.run(argv, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
    if cp.returncode:
        raise RuntimeError("FFmpeg error: " + cp.stderr[-1800:])
    return cp.stdout.strip()


def openai_post(endpoint: str, body: dict, timeout: int = 200) -> requests.Response:
    if not API_KEY:
        raise RuntimeError("OPENAI_API_KEY missing on render server")
    r = requests.post("https://api.openai.com/v1/" + endpoint,
                      headers={"Authorization": "Bearer " + API_KEY},
                      json=body, timeout=timeout)
    if not r.ok:
        raise RuntimeError(f"OpenAI {endpoint} returned {r.status_code}: {r.text[:650]}")
    return r


def draw_demo_image(path: Path, prompt: str, w: int, h: int):
    image = Image.new("RGB", (w, h), (31, 61, 95))
    d = ImageDraw.Draw(image)
    for y in range(h):
        d.line((0, y, w, y), fill=(30+int(30*y/h), 57+int(65*y/h), 100+int(30*y/h)))
    fontpath = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font = ImageFont.truetype(fontpath, 30)
    import textwrap
    text = "\n".join(textwrap.wrap(prompt, width=28)[:8])
    d.multiline_text((35, h//3), text, fill="white", font=font, spacing=15)
    image.save(path, quality=88)


def make_image(path: Path, prompt: str, ratio: str):
    size = {"9:16": "1024x1536", "16:9": "1536x1024", "1:1": "1024x1024"}[ratio]
    r = openai_post("images/generations", {
        "model": "gpt-image-1-mini", "prompt": "Realistic cinematic travel photography, no text, no letters, no watermark. " + prompt,
        "size": size, "quality": "low", "n": 1, "output_format": "jpeg"}, timeout=240)
    payload = r.json()
    b64 = (payload.get("data") or [{}])[0].get("b64_json")
    if not b64:
        raise RuntimeError("OpenAI image response did not include base64 image")
    path.write_bytes(base64.b64decode(b64))


def make_audio(path: Path, narration: str, language: str, seconds: int):
    if DEMO:
        # The test mode creates silent audio; production uses OpenAI TTS.
        run_cmd(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                 "anullsrc=r=44100:cl=stereo", "-t", str(seconds), "-q:a", "7", str(path)])
        return
    r = openai_post("audio/speech", {"model": "gpt-4o-mini-tts", "voice": "alloy",
                "response_format": "mp3", "input": narration,
                "instructions": f"Speak clearly and naturally in the language {language}; professional travel narration."}, timeout=180)
    path.write_bytes(r.content)


def probe_dur(path: Path) -> float:
    return float(run_cmd(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "default=noprint_wrappers=1:nokey=1", str(path)]))


def srt_timestamp(s: float) -> str:
    ms = int(s * 1000)
    return f"{ms//3600000:02d}:{ms//60000%60:02d}:{ms//1000%60:02d},{ms%1000:03d}"


def make_srt(path: Path, text: str, dur: float):
    # Sentence-level approximation, not word-aligned forced transcription.
    parts = [p.strip() for p in re.split(r"(?<=[.!?。！？])\s+", text.strip()) if p.strip()]
    if not parts:
        parts = [text.strip()]
    # Split extremely long sentences into word groups so subtitles stay readable.
    final = []
    for part in parts:
        words = part.split()
        if len(words) > 13:
            for k in range(0, len(words), 11):
                final.append(" ".join(words[k:k+11]))
        else:
            final.append(part)
    lines = []
    weights = [max(1,len(x)) for x in final]
    total = sum(weights)
    t = 0.0
    for i, (part, w) in enumerate(zip(final, weights), 1):
        finish = min(dur, t + dur * w/total) if i < len(final) else dur
        lines.append(f"{i}\n{srt_timestamp(t)} --> {srt_timestamp(finish)}\n{part}\n")
        t = finish
    path.write_text("\n".join(lines), encoding="utf-8")


def make_scene_video(image: Path, audio: Path, srt: Path, out: Path, seconds: float, ratio: str):
    w,h = {"9:16": (720,1280), "16:9": (1280,720), "1:1": (720,720)}[ratio]
    # Escape colon/backslash for ffmpeg filtergraph when path is generated internally.
    srt_path = str(srt).replace("\\", "\\\\").replace(":", "\\:")
    filt = (f"scale={w}:{h}:force_original_aspect_ratio=increase,"
            f"crop={w}:{h},"
            f"zoompan=z='min(zoom+0.00075,1.10)':d=1:s={w}x{h}:fps=24,"
            f"format=yuv420p,subtitles='{srt_path}':"
            f"force_style='FontName=DejaVu Sans,FontSize=24,Alignment=2,MarginV=46,"
            f"Outline=2,Shadow=0,BorderStyle=1'")
    run_cmd(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-threads", "2",
             "-loop", "1", "-framerate", "24", "-i", str(image), "-i", str(audio),
             "-vf", filt, "-af", "apad", "-t", str(round(seconds, 3)),
             "-c:v", "libx264", "-preset", "ultrafast", "-crf", "25", "-pix_fmt", "yuv420p",
             "-r", "24", "-c:a", "aac", "-ar", "44100", "-ac", "2", "-b:a", "128k",
             "-movflags", "+faststart", str(out)], timeout=240)


def work(job_id: str, request_body: RenderRequest, base_url: str):
    jobdir = DATA / job_id
    try:
        with LOCK:
            JOBS[job_id]["status"] = "running"
        jobdir.mkdir(parents=True, exist_ok=True)
        items = []
        for index, s in enumerate(request_body.scenes, 1):
            prefix = jobdir / f"scene_{index:02}"
            img, aud, srt, video = [prefix.with_suffix(e) for e in (".jpg", ".mp3", ".srt", ".mp4")]
            if DEMO:
                w,h = {"9:16": (720,1280), "16:9": (1280,720), "1:1": (720,720)}[request_body.ratio]
                draw_demo_image(img, s.visual_prompt, w, h)
            else:
                make_image(img, s.visual_prompt, request_body.ratio)
            make_audio(aud, s.narration, request_body.language, s.seconds)
            voice_len = probe_dur(aud)
            # Preserve narration; target duration remains approximate if TTS is long.
            scene_seconds = max(s.seconds, voice_len + 0.15)
            make_srt(srt, s.narration, scene_seconds)
            make_scene_video(img, aud, srt, video, scene_seconds, request_body.ratio)
            items.append(video)
            with LOCK:
                JOBS[job_id]["progress"] = f"{index}/{len(request_body.scenes)} scenes"
        concat = jobdir / "concat.txt"
        concat.write_text("".join(f"file '{p.name}'\n" for p in items), encoding="utf-8")
        filename = f"{uuid.uuid4().hex}.mp4"
        final = jobdir / filename
        run_cmd(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0",
                 "-i", str(concat), "-c", "copy", "-movflags", "+faststart", str(final)], timeout=180)
        actual_dur = round(probe_dur(final), 1)
        with LOCK:
            JOBS[job_id].update(status="succeeded", progress="ready", duration=actual_dur,
                                video_url=f"{base_url}/videos/{job_id}/{filename}")
    except Exception as ex:
        with LOCK:
            JOBS[job_id].update(status="failed", error=str(ex)[:1200])


@app.get("/health")
def health():
    return {"status": "ok", "ffmpeg": True, "demo_mode": DEMO, "has_openai_key": bool(API_KEY)}


@app.post("/jobs", status_code=202, dependencies=[Depends(requires_auth)])
def create_job(body: RenderRequest, request: Request):
    if not DEMO and not API_KEY:
        raise HTTPException(503, "Set OPENAI_API_KEY on renderer server")
    with LOCK:
        running = sum(s["status"] in ("queued", "running") for s in JOBS.values())
        if running >= 3:
            raise HTTPException(429, "Render queue full. Retry later.")
    job_id = uuid.uuid4().hex
    base = os.environ.get("PUBLIC_BASE_URL", "").rstrip("/") or str(request.base_url).rstrip("/")
    with LOCK:
        JOBS[job_id] = {"job_id": job_id, "title": body.title, "status": "queued", "progress": "waiting", "video_url": None, "created": time.time()}
    POOL.submit(work, job_id, body, base)
    return {"job_id": job_id, "status": "queued", "status_url": f"{base}/jobs/{job_id}"}


@app.get("/jobs/{job_id}", dependencies=[Depends(requires_auth)])
def status(job_id: str):
    with LOCK:
        value = JOBS.get(job_id)
        if value is None:
            raise HTTPException(404, "Job not found (may have expired after server restart)")
        return {k: v for k, v in value.items() if k != "created"}


@app.get("/videos/{job_id}/{filename}")
def download(job_id: str, filename: str):
    if not re.fullmatch(r"[0-9a-f]{32}", job_id) or not re.fullmatch(r"[0-9a-f]{32}\.mp4", filename):
        raise HTTPException(404, "Unknown file")
    with LOCK:
        value = JOBS.get(job_id)
        if not value or value["status"] != "succeeded" or not (value.get("video_url") or "").endswith("/" + filename):
            raise HTTPException(404, "Video not found")
    file_path = DATA / job_id / filename
    if not file_path.is_file():
        raise HTTPException(404, "Temporary video expired")
    return FileResponse(file_path, media_type="video/mp4", filename="generated_video.mp4")
