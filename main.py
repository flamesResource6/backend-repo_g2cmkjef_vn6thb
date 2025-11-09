import os
import uuid
import shutil
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from database import db, create_document
from schemas import Job, Asset

# Optional imports made safe so startup never crashes if a lib is missing
import numpy as np

try:
    import cv2  # type: ignore
    HAVE_CV2 = True
except Exception:
    cv2 = None  # type: ignore
    HAVE_CV2 = False

try:
    from PIL import Image, ImageDraw, ImageFont
    HAVE_PIL = True
except Exception:
    Image = ImageDraw = ImageFont = None  # type: ignore
    HAVE_PIL = False

try:
    from anthropic import Anthropic
except Exception:
    Anthropic = None

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Use a local, writable data directory in the project workspace
BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, "data")
ASSETS_DIR = os.path.join(DATA_DIR, "assets")
VIDEOS_DIR = os.path.join(DATA_DIR, "videos")
for d in [DATA_DIR, ASSETS_DIR, VIDEOS_DIR]:
    os.makedirs(d, exist_ok=True)


class CreateJobResponse(BaseModel):
    job_id: str
    status: str
    job: Optional[dict] = None


@app.get("/")
async def root():
    return {"message": "AI Gaming Video Editor API"}


@app.post("/api/jobs", response_model=CreateJobResponse)
async def create_job(
    mode: str = Form("full"),
    target_duration_sec: int = Form(900),
    output_format: str = Form("16:9"),
    files: List[UploadFile] = File(...),
):
    # Save files to disk
    saved_paths: List[str] = []
    for f in files:
        file_id = f"{uuid.uuid4()}_{f.filename}"
        dest = os.path.join(VIDEOS_DIR, file_id)
        with open(dest, "wb") as out:
            out.write(await f.read())
        saved_paths.append(dest)

    # Initialize a job doc in DB
    job = Job(
        mode=mode,
        target_duration_sec=int(target_duration_sec),
        output_format=output_format,
        status="processing",
        progress=0.05,
        source_files=saved_paths,
    )
    # Note: create_document requires a configured DB; environment in this sandbox
    # provides MongoDB. If not, it would raise, but startup remains fine.
    job_id = create_document("job", job)

    # Run lightweight AI pipeline synchronously for demo purposes
    recognized_game = recognize_game(saved_paths)
    _ = detect_highlights(saved_paths)

    # Generate 6 variants
    results = []
    for i in range(6):
        out_path = os.path.join(ASSETS_DIR, f"{job_id}_v{i+1}.mp4")
        synthesize_video(saved_paths, out_path, variant=i)
        cover_path = generate_cover_image(job_id, i)
        results.append({
            "video_url": f"/api/assets/{os.path.basename(out_path)}",
            "cover_url": f"/api/assets/{os.path.basename(cover_path)}",
            "meta": {"variant": i + 1}
        })

    # Update job document
    try:
        from bson import ObjectId  # provided by pymongo
        oid = ObjectId(job_id)
        db["job"].update_one(
            {"_id": oid},
            {"$set": {
                "recognized_game": recognized_game,
                "results": results,
                "status": "done",
                "progress": 1.0,
            }}
        )
        doc = db["job"].find_one({"_id": oid})
    except Exception:
        db["job"].update_one(
            {"_id": job_id},
            {"$set": {
                "recognized_game": recognized_game,
                "results": results,
                "status": "done",
                "progress": 1.0,
            }}
        )
        doc = db["job"].find_one({"_id": job_id})

    # Store assets records
    for res in results:
        create_document("asset", Asset(job_id=job_id, kind="video", url=res["video_url"]).model_dump())
        create_document("asset", Asset(job_id=job_id, kind="cover", url=res["cover_url"]).model_dump())

    if doc and "_id" in doc:
        doc["_id"] = str(doc["_id"])

    return CreateJobResponse(job_id=job_id, status="done", job=doc)


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    try:
        from bson import ObjectId
        doc = db["job"].find_one({"_id": ObjectId(job_id)})
    except Exception:
        doc = db["job"].find_one({"_id": job_id})
    if not doc:
        return {"error": "job not found"}
    if "_id" in doc:
        doc["_id"] = str(doc["_id"])
    return doc


@app.get("/api/assets/{name}")
async def get_asset(name: str):
    path = os.path.join(ASSETS_DIR, name)
    if os.path.exists(path):
        if name.lower().endswith((".mp4", ".mov", ".mkv")):
            return FileResponse(path, media_type="video/mp4")
        elif name.lower().endswith((".png", ".jpg", ".jpeg")):
            return FileResponse(path, media_type="image/png")
    return {"error": "asset not found"}


# ---------------------
# Lightweight AI blocks
# ---------------------

def recognize_game(paths: List[str]) -> Optional[str]:
    """Heuristic + optional LLM classification of likely game name."""
    game_candidates = ["Brawl Stars", "Clash Royale", "Fortnite", "Valorant"]

    # Try to infer by filename hints
    for p in paths:
        name = os.path.basename(p).lower()
        for g in game_candidates:
            if g.lower().replace(" ", "") in name:
                return g

    # If Anthropic is available, ask it based on coarse prior
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if Anthropic and api_key:
        try:
            client = Anthropic(api_key=api_key)
            desc = "Guess the most likely game among: " + ", ".join(game_candidates)
            msg = client.messages.create(
                model="claude-3-5-sonnet-latest",
                max_tokens=64,
                messages=[{"role": "user", "content": desc}],
            )
            text = msg.content[0].text if getattr(msg, 'content', None) else None
            for g in game_candidates:
                if text and g.lower() in text.lower():
                    return g
        except Exception:
            pass
    return None


def detect_highlights(paths: List[str]) -> List[dict]:
    """Intensity-based highlight detection using frame diffs as a proxy for action peaks."""
    if not HAVE_CV2:
        return []
    peaks = []
    for path in paths:
        cap = cv2.VideoCapture(path)  # type: ignore
        if not cap.isOpened():
            continue
        prev = None
        scores = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # type: ignore
            gray = cv2.resize(gray, (160, 90))  # type: ignore
            if prev is not None:
                diff = cv2.absdiff(gray, prev)  # type: ignore
                score = float(np.mean(diff))
                scores.append(score)
            prev = gray
        cap.release()
        if scores:
            arr = np.array(scores)
            thr = float(np.percentile(arr, 95))
            for i, s in enumerate(scores):
                if s >= thr:
                    peaks.append({"source": path, "frame_index": i, "intensity": s})
    return peaks[:50]


def synthesize_video(sources: List[str], out_path: str, variant: int = 0) -> None:
    """Create an output video. If OpenCV available, stitch; otherwise copy source."""
    if HAVE_CV2 and len(sources) > 0:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore
        fps = 24
        size = (640, 360)
        writer = cv2.VideoWriter(out_path, fourcc, fps, size)  # type: ignore
        for src in sources:
            cap = cv2.VideoCapture(src)  # type: ignore
            if not cap.isOpened():
                continue
            frames_to_take = 60 + 10 * variant
            count = 0
            while count < frames_to_take:
                ok, frame = cap.read()
                if not ok:
                    break
                frame = cv2.resize(frame, size)  # type: ignore
                if variant % 3 == 1:
                    frame = cv2.GaussianBlur(frame, (5, 5), 0)  # type: ignore
                if variant % 3 == 2:
                    overlay = frame.copy()
                    # light border overlay
                    if frame.shape[1] > 40 and frame.shape[0] > 40:
                        overlay[:, :40] = (0, 255, 180)
                        overlay[:, -40:] = (0, 255, 180)
                        overlay[:40, :] = (0, 255, 180)
                        overlay[-40:, :] = (0, 255, 180)
                        alpha = 0.08
                        frame = np.clip(overlay * alpha + frame * (1 - alpha), 0, 255).astype(frame.dtype)
                writer.write(frame)  # type: ignore
                count += 1
            cap.release()
        writer.release()  # type: ignore
    else:
        # Fallback: copy the first source as the variant output if available
        if sources:
            src = sources[0]
            shutil.copyfile(src, out_path)
        else:
            # create an empty file to avoid 404
            with open(out_path, "wb") as f:
                f.write(b"")


def generate_cover_image(job_id: str, variant: int) -> str:
    """Generate a synthetic cover image using PIL if available, else NumPy+OpenCV fallback."""
    path = os.path.join(ASSETS_DIR, f"{job_id}_cover_{variant+1}.png")
    w, h = 1280, 720
    if HAVE_PIL:
        img = Image.new("RGB", (w, h))
        dr = ImageDraw.Draw(img)
        for y in range(h):
            alpha = y / h
            c = (
                int((30 + 20 * variant) * (1 - alpha) + 60 * alpha) % 255,
                int(180 * (1 - alpha) + (120 + 25 * variant) * alpha) % 255,
                int(120 * (1 - alpha) + 220 * alpha) % 255,
            )
            dr.line([(0, y), (w, y)], fill=c)
        dr.rectangle([40, 40, w - 40, h - 40], outline=(10, 10, 10), width=6)
        dr.text((60, 120), f"Versione {variant+1}", fill=(20, 20, 20))
        dr.text((60, 200), "AI Gaming Edit", fill=(10, 10, 10))
        img.save(path)
        return path
    else:
        img = np.zeros((h, w, 3), dtype=np.uint8)
        color1 = (int(30 + 20 * variant) % 255, 180, 120)
        color2 = (60, int(120 + 25 * variant) % 255, 220)
        for y in range(h):
            alpha = y / h
            c = (
                int(color1[0] * (1 - alpha) + color2[0] * alpha),
                int(color1[1] * (1 - alpha) + color2[1] * alpha),
                int(color1[2] * (1 - alpha) + color2[2] * alpha),
            )
            img[y, :] = c
        if HAVE_CV2:
            cv2.imwrite(path, img)  # type: ignore
            return path
        with open(path, "wb") as f:
            f.write(b"")
        return path


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
