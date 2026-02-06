import json
from pathlib import Path

import cv2
import numpy as np
import faiss
import insightface

def read_image_bgr(path: Path):
    return cv2.imread(str(path))

def main():
    # ---- CONFIG ----
    PHOTOS_ROOT = Path("sample_photos")          # same folder you indexed
    INDEX_DIR   = Path("output/index")           # created by build_index.py
    REF_IMAGE   = Path("ref.jpg")                # put a reference face here
    TOP_K       = 20
    DET_SIZE    = (640, 640)
    # --------------

    faiss_path = INDEX_DIR / "embeddings.faiss"
    manifest_path = INDEX_DIR / "manifest.json"

    if not faiss_path.exists() or not manifest_path.exists():
        raise RuntimeError("Index files not found. Run build_index.py first.")
    if not REF_IMAGE.exists():
        raise RuntimeError("ref.jpg not found. Put a clear selfie at indexer/ref.jpg")

    print("Loading face model…")
    app = insightface.app.FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=0, det_size=DET_SIZE)

    print("Loading index…")
    index = faiss.read_index(str(faiss_path))
    manifest = json.load(open(manifest_path, "r"))

    # Embed the reference photo (use the first detected face)
    ref_img = read_image_bgr(REF_IMAGE)
    if ref_img is None:
        raise RuntimeError("Could not read ref.jpg")

    faces = app.get(ref_img)
    if not faces or faces[0].normed_embedding is None:
        raise RuntimeError("No face found in ref.jpg. Use a clearer face image.")

    q = np.asarray(faces[0].normed_embedding, dtype=np.float32).reshape(1, -1)

    # Search FAISS (inner product == cosine similarity because embeddings are normalized)
    D, I = index.search(q, TOP_K)

    print(f"\nTop {TOP_K} face matches:")
    for rank, (sim, idx) in enumerate(zip(D[0], I[0]), start=1):
        if idx < 0:
            continue
        photo_rel = manifest[idx]["photo_path"]
        print(f"{rank:02d}. sim={sim:.3f}  photo={photo_rel}")

if __name__ == "__main__":
    main()