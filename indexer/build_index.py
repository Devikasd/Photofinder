import os
import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

import insightface
import faiss

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

def iter_images(root: Path):
    """
    Recursively yield all image files under a directory.
    This allows us to preserve any existing folder structure.
    """
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            yield p

def read_image_bgr(path: Path):
    """
    Read image using OpenCV.
    OpenCV loads images in BGR format by default (which InsightFace expects).
    """
    return cv2.imread(str(path))

def main():
    # ---- CONFIG (temporary, local testing) ----
    PHOTOS_ROOT = Path("sample_photos")   # create this folder with ~10–20 images
    OUT_DIR = Path("output/index")
    DET_SIZE = (640, 640)
    # -------------------------------------------

    if not PHOTOS_ROOT.exists():
        raise RuntimeError(f"{PHOTOS_ROOT} does not exist")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load a pretrained InsightFace model (ArcFace-based).

    print("Loading face model…")
    app = insightface.app.FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=0, det_size=DET_SIZE)

    embeddings = [] #A list of 512-dim float vectors (one per detected face).
    manifest = [] #A parallel list mapping embedding index -> photo metadata.

    images = list(iter_images(PHOTOS_ROOT))
    print(f"Images found: {len(images)}")

    for img_path in tqdm(images, desc="Indexing faces"):
        img = read_image_bgr(img_path)
        if img is None:
            continue

        # Detect faces + compute embeddings.

        faces = app.get(img)
        if not faces:
            continue
        
        # Store paths relative to PHOTOS_ROOT so the dataset

        rel_path = str(img_path.relative_to(PHOTOS_ROOT)).replace("\\", "/")
        # InsightFace embeddings are already L2-normalized.
            # This is important because:
            #   cosine_similarity(a, b) == dot_product(a, b)

        for f in faces:
            if f.normed_embedding is None:
                continue

            emb = np.asarray(f.normed_embedding, dtype=np.float32)
            embeddings.append(emb)
            manifest.append({
                "photo_path": rel_path
            })

    if not embeddings:
        raise RuntimeError("No faces detected. Try different images.")
    
    # Stack into a single matrix of shape (num_faces, 512)

    X = np.vstack(embeddings).astype(np.float32)
    dim = X.shape[1]

    print(f"Total faces indexed: {len(X)}")

    index = faiss.IndexFlatIP(dim)
    index.add(X)

    faiss_path = OUT_DIR / "embeddings.faiss"
    manifest_path = OUT_DIR / "manifest.json"

    faiss.write_index(index, str(faiss_path))
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print("Index written to:", OUT_DIR)

if __name__ == "__main__":
    main()