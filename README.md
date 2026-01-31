# Photofinder
Wedding Photo Face Finder

A private, shareable web app that lets friends and family upload reference photos and receive a ZIP of wedding photos where they appear — without exposing the full photo library.

# Why this exists

Wedding photos live in a private library (~5,000 images)

Sharing folders gives too much access

This tool returns only the user’s own photos, nothing else

# High-level design

UI: Gradio app hosted on a free Hugging Face Space

Storage: Private Hugging Face dataset repo holds:

original wedding photos

precomputed face embeddings + index

# Flow:

User uploads 1–5 reference photos

App embeds faces and searches a prebuilt index

Matching originals are collected

A ZIP is generated and returned

# Privacy & access model

Users never see the full library

No Google Drive access or folder sharing

Only matched photos are downloadable

Results are generated per request and discarded

# Tech stack

Face embeddings: InsightFace (ArcFace)

Similarity search: FAISS (cosine similarity)

UI: Gradio

Hosting: Hugging Face Spaces (free CPU)

Language: Python
