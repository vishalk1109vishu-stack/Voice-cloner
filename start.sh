#!/bin/bash
# Voice Cloning Studio v2 — Linux / macOS launcher
cd "$(dirname "$0")"
if [ -d "venv" ]; then
    source venv/bin/activate
fi
python app.py
