#!/bin/bash
set -e

echo "=== STL Slicer ==="

# Install deps if needed
pip install flask pypdf werkzeug trimesh scipy networkx shapely mapbox-earcut manifold3d --quiet

# Open browser after short delay
(sleep 2 && python3 -c "import webbrowser; webbrowser.open('http://localhost:5000/slicer')") &

# Start server
python3 app.py
