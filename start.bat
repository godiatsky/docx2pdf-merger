@echo off
echo === STL Slicer ===
echo Installing dependencies...
pip install flask pypdf werkzeug trimesh scipy networkx shapely mapbox-earcut manifold3d rtree --quiet
echo Starting server...
start "" http://localhost:5000/slicer
python app.py
