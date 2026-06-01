import io, os, re, subprocess, tempfile
from flask import Flask, request, send_file, jsonify
from pypdf import PdfWriter, PdfReader
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 150 * 1024 * 1024  # 150MB

def extract_num(fname):
    nums = re.findall(r'\d+', os.path.splitext(fname)[0])
    return nums[-1] if nums else '1'

def docx_to_pdf(docx_path, out_dir):
    result = subprocess.run(
        ['soffice', '--headless', '--convert-to', 'pdf',
         '--outdir', out_dir, docx_path],
        capture_output=True, text=True, timeout=60
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr)
    stem = os.path.splitext(os.path.basename(docx_path))[0]
    pdf = os.path.join(out_dir, stem + '.pdf')
    if not os.path.exists(pdf):
        raise RuntimeError('PDF не создан')
    return pdf

def merge_pdfs(p1, p2, out):
    w = PdfWriter()
    for p in (p1, p2):
        r = PdfReader(p)
        for pg in r.pages:
            w.add_page(pg)
    with open(out, 'wb') as f:
        w.write(f)

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/merge', methods=['POST'])
def merge():
    f1 = request.files.get('file1')
    f2 = request.files.get('file2')
    if not f1 or not f2:
        return jsonify(error='Нужно два файла'), 400
    if not f1.filename.endswith('.docx') or not f2.filename.endswith('.docx'):
        return jsonify(error='Только .docx файлы'), 400

    num = extract_num(secure_filename(f1.filename)) or \
          extract_num(secure_filename(f2.filename))
    out_name = f'documents_{num}.pdf'

    with tempfile.TemporaryDirectory() as tmp:
        p1 = os.path.join(tmp, 'doc1.docx')
        p2 = os.path.join(tmp, 'doc2.docx')
        f1.save(p1); f2.save(p2)

        pdf1 = docx_to_pdf(p1, tmp)
        pdf2 = docx_to_pdf(p2, tmp)

        out = os.path.join(tmp, out_name)
        merge_pdfs(pdf1, pdf2, out)

        return send_file(out, as_attachment=True,
                         download_name=out_name,
                         mimetype='application/pdf')

@app.route('/slicer')
def slicer_page():
    return app.send_static_file('slicer.html')


# ─── STL helper functions ───────────────────────────────────────────────────

def _scale_mesh(mesh, scale_str):
    """Scale mesh. scale_str: '80%' → multiply, '150' → fit longest axis to 150mm."""
    import numpy as np
    scale_str = scale_str.strip()
    if not scale_str:
        return mesh
    try:
        if scale_str.endswith('%'):
            factor = float(scale_str[:-1]) / 100.0
        else:
            target_mm = float(scale_str.replace('mm', '').strip())
            longest = float(max(mesh.extents))
            if longest == 0:
                return mesh
            factor = target_mm / longest
        if factor <= 0:
            return mesh
        matrix = np.eye(4)
        matrix[:3, :3] *= factor
        mesh.apply_transform(matrix)
    except Exception:
        pass
    return mesh


def _facade_cut(mesh, axis_idx, depth_ratio):
    """Cut mesh to keep only the front depth_ratio fraction along the depth axis."""
    try:
        import trimesh
        import numpy as np

        # The two axes perpendicular to the slice axis
        perp = [i for i in range(3) if i != axis_idx]

        # Find which perpendicular axis has larger extent (that's "depth")
        extents = mesh.extents
        if extents[perp[0]] >= extents[perp[1]]:
            depth_ax = perp[0]
        else:
            depth_ax = perp[1]

        lo = float(mesh.bounds[0][depth_ax])
        hi = float(mesh.bounds[1][depth_ax])
        depth_span = hi - lo

        # Keep front depth_ratio fraction (from lo side)
        cut_at = lo + depth_span * depth_ratio

        plane_normal = np.zeros(3)
        plane_normal[depth_ax] = -1.0  # normal pointing away from kept region
        plane_origin = np.zeros(3)
        plane_origin[depth_ax] = cut_at

        result = trimesh.intersections.slice_mesh_plane(
            mesh, plane_normal, plane_origin, cap=True
        )
        if result is not None and len(result.faces) > 0:
            return result
    except Exception:
        pass
    return mesh


def _make_lens_cutter(w_wide, w_narrow, mesh_z_min, mesh_z_max, big, ax, center):
    """Create a lens/leaf shaped cutting tool along the given axis.

    The lens narrows from w_wide (at mid model height) to w_narrow (at top/bottom).
    mesh_z_min/mesh_z_max are the actual world-Z bounds of the model.
    """
    import trimesh
    import numpy as np

    try:
        from shapely.geometry import Polygon

        z_center = (mesh_z_min + mesh_z_max) / 2.0
        z_half = (mesh_z_max - mesh_z_min) / 2.0

        if z_half <= 0 or w_wide <= 0:
            raise ValueError("Invalid dimensions")

        # Build profile: (width, world_z) pairs spanning full model height
        n_pts = 48
        zs = np.linspace(mesh_z_min, mesh_z_max, n_pts)
        ws = []
        for z in zs:
            # cos² profile: wide at center, narrow at ends
            t = (z - z_center) / z_half  # -1..1
            # clamp to avoid numerical issues
            t = max(-1.0, min(1.0, t))
            cos_val = np.cos(t * np.pi / 2.0)
            w = w_narrow + (w_wide - w_narrow) * cos_val ** 2
            ws.append(w)

        # Build 2D polygon in (ax_coord, Z) plane
        # Right side: z from bottom to top at +w/2
        # Left side: z from top to bottom at -w/2
        right = [(ws[i] / 2.0, zs[i]) for i in range(n_pts)]
        left  = [(-ws[i] / 2.0, zs[i]) for i in range(n_pts - 1, -1, -1)]
        coords = right + left
        poly = Polygon(coords)
        if not poly.is_valid:
            poly = poly.buffer(0)

        # Extrude along the "big" direction (the third axis)
        extruded = trimesh.creation.extrude_polygon(poly, big)

        # Now apply transform to map local→world axes
        # extruded local: X=ax_coord (col 0), Y=Z_world (col 1), Z=extrusion (col 2)
        # We need to re-map so the extrusion is centred
        T = np.eye(4)

        if ax == 0:
            # slice axis X: lens profile in (X, Z_world) plane, extrude along Y
            # local X → world X, local Y → world Z, local Z → world Y
            T = np.array([
                [1, 0, 0, 0],
                [0, 0, 1, -big/2],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
            ], dtype=float)
        elif ax == 1:
            # slice axis Y: lens profile in (Y, Z_world) plane, extrude along X
            # local X → world Y, local Y → world Z, local Z → world X
            T = np.array([
                [0, 0, 1, -big/2],
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
            ], dtype=float)
        else:
            # ax == 2: just use rectangular box (lens in XY plane is complex)
            raise ValueError("Use box for Z axis")

        extruded.apply_transform(T)

        # Centre the cutter on the slab centre
        bounds = extruded.bounds
        shift = np.zeros(3)
        shift[ax] = center - (bounds[0][ax] + bounds[1][ax]) / 2.0
        T2 = np.eye(4)
        T2[:3, 3] = shift
        extruded.apply_transform(T2)

        return extruded

    except Exception:
        # Fall back to rectangular box
        import trimesh
        import numpy as np
        extents = [big, big, big]
        extents[ax] = w_wide
        t = np.eye(4)
        t[ax, 3] = center
        return trimesh.creation.box(extents=extents, transform=t)


def _make_base_plate(slabs_combined, base_thickness):
    """Create a horizontal flat box at the bottom of sliced mesh."""
    import trimesh
    import numpy as np

    try:
        bounds = slabs_combined.bounds
        x_min, y_min, z_min = bounds[0]
        x_max, y_max, z_max = bounds[1]

        border = 2.0
        width  = (x_max - x_min) + 2 * border
        depth  = (y_max - y_min) + 2 * border
        height = base_thickness

        cx = (x_min + x_max) / 2.0
        cy = (y_min + y_max) / 2.0
        cz = z_min - height / 2.0

        t = np.eye(4)
        t[0, 3] = cx
        t[1, 3] = cy
        t[2, 3] = cz

        base = trimesh.creation.box(extents=[width, depth, height], transform=t)
        return base
    except Exception:
        return None


def _engrave_number(slab, number, ax, size_mm, engrave_depth=0.8):
    """Engrave slice number on the narrow face of the slab."""
    try:
        import trimesh
        import numpy as np
        from matplotlib.textpath import TextPath
        from matplotlib.path import Path
        from shapely.geometry import Polygon, MultiPolygon
        from shapely.ops import unary_union

        bounds = slab.bounds
        extents = slab.extents

        # Find the narrow face axis (perpendicular to slice axis and Z)
        perp_axes = [i for i in range(3) if i != ax and i != 2]
        if not perp_axes:
            perp_axes = [0 if ax != 0 else 1]
        face_ax = perp_axes[0]

        # Position: bottom area of the slab, on the face with smaller extent
        tp = TextPath((0, 0), str(number), size=size_mm)
        path_data = tp.to_polygons()

        polys = []
        for verts in path_data:
            if len(verts) >= 3:
                try:
                    p = Polygon(verts)
                    if p.is_valid and p.area > 0:
                        polys.append(p)
                except Exception:
                    continue

        if not polys:
            return slab

        text_poly = unary_union(polys)
        if text_poly.is_empty:
            return slab

        # Extrude text to engrave_depth * 2 (we'll intersect/difference later)
        text_3d = trimesh.creation.extrude_polygon(
            text_poly if isinstance(text_poly, Polygon)
            else list(text_poly.geoms)[0],
            engrave_depth * 2
        )

        # Position on slab face
        tb = text_3d.bounds
        text_w = tb[1][0] - tb[0][0]
        text_h = tb[1][1] - tb[0][1]

        # Place at bottom-left area of the slab face
        target = np.zeros(3)
        target[face_ax]  = bounds[0][face_ax] - engrave_depth  # front face
        target[2]        = bounds[0][2] + 2.0  # 2mm from bottom
        # Center horizontally on the face
        other_ax = [i for i in range(3) if i != ax and i != face_ax][0]
        face_center = (bounds[0][other_ax] + bounds[1][other_ax]) / 2.0
        target[other_ax] = face_center - text_w / 2.0

        T = np.eye(4)
        # Rotate text so it appears on the face_ax face
        # text is in XY plane; we need it on the face_ax/Z plane
        if face_ax == 0:
            # text X→world Z, text Y→world Z already (no rotation needed for simple case)
            pass
        T[:3, 3] = target - [tb[0][0], tb[0][1], tb[0][2]]
        text_3d.apply_transform(T)

        result = slab.difference(text_3d, engine='manifold')
        if result is not None and len(result.faces) > 0:
            return result
    except Exception:
        pass
    return slab


# ─── routes ────────────────────────────────────────────────────────────────

@app.route('/api/slice', methods=['POST'])
def api_slice():
    try:
        import trimesh
        import numpy as np
    except ImportError:
        return jsonify(error='trimesh не установлен на сервере'), 500

    f = request.files.get('file')
    if not f:
        return jsonify(error='Файл не передан'), 400

    try:
        slices_n = max(5, min(200, int(request.form.get('slices', 30))))
        gap      = max(0.05, min(0.90, float(request.form.get('gap', 0.45))))
        axis     = request.form.get('axis', 'Y').upper()
    except (ValueError, TypeError):
        return jsonify(error='Неверные параметры'), 400

    if axis not in ('X', 'Y', 'Z'):
        return jsonify(error='Ось должна быть X, Y или Z'), 400

    # New parameters
    scale_str      = request.form.get('scale', '').strip()
    base_mode      = request.form.get('base', 'none').strip().lower()  # 'with' or 'none'
    base_thickness = float(request.form.get('base_thickness', 3.0))
    facade_on      = request.form.get('facade', 'false').lower() == 'true'
    facade_depth   = max(0.1, min(1.0, float(request.form.get('facade_depth', 0.5))))
    w_wide_raw     = float(request.form.get('w_wide', 0))
    w_narrow_raw   = float(request.form.get('w_narrow', 0))
    numbering_on   = request.form.get('numbering', 'false').lower() == 'true'
    num_size       = float(request.form.get('num_size', 5.0))

    with tempfile.TemporaryDirectory() as tmp:
        stl_in = os.path.join(tmp, 'in.stl')
        f.save(stl_in)

        mesh = trimesh.load(stl_in, force='mesh')
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(list(mesh.geometry.values()))
        if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
            return jsonify(error='Не удалось загрузить mesh'), 400

        # Apply scale
        if scale_str:
            try:
                mesh = _scale_mesh(mesh, scale_str)
            except Exception:
                pass

        # Apply facade cut
        ax = {'X': 0, 'Y': 1, 'Z': 2}[axis]
        if facade_on:
            try:
                mesh = _facade_cut(mesh, ax, facade_depth)
            except Exception:
                pass

        # Center the mesh at the origin. All cutters (box and lens) are built
        # around the origin on the non-slice axes, so an off-origin model would
        # only be partially intersected (or missed entirely). This also matches
        # the front-end preview, which centers the geometry.
        try:
            mesh.apply_translation(-mesh.bounds.mean(axis=0))
        except Exception:
            pass

        lo  = float(mesh.bounds[0][ax])
        hi  = float(mesh.bounds[1][ax])
        pitch     = (hi - lo) / slices_n
        thickness = pitch * (1.0 - gap)
        big       = float(max(mesh.extents) * 10)

        # World-Z bounds for lens profile (varies with height, not slice axis)
        mesh_z_min = float(mesh.bounds[0][2])
        mesh_z_max = float(mesh.bounds[1][2])

        # Determine lens parameters
        use_lens = (w_wide_raw > 0) and (ax != 2)
        w_wide   = w_wide_raw if w_wide_raw > 0 else thickness
        w_narrow = w_narrow_raw if w_narrow_raw > 0 else w_wide

        # Clamp w_wide so slab can't overlap adjacent slabs
        if use_lens:
            w_wide   = min(w_wide, thickness * 0.99)
            w_narrow = min(w_narrow, w_wide)

        slabs = []
        for i in range(slices_n):
            center  = lo + (i + 0.5) * pitch

            if use_lens:
                try:
                    cutter = _make_lens_cutter(w_wide, w_narrow, mesh_z_min, mesh_z_max, big, ax, center)
                except Exception:
                    # fall back to box
                    extents = [big, big, big]
                    extents[ax] = thickness
                    t = np.eye(4)
                    t[ax, 3] = center
                    cutter = trimesh.creation.box(extents=extents, transform=t)
            else:
                extents = [big, big, big]
                extents[ax] = thickness
                t = np.eye(4)
                t[ax, 3] = center
                cutter = trimesh.creation.box(extents=extents, transform=t)

            try:
                slab = mesh.intersection(cutter, engine='manifold')
                if slab is not None and len(slab.faces) > 0:
                    if numbering_on:
                        try:
                            slab = _engrave_number(slab, i + 1, ax, num_size)
                        except Exception:
                            pass
                    slabs.append(slab)
            except Exception:
                continue

        if not slabs:
            return jsonify(error='Не удалось нарезать модель'), 500

        combined = trimesh.util.concatenate(slabs)

        # Add base plate
        if base_mode == 'with':
            try:
                base = _make_base_plate(combined, base_thickness)
                if base is not None:
                    combined = trimesh.util.concatenate([combined, base])
            except Exception:
                pass

        out_bytes = combined.export(file_type='stl')

    stem  = os.path.splitext(secure_filename(f.filename or 'model'))[0]
    dname = f'{stem}_sliced_{axis}{slices_n}.stl'
    return send_file(
        io.BytesIO(out_bytes),
        as_attachment=True,
        download_name=dname,
        mimetype='application/octet-stream',
    )


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
