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


def _repair_mesh(mesh):
    """Make a mesh watertight so the manifold boolean engine accepts it.

    Real-world / downloaded STLs are almost never watertight; the manifold
    engine then refuses them ("Not all meshes are volumes!") and every slab
    intersection returns empty. We merge/clean and fill holes to recover a
    closed solid wherever possible.
    """
    import trimesh
    try:
        mesh.merge_vertices()
        mesh.update_faces(mesh.unique_faces())
        mesh.update_faces(mesh.nondegenerate_faces())
        mesh.remove_unreferenced_vertices()
        if not mesh.is_watertight:
            trimesh.repair.fill_holes(mesh)
        trimesh.repair.fix_normals(mesh)
        trimesh.repair.fix_winding(mesh)
    except Exception:
        pass
    return mesh




def _facade_cut(mesh, axis_idx, depth_ratio):
    """Cut mesh to keep only the front depth_ratio fraction along the depth axis."""
    try:
        import trimesh
        import numpy as np

        perp = [i for i in range(3) if i != axis_idx]
        extents = mesh.extents
        depth_ax = perp[0] if extents[perp[0]] >= extents[perp[1]] else perp[1]

        lo = float(mesh.bounds[0][depth_ax])
        hi = float(mesh.bounds[1][depth_ax])
        depth_span = hi - lo

        kept_size = depth_span * depth_ratio
        center_val = lo + kept_size / 2.0

        big = float(max(mesh.extents) * 10)
        box_extents = [big, big, big]
        box_extents[depth_ax] = kept_size

        T = np.eye(4)
        T[depth_ax, 3] = center_val
        box = trimesh.creation.box(extents=box_extents, transform=T)

        _repair_mesh(mesh)
        result = mesh.intersection(box, engine='manifold')
        if result is not None and len(result.faces) > 0:
            return result
    except Exception:
        pass
    return mesh


def _fin_profile_parabolic(w_wide, w_narrow, h_lo, h_hi):
    """Parabolic lens fin cross-section as a Shapely polygon in (depth, height) space."""
    import numpy as np
    from shapely.geometry import Polygon

    h_half = (h_hi - h_lo) / 2.0
    if h_half <= 0 or w_wide <= 0:
        return None
    h_ctr = (h_lo + h_hi) / 2.0

    hs = np.linspace(h_lo, h_hi, 40)
    ws = [max(w_wide - (w_wide - w_narrow) * min(1.0, abs(h - h_ctr) / h_half) ** 2, 0.1)
          for h in hs]

    right = [(ws[i] / 2.0, hs[i]) for i in range(len(hs))]
    left  = [(-ws[i] / 2.0, hs[i]) for i in range(len(hs) - 1, -1, -1)]
    poly = Polygon(right + left)
    if not poly.is_valid:
        poly = poly.buffer(0)
    return poly if not poly.is_empty else None


def _make_fin_modifier(fin_profile, d_lo, d_hi, h_lo, h_hi,
                       ax, d_ax, h_ax, slab_center, slab_thick):
    """Negative modifier for one slab from a 2D fin profile polygon.

    Computes (model_bbox − fin_profile) so the modifier is bounded by the
    model's actual depth extent — no kilometer-long slabs.

    fin_profile: Shapely polygon in (u=depth_ax, v=height_ax) space
    d_lo/d_hi:  model bounds along depth_ax
    h_lo/h_hi:  model bounds along height_ax
    """
    try:
        import trimesh
        import numpy as np
        from shapely.geometry import box as shapely_box, MultiPolygon

        margin = 1.0
        bbox = shapely_box(d_lo - margin, h_lo - margin,
                           d_hi + margin, h_hi + margin)
        neg = bbox.difference(fin_profile)
        if neg.is_empty:
            return None

        h = slab_thick + 0.2
        polys = list(neg.geoms) if isinstance(neg, MultiPolygon) else [neg]
        parts = []
        for p in polys:
            if p.area < 1e-6:
                continue
            try:
                parts.append(trimesh.creation.extrude_polygon(p, h))
            except Exception:
                continue
        if not parts:
            return None

        out = trimesh.util.concatenate(parts) if len(parts) > 1 else parts[0]

        # Map local(u=depth, v=height, z_ext=slice) → world
        shift = slab_center - h / 2.0
        T = np.zeros((4, 4)); T[3, 3] = 1.0
        T[d_ax, 0] = 1.0
        T[h_ax, 1] = 1.0
        T[ax,   2] = 1.0
        T[ax,   3] = shift
        out.apply_transform(T)
        return out
    except Exception:
        return None


def _make_base_plate(slabs_combined, base_width=0, base_length=0, mesh_bounds=None):
    """Create a 1mm-thick horizontal base plate at the bottom (Z_min) of the piece.

    Always flat in the XY plane — the standard 3D printing orientation where
    the printer starts from the base and builds upward in Z.

    base_width  → X extent (mm); 0 = auto from model X + 20mm
    base_length → Y extent (mm); 0 = auto from model Y + 20mm
    """
    import trimesh
    import numpy as np

    try:
        bounds = mesh_bounds if mesh_bounds is not None else slabs_combined.bounds
        ext    = bounds[1] - bounds[0]

        bw = float(base_width)  if base_width  > 0 else float(ext[0]) + 20.0
        bl = float(base_length) if base_length > 0 else float(ext[1]) + 20.0

        cx = (bounds[0][0] + bounds[1][0]) / 2.0
        cy = (bounds[0][1] + bounds[1][1]) / 2.0
        cz = float(bounds[0][2]) - 0.5   # 0.5mm below the slabs

        t = np.eye(4)
        t[0, 3] = cx; t[1, 3] = cy; t[2, 3] = cz
        return trimesh.creation.box(extents=[bw, bl, 1.0], transform=t)
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
    base_width     = float(request.form.get('base_width',  0))
    base_length    = float(request.form.get('base_length', 0))
    flip           = request.form.get('flip', 'stand').strip().lower()  # 'stand' or 'lay'
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

        # Apply flip: rotate 90° around X axis (стоячи=0°, лежачи=90°)
        if flip == 'lay':
            try:
                rot = trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
                mesh.apply_transform(rot)
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

        # Light repair so normals/winding are consistent; no split needed.
        _repair_mesh(mesh)

        lo  = float(mesh.bounds[0][ax])
        hi  = float(mesh.bounds[1][ax])
        # pitch = total/(n-gap) ensures first slab starts exactly at lo and
        # last slab ends exactly at hi with equal internal gaps, no edge cutoff.
        pitch     = (hi - lo) / max(slices_n - gap, 1e-6)
        thickness = pitch * (1.0 - gap)
        big       = float(max(mesh.extents) * 10)

        # World-Z bounds for lens profile (varies with height, not slice axis)
        mesh_z_min = float(mesh.bounds[0][2])
        mesh_z_max = float(mesh.bounds[1][2])

        # Determine lens parameters (w_wide/w_narrow resolved later with model depth)
        use_lens = (w_wide_raw > 0) and (ax != 2)

        # Modifier approach: keep original mesh intact, subtract gap boxes.
        # The slicer (Bambu Studio) handles solid infill — no mesh cutting needed,
        # so non-watertight source models always produce solid fins.

        other_ax = [i for i in range(3) if i != ax]
        a1, a2 = other_ax

        # Compute slab boundary positions
        slab_lo_list, slab_hi_list = [], []
        for i in range(slices_n):
            center = lo + pitch * (i + (1.0 - gap) / 2.0)
            slab_lo_list.append(center - thickness / 2.0)
            slab_hi_list.append(center + thickness / 2.0)

        # Gap ranges: only between slabs (slabs span lo→hi exactly, no outer gaps needed)
        gap_ranges = []
        for i in range(slices_n - 1):
            gap_ranges.append((slab_hi_list[i], slab_lo_list[i + 1]))

        # Build ONE gap template mesh (centered at ax=0) + collect centers.
        # All gaps have the same size because pitch is uniform.
        cx = (mesh.bounds[0][a1] + mesh.bounds[1][a1]) / 2.0
        cz = (mesh.bounds[0][a2] + mesh.bounds[1][a2]) / 2.0
        cover_a1 = float(mesh.extents[a1]) + 20.0
        cover_a2 = float(mesh.extents[a2]) + 20.0
        gap_template = None
        gap_centers_ax = []
        for g_lo, g_hi in gap_ranges:
            sz = g_hi - g_lo
            if sz <= 0:
                continue
            if gap_template is None:
                extents_gt = np.zeros(3)
                extents_gt[ax] = sz
                extents_gt[a1] = cover_a1
                extents_gt[a2] = cover_a2
                gap_template = trimesh.creation.box(extents=extents_gt)
                t0 = np.zeros(3); t0[a1] = cx; t0[a2] = cz
                gap_template.apply_translation(t0)
            gap_centers_ax.append((g_lo + g_hi) / 2.0)

        # Lens modifier: ONE template mesh (centered at ax=0) + per-slab centers.
        # Template = extrusion of (bbox − fin_profile) side strips, slab_h thick,
        # already in world (d_ax, h_ax) coordinates, centered in ax.
        lens_template = None
        slab_centers_ax = []
        if use_lens:
            non_slice = [i for i in (0, 1, 2) if i != ax]
            h_ax = (non_slice[0]
                    if mesh.extents[non_slice[0]] >= mesh.extents[non_slice[1]]
                    else non_slice[1])
            d_ax = [i for i in non_slice if i != h_ax][0]

            h_lo_m = float(mesh.bounds[0][h_ax])
            h_hi_m = float(mesh.bounds[1][h_ax])
            d_lo_m = float(mesh.bounds[0][d_ax])
            d_hi_m = float(mesh.bounds[1][d_ax])
            model_depth = d_hi_m - d_lo_m

            eff_wide   = w_wide_raw   if w_wide_raw   > 0 else model_depth
            eff_narrow = w_narrow_raw if w_narrow_raw > 0 else eff_wide * 0.3
            eff_wide   = min(eff_wide,   model_depth * 0.99)
            eff_narrow = min(eff_narrow, eff_wide)

            fin_profile = _fin_profile_parabolic(eff_wide, eff_narrow, h_lo_m, h_hi_m)
            if fin_profile is not None:
                from shapely.geometry import box as shapely_box, MultiPolygon
                from shapely import affinity as _aff
                d_ctr = (d_lo_m + d_hi_m) / 2.0
                fin_profile = _aff.translate(fin_profile, xoff=d_ctr)
                margin = 1.0
                left_bbox  = shapely_box(d_lo_m - margin, h_lo_m - margin,
                                         d_ctr,            h_hi_m + margin)
                right_bbox = shapely_box(d_ctr,            h_lo_m - margin,
                                         d_hi_m + margin,  h_hi_m + margin)
                side_polys = []
                for half_bbox in (left_bbox, right_bbox):
                    strip = half_bbox.difference(fin_profile)
                    if not strip.is_empty and strip.area > 1.0:
                        side_polys.extend(
                            list(strip.geoms) if isinstance(strip, MultiPolygon) else [strip]
                        )
                if side_polys:
                    slab_h = thickness + 0.4
                    parts_t = []
                    for p in side_polys:
                        if p.area < 1e-6:
                            continue
                        try:
                            parts_t.append(trimesh.creation.extrude_polygon(p, slab_h))
                        except Exception:
                            continue
                    if parts_t:
                        raw = (trimesh.util.concatenate(parts_t)
                               if len(parts_t) > 1 else parts_t[0])
                        # Map local(x=d_ax, y=h_ax, z=ax) → world; center at ax=0
                        T = np.zeros((4, 4)); T[3, 3] = 1.0
                        T[d_ax, 0] = 1.0; T[h_ax, 1] = 1.0; T[ax, 2] = 1.0
                        T[ax, 3] = -slab_h / 2.0
                        raw.apply_transform(T)
                        lens_template = raw
                        slab_centers_ax = [
                            (slab_lo_list[i] + slab_hi_list[i]) / 2.0
                            for i in range(slices_n)
                        ]

        # ─── Export Bambu 3MF with negative_part modifiers ──────────────────────
        # Component reuse: one template mesh per modifier type, N component
        # instances with per-slab/gap ax-translation transforms.
        import zipfile, uuid as _uuid
        from lxml import etree

        ns_core = "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"
        ns_prod = "http://schemas.microsoft.com/3dmanufacturing/production/2015/06"
        ns_bs   = "http://schemas.bambulab.com/package/2021"
        P_      = f'{{{ns_prod}}}'
        IDENT16 = "1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1"
        IDENT12 = "1 0 0 0 1 0 0 0 1 0 0 0"

        def _uid(): return str(_uuid.uuid4()).upper()

        def _write_mesh(parent, obj_id, tmesh, name):
            obj_el = etree.SubElement(parent, 'object',
                                      id=str(obj_id), type='model', name=name)
            m_el = etree.SubElement(obj_el, 'mesh')
            v_el = etree.SubElement(m_el, 'vertices')
            for v in tmesh.vertices:
                etree.SubElement(v_el, 'vertex',
                                 x=f'{v[0]:.5f}', y=f'{v[1]:.5f}', z=f'{v[2]:.5f}')
            t_el = etree.SubElement(m_el, 'triangles')
            for face in tmesh.faces:
                etree.SubElement(t_el, 'triangle',
                                 v1=str(face[0]), v2=str(face[1]), v3=str(face[2]))

        def _ax_tf(t):
            """Pure ax-translation transform string (3MF 12-value row-major)."""
            tr = [0.0, 0.0, 0.0]; tr[ax] = float(t)
            return f"1 0 0 0 1 0 0 0 1 {tr[0]:.5f} {tr[1]:.5f} {tr[2]:.5f}"

        model_name = os.path.splitext(secure_filename(f.filename or 'model'))[0]

        # Template objects: (mesh, name, subtype)
        # Object IDs: 1=original, 2=gap_template (if any), 3=lens_template (if any)
        tmpl_parts = [(mesh, model_name, 'normal_part')]
        if gap_template is not None:
            tmpl_parts.append((gap_template, 'gap_template', 'negative_part'))
        if lens_template is not None:
            tmpl_parts.append((lens_template, 'lens_template', 'negative_part'))

        orig_id      = 1
        gap_tmpl_id  = 2 if gap_template  is not None else None
        lens_tmpl_id = (2 if gap_template is None else 3) if lens_template is not None else None

        asm_id = len(tmpl_parts) + 1

        # Optional base plate
        base_plate = None
        if base_mode == 'with':
            try:
                base_plate = _make_base_plate(None, base_width, base_length, mesh.bounds)
            except Exception:
                pass
        bp_id = asm_id + 1 if base_plate is not None else None

        # Component instances: (objectid, transform_str)
        comp_instances = [(orig_id, IDENT12)]
        for c in gap_centers_ax:
            comp_instances.append((gap_tmpl_id, _ax_tf(c)))
        for c in slab_centers_ax:
            comp_instances.append((lens_tmpl_id, _ax_tf(c)))

        # ── 3D/Objects/object_1.model: template mesh data ─────────────────────
        obj_root = etree.Element('model', nsmap={None: ns_core}, unit='millimeter')
        obj_res = etree.SubElement(obj_root, 'resources')
        for idx, (tmesh_, name_, _) in enumerate(tmpl_parts, start=1):
            _write_mesh(obj_res, idx, tmesh_, name_)
        obj_xml = etree.tostring(obj_root, xml_declaration=True,
                                 encoding='UTF-8', pretty_print=True)

        # ── 3D/3dmodel.model: assembly with component reuse ───────────────────
        main_root = etree.Element('model',
                                  nsmap={None: ns_core, 'p': ns_prod, 'BambuStudio': ns_bs},
                                  unit='millimeter')
        main_root.set('requiredextensions', 'p')
        main_res = etree.SubElement(main_root, 'resources')

        asm_obj = etree.SubElement(main_res, 'object',
                                   id=str(asm_id), type='model', name=model_name)
        asm_obj.set(P_ + 'UUID', _uid())
        comps = etree.SubElement(asm_obj, 'components')
        for obj_id_, tf_str in comp_instances:
            comp = etree.SubElement(comps, 'component',
                                    objectid=str(obj_id_), transform=tf_str)
            comp.set(P_ + 'path', '/3D/Objects/object_1.model')
            comp.set(P_ + 'UUID', _uid())

        if base_plate is not None:
            _write_mesh(main_res, bp_id, base_plate, 'base')

        ext_z = float(mesh.extents[2])
        build_tf = f"1 0 0 0 1 0 0 0 1 128.00000 128.00000 {ext_z / 2.0:.5f}"
        build_el = etree.SubElement(main_root, 'build')
        build_el.set(P_ + 'UUID', _uid())
        asm_item = etree.SubElement(build_el, 'item',
                                    objectid=str(asm_id),
                                    transform=build_tf, printable='1')
        asm_item.set(P_ + 'UUID', _uid())
        if bp_id:
            bp_item = etree.SubElement(build_el, 'item',
                                       objectid=str(bp_id), printable='1')
            bp_item.set(P_ + 'UUID', _uid())

        main_xml = etree.tostring(main_root, xml_declaration=True,
                                  encoding='UTF-8', pretty_print=True)

        # ── Metadata/model_settings.config ────────────────────────────────────
        cfg = etree.Element('config')
        obj_cfg = etree.SubElement(cfg, 'object', id=str(asm_id))
        etree.SubElement(obj_cfg, 'metadata', key='name', value=model_name)
        etree.SubElement(obj_cfg, 'metadata', key='extruder', value='1')
        etree.SubElement(obj_cfg, 'metadata', face_count=str(len(mesh.faces)))
        for idx, (tmesh_, name_, subtype) in enumerate(tmpl_parts, start=1):
            p_el = etree.SubElement(obj_cfg, 'part', id=str(idx), subtype=subtype)
            etree.SubElement(p_el, 'metadata', key='name', value=name_)
            etree.SubElement(p_el, 'metadata', key='matrix', value=IDENT16)
            if subtype == 'negative_part':
                etree.SubElement(p_el, 'metadata', key='extruder', value='0')
            etree.SubElement(p_el, 'mesh_stat',
                             face_count=str(len(tmesh_.faces)),
                             edges_fixed='0', degenerate_facets='0',
                             facets_removed='0', facets_reversed='0',
                             backwards_edges='0')
        plate_el = etree.SubElement(cfg, 'plate')
        for k, v in [('plater_id', '1'), ('plater_name', ''), ('locked', 'false')]:
            etree.SubElement(plate_el, 'metadata', key=k, value=v)
        mi_el = etree.SubElement(plate_el, 'model_instance')
        for k, v in [('object_id', str(asm_id)), ('instance_id', '0'), ('identify_id', '1')]:
            etree.SubElement(mi_el, 'metadata', key=k, value=v)
        assm_sec = etree.SubElement(cfg, 'assemble')
        etree.SubElement(assm_sec, 'assemble_item',
                         object_id=str(asm_id), instance_id='0',
                         transform=IDENT12, offset="0 0 0")
        cfg_xml = etree.tostring(cfg, xml_declaration=True,
                                 encoding='UTF-8', pretty_print=True)

        # ── Pack ZIP ──────────────────────────────────────────────────────────
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.writestr('[Content_Types].xml',
                        '<?xml version="1.0" encoding="utf-8"?>'
                        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
                        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
                        '<Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>'
                        '<Default Extension="config" ContentType="application/xml"/>'
                        '</Types>')
            zf.writestr('_rels/.rels',
                        '<?xml version="1.0" encoding="utf-8"?>'
                        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                        '<Relationship Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"'
                        ' Target="/3D/3dmodel.model" Id="rel0"/>'
                        '</Relationships>')
            zf.writestr('3D/_rels/3dmodel.model.rels',
                        '<?xml version="1.0" encoding="utf-8"?>'
                        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                        '<Relationship Target="/3D/Objects/object_1.model" Id="rel-1"'
                        ' Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>'
                        '</Relationships>')
            zf.writestr('3D/3dmodel.model', main_xml)
            zf.writestr('3D/Objects/object_1.model', obj_xml)
            zf.writestr('Metadata/model_settings.config', cfg_xml)

        out_bytes = buf.getvalue()

    stem  = os.path.splitext(secure_filename(f.filename or 'model'))[0]
    dname = f'{stem}_sliced_{axis}{slices_n}.3mf'
    return send_file(
        io.BytesIO(out_bytes),
        as_attachment=True,
        download_name=dname,
        mimetype='model/3mf',
    )


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
