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

    with tempfile.TemporaryDirectory() as tmp:
        stl_in = os.path.join(tmp, 'in.stl')
        f.save(stl_in)

        mesh = trimesh.load(stl_in, force='mesh')
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(list(mesh.geometry.values()))
        if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
            return jsonify(error='Не удалось загрузить mesh'), 400

        ax  = {'X': 0, 'Y': 1, 'Z': 2}[axis]
        lo  = float(mesh.bounds[0][ax])
        hi  = float(mesh.bounds[1][ax])
        pitch     = (hi - lo) / slices_n
        thickness = pitch * (1.0 - gap)
        big       = float(max(mesh.extents) * 10)

        slabs = []
        for i in range(slices_n):
            center  = lo + (i + 0.5) * pitch
            extents = [big, big, big]
            extents[ax] = thickness
            t = np.eye(4)
            t[ax, 3] = center
            box = trimesh.creation.box(extents=extents, transform=t)
            try:
                slab = mesh.intersection(box, engine='manifold')
                if slab is not None and len(slab.faces) > 0:
                    slabs.append(slab)
            except Exception:
                continue

        if not slabs:
            return jsonify(error='Не удалось нарезать модель'), 500

        combined  = trimesh.util.concatenate(slabs)
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
