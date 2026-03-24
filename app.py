import os, re, subprocess, tempfile
from flask import Flask, request, send_file, jsonify
from pypdf import PdfWriter, PdfReader
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024  # 20MB

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

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
