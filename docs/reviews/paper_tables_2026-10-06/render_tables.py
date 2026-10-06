"""Render unmeasured paper tables. Run from any directory with Python 3.
Requires reportlab, DejaVu Serif fonts, and pdftoppm for PNG previews.
No model calls are made. JSON is the editable specification.
"""
import csv
import json
import subprocess
from pathlib import Path
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

ROOT = Path(__file__).resolve().parent
FONT_DIR = Path('/usr/share/fonts/truetype/dejavu')
pdfmetrics.registerFont(TTFont('Paper', str(FONT_DIR / 'DejaVuSerif.ttf')))
pdfmetrics.registerFont(TTFont('PaperBold', str(FONT_DIR / 'DejaVuSerif-Bold.ttf')))


def make_rows(spec):
    rows = []
    for model in spec['models']:
        for method in spec['methods']:
            supported = method.get('models')
            if supported is not None and model['id'] not in supported:
                continue
            rows.append((model, method))
    return rows


def render(name, spec):
    rows = make_rows(spec)
    columns = [col for group in spec['groups'] for col in group['columns']]
    w, left, right = 570, 15, 555
    model_w, method_w = 76, 146
    numeric_start = left + model_w + method_w
    cell_w = (right - numeric_start) / len(columns)
    row_h = 14
    h = 18 + 44 + row_h * len(rows) + 37 + len(spec['notes']) * 11 + 13
    c = canvas.Canvas(str(ROOT / f'{name}.pdf'), pagesize=(w, h))
    c.setTitle(spec['title'] + ' - unmeasured experiment plan')

    def text(s, x, y, size=7.8, bold=False, center=False):
        font = 'PaperBold' if bold else 'Paper'
        c.setFont(font, size)
        if center:
            c.drawCentredString(x, y, s)
        else:
            c.drawString(x, y, s)

    def line(y, width=.4, x1=left, x2=right):
        c.setLineWidth(width)
        c.line(x1, y, x2, y)

    top = h - 18
    line(top, 1)
    text('Model', left+3, top-26, 8, True)
    text('Method', left+model_w+3, top-26, 8, True)
    idx = 0
    for group in spec['groups']:
        start = numeric_start + idx * cell_w
        n = len(group['columns'])
        text(group['label'], start+n*cell_w/2, top-13, 8, True, True)
        line(top-19, .4, start+2, start+n*cell_w-2)
        for j, col in enumerate(group['columns']):
            text(col['label'], start+(j+.5)*cell_w, top-31, 7.2, center=True)
        idx += n
    line(top-40, .65)
    # Model names are vertically centered within each method block.
    row_offset = 0
    for model in spec['models']:
        block = [(m, method) for m, method in rows if m['id'] == model['id']]
        if not block:
            continue
        y = top-40-row_offset*row_h
        if row_offset:
            line(y, .6)
        names = model['display'].split('\n')
        mid = y-len(block)*row_h/2
        for i, part in enumerate(names):
            text(part, left+3, mid+(len(names)-1)*5-i*10-2, 7.8, True)
        for j, (_, method) in enumerate(block):
            yy = y-j*row_h-10
            text(method['label'], left+model_w+3, yy, 7.5, method.get('proposed',False))
            for k in range(len(columns)):
                text('—', numeric_start+(k+.5)*cell_w, yy, 8, center=True)
        row_offset += len(block)
    bottom = top-40-len(rows)*row_h
    line(bottom, 1)
    text(spec['title'], left, bottom-18, 9, True)
    for i, note in enumerate(spec['notes']):
        text(note, left, bottom-32-i*11, 7)
    c.save()

    fields = ['model','model_id','method','status','split_manifest','repeat_id'] + [col['key'] for col in columns]
    with (ROOT/f'{name}.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,lineterminator="\n")
        writer.writeheader()
        for model, method in rows:
            writer.writerow({'model':model['display'].replace('\n',' '),'model_id':model['id'],'method':method['key'],'status':'planned'})

    def esc(s):
        return s.replace('_',r'\_').replace('%',r'\%').replace('↑',r'$\uparrow$').replace('↓',r'$\downarrow$').replace('≥',r'$\geq$').replace('—',r'\textemdash')
    def tr(cells): return ' & '.join(cells)+r' \\'
    lines=[r'\begin{table*}[t]',r'\centering',r'\scriptsize',r'\setlength{\tabcolsep}{3pt}',r'\resizebox{\textwidth}{!}{%',r'\begin{tabular}{ll'+'c'*len(columns)+'}',r'\toprule']
    lines.append(tr([r'\multirow{2}{*}{Model}',r'\multirow{2}{*}{Method}']+[rf'\multicolumn{{{len(g["columns"])}}}{{c}}{{{esc(g["label"])}}}' for g in spec['groups']]))
    idx=3; rules=[]
    for group in spec['groups']:
        end=idx+len(group['columns'])-1
        rules.append(rf'\cmidrule(lr){{{idx}-{end}}}'); idx=end+1
    lines.extend([' '.join(rules),tr(['','']+[esc(c['label']) for c in columns]),r'\midrule'])
    for i,model in enumerate(spec['models']):
        block=[method for m,method in rows if m['id']==model['id']]
        if not block: continue
        if i: lines.append(r'\midrule')
        for j, method in enumerate(block):
            mcell=rf'\multirow{{{len(block)}}}{{*}}{{\shortstack[l]{{'+r'\\'.join(esc(x) for x in model['display'].split('\n'))+'}}' if j==0 else ''
            label=esc(method['label'])
            if method.get('proposed'): label=r'\textbf{'+label+'}'
            lines.append(tr([mcell,label]+[r'\textemdash']*len(columns)))
    lines.extend([r'\bottomrule',r'\end{tabular}}',r'\caption{'+esc(spec['title'])+'. '+esc(' '.join(spec['notes']))+'}',rf'\label{{tab:{name}_plan}}',r'\end{table*}'])
    (ROOT/f'{name}.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    subprocess.run(['pdftoppm','-png','-r','160','-singlefile',str(ROOT/f'{name}.pdf'),str(ROOT/name)],check=True)
    return len(rows)


if __name__ == '__main__':
    specs=json.loads((ROOT/'table_specs.json').read_text())
    for name,spec in specs.items():
        count=render(name,spec)
        print(f'{name}: {count} planned rows')
