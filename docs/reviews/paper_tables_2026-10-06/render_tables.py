"""Render paper tables from an auditable result ledger. Run from any directory.
Requires reportlab, DejaVu Serif fonts, and pdftoppm for PNG previews.
No model calls are made. JSON is the editable specification.
"""
import csv
import json
import hashlib
import math
import subprocess
from pathlib import Path
from landscape_layout import render_landscape
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


def load_results(specs):
    data = json.loads((ROOT / 'measured_results.json').read_text())
    for source in data['sources'].values():
        actual = hashlib.sha256((ROOT / source['snapshot']).read_bytes()).hexdigest()
        if actual != source['sha256']:
            raise ValueError(f"Source snapshot changed: {source['snapshot']}")
    index = {}
    for cell in data['cells']:
        name = cell['table']
        spec = specs[name]
        allowed_rows = {(m['id'], method['key']) for m, method in make_rows(spec)}
        allowed_cols = {c['key'] for g in spec['groups'] for c in g['columns']}
        key = (name, cell['model_id'], cell['method'], cell['metric'])
        if key in index:
            raise ValueError(f'Duplicate measurement: {key}')
        assert (cell['model_id'], cell['method']) in allowed_rows, key
        assert cell['metric'] in allowed_cols, key
        assert cell['cohort'] in data['cohorts'], key
        assert 0 <= cell['numerator'] <= cell['denominator'], key
        assert cell['denominator'] > 0, key
        assert cell['expected_n'] - cell['denominator'] == cell['missing_n'], key
        assert math.isclose(cell['value_pct'], 100 * cell['numerator'] / cell['denominator']), key
        snapshot = json.loads((ROOT / data['sources'][cell['source']]['snapshot']).read_text())
        for part in cell['pointer']:
            snapshot = snapshot[part]
        if 'scores' in snapshot:
            n, d = snapshot['scores']['4'] + snapshot['scores']['5'], snapshot['n']
        elif 'at_least_4_count' in snapshot:
            n, d = snapshot['at_least_4_count'], snapshot['valid']
        elif 'ge4' in snapshot:
            n, d = snapshot['ge4'], snapshot.get('valid', snapshot.get('n'))
        elif 'positive' in snapshot:
            n, d = snapshot['positive'], snapshot['valid']
        elif cell['metric'].endswith('tpr_pct'):
            n, d = snapshot['tp'], snapshot['tp'] + snapshot['fn']
        else:
            n, d = snapshot['fp'], snapshot['fp'] + snapshot['tn']
        assert (n, d) == (cell['numerator'], cell['denominator']), key
        index[key] = cell
    return data, index


def render(name, spec, data, measurements):
    rows = make_rows(spec)
    def cell_for(model, method, col):
        return measurements.get((name, model['id'], method['key'], col['key']))
    columns = [col for group in spec['groups'] for col in group['columns']]
    w, left, right = 570, 15, 555
    model_w, method_w = 76, 146
    numeric_start = left + model_w + method_w
    cell_w = (right - numeric_start) / len(columns)
    row_h = 14
    h = 18 + 44 + row_h * len(rows) + 37 + len(spec['notes']) * 11 + 13
    c = canvas.Canvas(str(ROOT / f'{name}.pdf'), pagesize=(w, h))
    c.setTitle(spec['title'] + ' - stored results and planned comparisons')

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
            for k, col in enumerate(columns):
                cell = cell_for(model, method, col)
                xx = numeric_start+(k+.5)*cell_w
                if cell:
                    value = f"{cell['value_pct']:.1f}"
                    text(value, xx-1, yy, 8, center=True)
                    offset = pdfmetrics.stringWidth(value, 'Paper', 8)/2
                    text(cell['cohort'], xx+offset, yy+3, 4.8)
                else:
                    text('—', xx, yy, 8, center=True)
        row_offset += len(block)
    bottom = top-40-len(rows)*row_h
    line(bottom, 1)
    text(spec['title'], left, bottom-18, 9, True)
    for i, note in enumerate(spec['notes']):
        text(note, left, bottom-32-i*11, 7)
    c.save()

    fields = ['model','model_id','method','status','split_manifest','repeat_id','cell_provenance'] + [col['key'] for col in columns]
    with (ROOT/f'{name}.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,lineterminator="\n")
        writer.writeheader()
        for model, method in rows:
            found = {col['key']: cell_for(model, method, col) for col in columns if cell_for(model, method, col)}
            row = {'model':model['display'].replace('\n',' '),'model_id':model['id'],'method':method['key'],
                   'status':'partially_measured' if found else 'planned',
                   'split_manifest':';'.join(sorted({c['cohort'] for c in found.values()})),
                   'repeat_id':'original_single_run' if found else '',
                   'cell_provenance':json.dumps({k:{f:c[f] for f in ['numerator','denominator','expected_n','missing_n','cohort','source','pointer']} for k,c in found.items()},ensure_ascii=False)}
            row.update({k:f"{c['value_pct']:.1f}" for k,c in found.items()})
            writer.writerow(row)

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
            vals=[]
            for col in columns:
                cell=cell_for(model, method, col)
                vals.append(f"{cell['value_pct']:.1f}" + r'\textsuperscript{' + cell['cohort'] + '}' if cell else r'\textemdash')
            lines.append(tr([mcell,label]+vals))
    lines.extend([r'\bottomrule',r'\end{tabular}}',r'\caption{'+esc(spec['title'])+'. '+esc(' '.join(spec['notes']))+'}',rf'\label{{tab:{name}_plan}}',r'\end{table*}'])
    (ROOT/f'{name}.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    subprocess.run(['pdftoppm','-png','-r','160','-singlefile',str(ROOT/f'{name}.pdf'),str(ROOT/name)],check=True)
    return len(rows)


if __name__ == '__main__':
    specs=json.loads((ROOT/'table_specs.json').read_text())
    data, measurements=load_results(specs)
    for name,spec in specs.items():
        count=render(name,spec,data,measurements)
        filled=sum(k[0]==name for k in measurements)
        layout=render_landscape(ROOT,name,spec,measurements)
        print(f'{name}: {count} CSV rows, {filled} measured cells; landscape {layout}')
