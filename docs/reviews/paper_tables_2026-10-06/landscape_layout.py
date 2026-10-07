"""Wide paper tables: model -> dataset -> metric headers, one row per method."""
import subprocess
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics


def dataset_groups(spec):
    """Keep paired premise metrics; expand QA categories into named datasets."""
    qa_names = {
        'medqa_accuracy_pct': 'MedQA', 'pubmedqa_accuracy_pct': 'PubMedQA',
        'medbullets_accuracy_pct': 'Medbullets', 'mmlu_accuracy_pct': 'MMLU',
        'gsm8k_em_pct': 'GSM8K',
    }
    groups = []
    for group in spec['groups']:
        if group['label'] in ('Medical QA', 'General QA'):
            for col in group['columns']:
                metric = 'EM ↑' if col['key'] == 'gsm8k_em_pct' else 'Acc. ↑'
                groups.append({'label': qa_names[col['key']], 'columns': [dict(col, label=metric)]})
        else:
            groups.append(group)
    return groups


def render_landscape(root, name, spec, measurements):
    models = [m for m in spec['models'] if m['id'] != 'model_independent']
    groups = dataset_groups(spec)
    methods = spec['methods']
    columns = [col for group in groups for col in group['columns']]
    per_model = len(columns)
    # Preserve readable native text instead of shrinking 45 metric columns to A4.
    group_widths = [max(44 * len(g['columns']), pdfmetrics.stringWidth(g['label'], 'Paper', 9) + 12)
                    for g in groups]
    model_w = sum(group_widths)
    left, method_w, gap = 22, 190, 12
    start = left + method_w
    right = start + len(models) * model_w + (len(models) - 1) * gap
    width = right + left
    row_h, header_h = 21, 65
    height = 42 + header_h + len(methods) * row_h + 24 + 11 * len(spec['notes']) + 12
    c = canvas.Canvas(str(root / f'{name}.pdf'), pagesize=(width, height))
    c.setTitle(spec['title'])

    def text(s, x, y, size=9, bold=False, center=False):
        c.setFont('PaperBold' if bold else 'Paper', size)
        (c.drawCentredString if center else c.drawString)(x, y, s)

    def line(y, weight=.5, x1=left, x2=right):
        c.setLineWidth(weight)
        c.line(x1, y, x2, y)

    def find(model, method, col):
        return measurements.get((name, model, method, col['key']))

    def draw_value(cell, x, y):
        if not cell:
            text('—', x, y, center=True)
            return
        label = f"{cell['value_pct']:.1f}"
        text(label, x - 1, y, center=True)

    def esc(s):
        for a, b in [('&', r'\&'), ('_', r'\_'), ('%', r'\%'), ('↑', r'$\uparrow$'),
                     ('↓', r'$\downarrow$'), ('≥', r'$\geq$'), ('—', r'\textemdash')]:
            s = s.replace(a, b)
        return s

    def tr(cells):
        return ' & '.join(cells) + r' \\'

    def texval(cell):
        return f"{cell['value_pct']:.1f}" if cell else r'\textemdash'

    text(spec['title'], left, height - 20, 12, True)
    top = height - 34
    line(top, 1.1)
    text('Method', left + 3, top - 38, 10, True)
    locations = []
    for i, model in enumerate(models):
        x0 = start + i * (model_w + gap)
        text(model['display'].replace('\n', ' '), x0 + model_w / 2, top - 17, 11, True, True)
        line(top - 23, .6, x0 + 2, x0 + model_w - 2)
        x = x0
        model_locations = []
        for group, gw in zip(groups, group_widths):
            text(group['label'], x + gw / 2, top - 37, 9, center=True)
            line(top - 43, .35, x + 2, x + gw - 2)
            cw = gw / len(group['columns'])
            for j, col in enumerate(group['columns']):
                mid = x + (j + .5) * cw
                text(col['label'], mid, top - 57, 8.5, center=True)
                model_locations.append((mid, col))
            x += gw
        locations.append(model_locations)
    y = top - header_h
    line(y, .8)
    ncols = 1 + per_model * len(models)
    latex = [r'\begin{table*}[t]', r'\centering', r'\scriptsize', r'\setlength{\tabcolsep}{3pt}',
             r'\resizebox{\textwidth}{!}{%', r'\begin{tabular}{l' + 'c' * (ncols - 1) + '}', r'\toprule',
             tr([r'\multirow{3}{*}{Method}'] + [r'\multicolumn{' + str(per_model) + '}{c}{' + esc(m['display'].replace('\n', ' ')) + '}' for m in models])]
    latex.append(' '.join(r'\cmidrule(lr){' + str(2 + i * per_model) + '-' + str(1 + (i + 1) * per_model) + '}' for i in range(len(models))))
    latex.append(tr([''] + [r'\multicolumn{' + str(len(g['columns'])) + '}{c}{' + esc(g['label']) + '}' for _ in models for g in groups]))
    latex.append(tr([''] + [esc(col['label']) for _ in models for col in columns]))
    latex.append(r'\midrule')
    for method in methods:
        if method.get('proposed'):
            c.setFillColorRGB(.94, .95, .96)
            c.rect(left, y - row_h, right - left, row_h, stroke=0, fill=1)
            c.setFillColorRGB(0, 0, 0)
            line(y, .5)
            latex.append(r'\midrule')
        yy = y - row_h + 7
        text(method['label'], left + 3, yy, 9, bool(method.get('proposed')))
        label = esc(method['label'])
        if method.get('proposed'):
            label = r'\textbf{' + label + '}'
        if method['key'] == 'tfidf':
            descriptions = []
            for group in groups:
                parts = []
                for col in group['columns']:
                    cell = find('model_independent', method['key'], col)
                    val = f"{cell['value_pct']:.1f}%" if cell else '—'
                    parts.append(col['label'] + ' ' + val)
                descriptions.append(group['label'] + ': ' + ', '.join(parts))
            shared = 'Shared across models — ' + ';  '.join(descriptions)
            text(shared, (start + right) / 2, yy, center=True)
            latex.append(tr([label, r'\multicolumn{' + str(ncols - 1) + '}{c}{' + esc(shared) + '}']))
        else:
            vals = []
            for model, positions in zip(models, locations):
                supported = method.get('models') is None or model['id'] in method['models']
                for x, col in positions:
                    cell = find(model['id'], method['key'], col)
                    if supported:
                        draw_value(cell, x, yy)
                    else:
                        text('n/a', x, yy, 8, center=True)
                    vals.append(texval(cell) if supported else 'n/a')
            latex.append(tr([label] + vals))
        y -= row_h
    line(y, 1.1)
    for i, note in enumerate(spec['notes']):
        text(note, left, y - 18 - i * 11, 8)
    assert y - 18 - (len(spec['notes']) - 1) * 11 > 5
    c.save()
    latex += [r'\bottomrule', r'\end{tabular}}', r'\caption{' + esc(spec['title']) + '. ' + esc(' '.join(spec['notes'])) + '}',
              r'\label{tab:' + name + '}', r'\end{table*}']
    (root / f'{name}.tex').write_text('\n'.join(latex) + '\n')
    subprocess.run(['pdftoppm', '-png', '-r', '120', '-singlefile', str(root / f'{name}.pdf'), str(root / name)], check=True)
    return {'width': width, 'height': height, 'visual_rows': len(methods), 'metric_columns': ncols - 1}
