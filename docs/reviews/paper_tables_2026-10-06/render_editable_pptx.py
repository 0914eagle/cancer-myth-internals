"""Build native editable PowerPoint tables from the audited paper-table ledger.
Requires python-pptx. No images or model calls; original cell provenance stays in JSON.
"""
import json
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.xmlchemy import OxmlElement
from landscape_layout import dataset_groups

ROOT = Path(__file__).resolve().parent
SPECS = json.loads((ROOT / 'table_specs.json').read_text())
# Use the same method roster and labels as the PNG/PDF renderer.
DATA = json.loads((ROOT / 'measured_results.json').read_text())
CELLS = {(c['table'], c['model_id'], c['method'], c['metric']): c for c in DATA['cells']}


def text_box(slide, text, x, y, w, h, size=12, bold=False):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    frame = box.text_frame
    frame.word_wrap = True
    frame.margin_left = frame.margin_right = 0
    frame.margin_top = frame.margin_bottom = 0
    for i, line in enumerate(text.split('\n')):
        p = frame.paragraphs[0] if i == 0 else frame.add_paragraph()
        p.text = line
        p.font.name = 'Arial'
        p.font.size = Pt(size)
        p.font.bold = bold
        p.font.color.rgb = RGBColor(25, 25, 25)
    return box


def format_cell(cell, text, font_size, bold=False, left=False, shaded=False):
    cell.text = text
    cell.margin_left = Inches(.07 if left else .015)
    cell.margin_right = Inches(.015)
    cell.margin_top = cell.margin_bottom = Inches(.025)
    cell.vertical_anchor = MSO_ANCHOR.MIDDLE
    cell.fill.solid()
    cell.fill.fore_color.rgb = RGBColor(239, 241, 243) if shaded else RGBColor(255, 255, 255)
    for p in cell.text_frame.paragraphs:
        p.alignment = PP_ALIGN.LEFT if left else PP_ALIGN.CENTER
        p.font.name = 'Arial'
        p.font.size = Pt(font_size)
        p.font.bold = bold
        p.font.color.rgb = RGBColor(20, 20, 20)
    props = cell._tc.get_or_add_tcPr()
    for edge in ('lnL', 'lnR', 'lnT', 'lnB'):
        line = OxmlElement('a:' + edge)
        line.set('w', '4000')
        fill = OxmlElement('a:solidFill')
        color = OxmlElement('a:srgbClr')
        color.set('val', 'B0B0B0')
        fill.append(color)
        line.append(fill)
        props.append(line)


def add_table_slide(prs, name, models, suffix):
    spec = SPECS[name]
    groups = dataset_groups(spec)
    columns = [col for g in groups for col in g['columns']]
    per = len(columns)
    methods = spec['methods']
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    title = {'table1': 'Table 1. False-premise detection',
             'table2': 'Table 2. Correction and normal-question preservation',
             'table3': 'Table 3. Medical and general QA preservation'}[name]
    text_box(slide, title, .32, .22, 12.7, .42, 21, True)
    text_box(slide, suffix + ' | Editable table cells', .34, .69, 12.6, .3, 11)
    nr, nc = 3 + len(methods), 1 + per * len(models)
    shape = slide.shapes.add_table(nr, nc, Inches(.32), Inches(1.1), Inches(12.69), Inches(5.1))
    table = shape.table
    table.columns[0].width = Inches(2.15)
    for col in list(table.columns)[1:]:
        col.width = Inches((12.69 - 2.15) / (nc - 1))
    for i, row in enumerate(table.rows):
        row.height = Inches(.30 if i < 3 else 4.2 / len(methods))
    fs = 9 if len(models) == 5 else 11
    if name == 'table3': fs -= 1
    for row in table.rows:
        for cell in row.cells:
            format_cell(cell, '', fs)
    table.cell(0, 0).merge(table.cell(2, 0))
    format_cell(table.cell(0, 0), 'Method', 12, True, left=True, shaded=True)
    for mi, model in enumerate(models):
        base = 1 + mi * per
        table.cell(0, base).merge(table.cell(0, base + per - 1))
        format_cell(table.cell(0, base), model['display'].replace('\n', ' '), fs + 1, True, shaded=True)
        offset = 0
        for group in groups:
            count = len(group['columns'])
            if count > 1:
                table.cell(1, base + offset).merge(table.cell(1, base + offset + count - 1))
            format_cell(table.cell(1, base + offset), group['label'], fs, True, shaded=True)
            for j, col in enumerate(group['columns']):
                format_cell(table.cell(2, base + offset + j), col['label'], fs, shaded=True)
            offset += count
    for ri, method in enumerate(methods, 3):
        proposed = method.get('proposed', False)
        format_cell(table.cell(ri, 0), method['label'], 11, proposed, True, proposed)
        for mi, model in enumerate(models):
            for ci, col in enumerate(columns):
                shared = name == 'table1' and method['key'] == 'tfidf'
                model_id = 'model_independent' if shared else model['id']
                cell = CELLS.get((name, model_id, method['key'], col['key']))
                supported = shared or method.get('models') is None or model['id'] in method['models']
                value = f"{cell['value_pct']:.1f}" if cell else ('—' if supported else 'n/a')
                format_cell(table.cell(ri, 1 + mi * per + ci), value, fs + .5, proposed, shaded=proposed)
    if name == 'table1':
        foot = ('Stored detection results: TPR = FPQ detection rate; FPR = false positives on normal questions (%).\n'
                'TF-IDF repeats the same model-independent detector across models. Probe needs hidden states; Luna is n/a.\n'
                'Direct/CoT retain historical protocols, not the newly proposed replacements. GEPA detection remains unmeasured.')
    elif name == 'table2':
        foot = ('Stored single-run results; common-fold evaluation is not yet complete. Score >= 4 (%).\n'
                'Cancer-Myth: public models up to 583/149; Luna 99/100. CREPE: Luna 751/2,253. Judges differ.\n'
                'Direct/CoT gate: historical label-only routing; not the new review-passing protocol. See repository for provenance.')
    else:
        foot = ('No completed compatible QA results are available. — = unmeasured; n/a = native hidden probe unavailable.\n'
                'Acc. = accuracy; EM = final-answer exact match. Freeze prompts and gates before QA evaluation.\n'
                'This slide is an editable evaluation plan; no missing results have been filled with invented values.')
    text_box(slide, foot, .34, 6.38, 12.65, .85, 10)
    slide.notes_slide.notes_text_frame.text = foot + '\nSource: docs/reviews/paper_tables_2026-10-06/measured_results.json'


def main():
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
    for name in ('table1', 'table2', 'table3'):
        models = [m for m in SPECS[name]['models'] if m['id'] != 'model_independent']
        add_table_slide(prs, name, models, 'Full table')
    for name in ('table1', 'table2', 'table3'):
        models = [m for m in SPECS[name]['models'] if m['id'] != 'model_independent']
        add_table_slide(prs, name, models[:3], 'Larger text: models 1–3')
        add_table_slide(prs, name, models[3:], 'Larger text: models 4–5')
    output = ROOT / 'paper_tables_editable.pptx'
    prs.save(output)
    reopened = Presentation(output)
    assert len(reopened.slides) == 9
    for slide in reopened.slides:
        assert sum(shape.has_table for shape in slide.shapes) == 1
        assert not any(shape.shape_type == 13 for shape in slide.shapes)
    print(output)


if __name__ == '__main__':
    main()
