"""Landscape views: model columns, method rows, dataset sections.
Unmeasured QA methods are collapsed only in the visual; CSV retains every cell.
"""
from pathlib import Path
import subprocess
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics


def render_landscape(root, name, spec, measurements):
    models=[m for m in spec['models'] if m['id']!='model_independent']
    groups=spec['groups'][:2]
    methods=spec['methods']
    detection=name=='table1'
    entries=[]
    for group in groups:
        entries.append(('band',group['label'],None,None))
        for method in methods:
            entries.append(('pair',method['label'],method,group['columns']))
    if not detection:
        qa_keys={c['key'] for g in spec['groups'][2:] for c in g['columns']}
        pending=not any(k[0]==name and k[3] in qa_keys for k in measurements)
        qa_title='Medical / general QA: accuracy (%) or GSM8K EM' + ('; all methods pending' if pending else '')
        entries.append(('band',qa_title,None,None))
        labels={'medqa_accuracy_pct':'MedQA','pubmedqa_accuracy_pct':'PubMedQA','medbullets_accuracy_pct':'Medbullets','mmlu_accuracy_pct':'MMLU','gsm8k_em_pct':'GSM8K'}
        for group in spec['groups'][2:]:
            for col in group['columns']:
                has_any=any(k[0]==name and k[3]==col['key'] for k in measurements)
                if not has_any:
                    entries.append(('qa',labels[col['key']]+' (all methods pending)',None,[col]))
                else:
                    # Expand automatically when actual per-method QA results become available.
                    entries.append(('band',labels[col['key']],None,None))
                    for method in methods:entries.append(('qa',method['label'],method,[col]))
    row_h=11.0 if not detection else 16.0
    band_h=17
    footer_h=19+9*len(spec['notes'])
    body_h=sum(band_h if kind=='band' else row_h for kind,*_ in entries)
    width=842
    height=max(595,70+body_h+footer_h)
    left,right=22,820
    method_w=205
    start=left+method_w
    model_w=(right-start)/len(models)
    c=canvas.Canvas(str(root/f'{name}.pdf'),pagesize=(width,height))
    c.setTitle(spec['title'])
    def text(s,x,y,size=8,bold=False,center=False):
        c.setFont('PaperBold' if bold else 'Paper',size)
        (c.drawCentredString if center else c.drawString)(x,y,s)
    def line(y,weight=.4):c.setLineWidth(weight);c.line(left,y,right,y)
    def value(cell,x,y):
        if cell:
            label=f"{cell['value_pct']:.1f}"
            text(label,x-1,y,8,center=True)
            text(cell['cohort'],x+pdfmetrics.stringWidth(label,'Paper',8)/2,y+3,5)
        else:text('—',x,y,8,center=True)
    def find(model,method,col):return measurements.get((name,model,method,col['key']))
    top=height-20
    text(spec['title'],left,top,10,True)
    y=top-10;line(y,1)
    text('Method',left+3,y-23,9,True)
    for i,m in enumerate(models):
        mid=start+(i+.5)*model_w
        text(m['display'].replace('\n',' '),mid,y-13,8.5,True,True)
        pair_labels=['FPQ TPR ↑','NFP FPR ↓'] if detection else ['FPQ ↑','NFP / TPQ ↑']
        for j,label in enumerate(pair_labels):text(label,start+(i+j/2+.25)*model_w,y-26,7.4,center=True)
    y-=34;line(y,.7)
    # Build LaTeX in the same horizontal structure.
    def esc(s):
        for a,b in [('&',r'\&'),('_',r'\_'),('%',r'\%'),('↑',r'$\uparrow$'),('↓',r'$\downarrow$'),('≥',r'$\geq$'),('—',r'\textemdash')]:s=s.replace(a,b)
        return s
    def tr(cells):return ' & '.join(cells)+r' \\'
    def texval(cell):return f"{cell['value_pct']:.1f}"+r'\textsuperscript{'+cell['cohort']+'}' if cell else r'\textemdash'
    ncols=1+2*len(models)
    latex=[r'\begin{table*}[t]',r'\centering',r'\scriptsize',r'\setlength{\tabcolsep}{3pt}',r'\resizebox{\textwidth}{!}{%',r'\begin{tabular}{l'+'c'*(ncols-1)+'}',r'\toprule',tr([r'\multirow{2}{*}{Method}']+[r'\multicolumn{2}{c}{'+esc(m['display'].replace('\n',' '))+'}' for m in models]),tr(['']+[esc(x) for _ in models for x in pair_labels]),r'\midrule']
    for kind,label,method,columns in entries:
        if kind=='band':
            c.setFillColorRGB(.94,.95,.96);c.rect(left,y-band_h,right-left,band_h,stroke=0,fill=1);c.setFillColorRGB(0,0,0)
            text(label,left+3,y-12,8.5,True)
            latex += [r'\midrule',r'\multicolumn{'+str(ncols)+r'}{l}{\textbf{'+esc(label)+r'}} \\']
            y-=band_h
            continue
        yy=y-row_h+3
        text(label,left+3,yy,7.8,bool(method and method.get('proposed')))
        tlabel=esc(label)
        if method and method.get('proposed'):tlabel=r'\textbf{'+tlabel+'}'
        vals=[]
        if method and method['key']=='tfidf':
            a=find('model_independent',method['key'],columns[0]);b=find('model_independent',method['key'],columns[1])
            txt=f"Shared detector: FPQ {a['value_pct']:.1f}% / NFP {b['value_pct']:.1f}% (H)" if a and b else 'Shared detector: not measured'
            text(txt,(start+right)/2,yy,8,center=True)
            latex.append(tr([tlabel,r'\multicolumn{'+str(ncols-1)+'}{c}{'+esc(txt)+'}']))
        else:
            for i,m in enumerate(models):
                supported=(method is None or method.get('models') is None or m['id'] in method['models'])
                if kind=='qa':
                    cell=find(m['id'],method['key'],columns[0]) if method else None
                    if supported:value(cell,start+(i+.5)*model_w,yy)
                    else:text('n/a',start+(i+.5)*model_w,yy,7,center=True)
                    vals.append(r'\multicolumn{2}{c}{'+(texval(cell) if supported else 'n/a')+'}')
                else:
                    for j,col in enumerate(columns):
                        cell=find(m['id'],method['key'],col)
                        if supported:value(cell,start+(i+j/2+.25)*model_w,yy)
                        else:text('n/a',start+(i+j/2+.25)*model_w,yy,7,center=True)
                        vals.append(texval(cell) if supported else 'n/a')
            latex.append(tr([tlabel]+vals))
        y-=row_h
    line(y,1)
    for i,note in enumerate(spec['notes']):text(note,left,y-15-i*9,7)
    assert y-15-(len(spec['notes'])-1)*9>5, 'Footer outside page'
    c.save()
    latex += [r'\bottomrule',r'\end{tabular}}',r'\caption{'+esc(spec['title'])+'. '+esc(' '.join(spec['notes']))+'}',r'\label{tab:'+name+'}',r'\end{table*}']
    (root/f'{name}.tex').write_text('\n'.join(latex)+'\n')
    subprocess.run(['pdftoppm','-png','-r','150','-singlefile',str(root/f'{name}.pdf'),str(root/name)],check=True)
    return {'width':width,'height':height,'visual_rows':len(entries)}
