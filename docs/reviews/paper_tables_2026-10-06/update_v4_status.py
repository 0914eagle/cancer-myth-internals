"""Put auditable Korean progress labels directly in the existing three table layouts."""
import collections,csv,datetime,hashlib,html,json
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[2]
RUN=ROOT/'results/paper_baselines_v4/main';GRADES=ROOT/'results/paper_baselines_v4/haiku'
OUT=BASE/'current_v4';OUT.mkdir(exist_ok=True)
MODELS=[('qwen25','Qwen2.5 7B'),('gemma3','Gemma 3 12B'),('luna6','GPT-6 Luna')]
METHODS=[('plain','Plain'),('balanced','Balanced instruction'),('cot','CoT'),('prewome','PreWoMe'),('extract_verify','Extract + Verify'),('tfidf_gate','TF-IDF text gate'),('direct_gate','Direct gate'),('cot_gate','CoT gate'),('probe_gate','Probe gate'),('gepa_gate','GEPA-gate'),('gepa','GEPA'),('gepa_both','GEPA (both stages)'),('ours','Ours')]
DETECT=[('tfidf_gate','TF-IDF text'),('direct_gate','Direct gate'),('cot_gate','CoT gate'),('probe_gate','Probe gate'),('gepa_gate','GEPA-gate')]
FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
BOLD='/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'

def read(path):return json.loads(path.read_text())
def rows(folder):return [read(p) for p in folder.rglob('*.json')]
def digest(value):return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True).encode()).hexdigest()

def capture():
    b=read(ROOT/'results/paper_baselines_v4/protocol/bundle.json');look={r['id']:r for r in b['rows']}
    expected={d:{y:sum(look[i]['label']==y for s in b['splits'] if s['dataset']==d for i in s['test']) for y in (1,0)} for d in ('cancer_myth','crepe')}
    snapshot={'captured_at_kst':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat(),'bundle_sha256':b['content_sha256'],'cells':{},'counts':{}}
    qa_sizes=collections.Counter(r['dataset'] for r in b['qa'])
    for model,_ in MODELS:
        root=RUN/model;progress=read(root/'generation_progress.json') if (root/'generation_progress.json').exists() else {}
        counters=collections.defaultdict(collections.Counter)
        for table in ('table1','table12','table3'):
            for r in rows(root/table):
                key=(table,r['dataset'],r['method'],r.get('label'),r.get('source',''),r.get('split',''))
                counters[key]['generated']+=1
                counters[key]['yes']+=r.get('verdict')==1
                counters[key]['correct']+=r.get('correct',False)
        for path in (GRADES/model/'table12').rglob('*.json'):
            r=read(path);source=RUN/path.relative_to(GRADES)
            if not source.exists() or r['id'] not in look:continue
            answer=read(source);judge=read(ROOT/'results/paper_baselines_v4/protocol/haiku.json')
            expected_key=digest({'question':answer['question'],'answer':answer['answer'],'judge':judge,'annotation':look[r['id']]})
            if r.get('evaluation_key')!=expected_key:continue
            key=('table12',r['dataset'],r['method'],r['label'],r.get('source',''),r.get('split',''))
            counters[key]['scored']+=1;counters[key]['good']+=r['rating']>=4
        for table in ('table12','table3'):
            for p in (root/'failures'/table).rglob('*.json'):
                r=read(p);rel=p.relative_to(root/'failures'/table)
                key=(table,r['dataset'],r['method'],r.get('label'),rel.parts[0],rel.parts[1])
                counters[key]['failed']+=1
        def count(table,dataset,method,y,source=None,split=None):
            total=collections.Counter()
            for key,c in counters.items():
                t,d,m,label,s,f=key
                if (t,d,m,label)==(table,dataset,method,y) and (source is None or s==source) and (split is None or f==split):total.update(c)
            return total
        for table,methods in (('table1',DETECT),('table2',METHODS)):
            for method,_ in methods:
                for dataset in expected:
                    for y in (1,0):
                        key='|'.join([table,model,method,dataset,str(y)])
                        c=count('table1' if table=='table1' else 'table12',dataset,method,y);n=expected[dataset][y]
                        snapshot['counts'][key]=dict(c,expected=n)
                        if method=='ours':text='후속 실험'
                        elif method=='probe_gate' and model=='luna6':text='해당 없음'
                        elif table=='table1':
                            text=f"완료\n{100*c['yes']/n:.1f}%" if c['generated']==n else f"판정 중\n{c['generated']}/{n}" if c['generated'] else '판정 필요'
                        elif c['generated']+c['failed']<n:
                            active=progress.get('state')=='running' and progress.get('stage')=='table12' and progress.get('method')==method and progress.get('dataset')==dataset
                            label='생성 중' if active else '추가 생성 필요' if c['generated'] else '생성 필요'
                            text=label+(f"\n{c['generated']}/{n}" if active or c['generated'] else '')
                        elif c['scored']+c['failed']<n:text=f"채점만 남음\n{c['scored']}/{c['generated']}"
                        else:text=f"완료\n{100*c['good']/n:.1f}%"
                        snapshot['cells'][key]=text
        for method,_ in METHODS:
            for dataset,n in qa_sizes.items():
                key='|'.join(['table3',model,method,dataset,'qa'])
                if method=='ours':text='후속 실험'
                elif method=='probe_gate' and model=='luna6':text='해당 없음'
                else:
                    shared=method in ('plain','balanced','cot','direct_gate','cot_gate')
                    contexts=[('shared','fixed')] if shared else [('cancer_myth','fixed' if method in ('prewome','extract_verify') else 'fold0'),('crepe','fixed' if method in ('prewome','extract_verify') else 'official')]
                    cs=[count('table3',dataset,method,None,s,f) for s,f in contexts]
                    if all(c['generated']+c['failed']==n for c in cs):
                        values='/'.join(f"{100*c['correct']/n:.1f}%" for c in cs)
                        fails=sum(c['failed'] for c in cs)
                        text=('완료' if not fails else f'처리 완료·실패 {fails}')+'\n'+values
                    elif any(c['generated'] for c in cs):text='추가 생성 필요\n'+str(sum(c['generated'] for c in cs))+'/'+str(n*len(cs))
                    else:text='생성 필요'+('\n(2개 설정)' if not shared else '')
                    snapshot['counts'][key]={'systems':[dict(c) for c in cs],'expected_per_system':n}
                snapshot['cells'][key]=text
    return snapshot


def render(snap,name,methods,groups):
    cw=140 if name!='table3' else 138;mw=294;left=28;top=88;head=122;rh=76
    columns=[(dataset,metric,label) for dataset,_,metrics in groups for metric,label in metrics]
    width=left*2+mw+cw*len(columns)*len(MODELS);height=top+head+rh*len(methods)+130
    im=Image.new('RGB',(width,height),'white');draw=ImageDraw.Draw(im);svg=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}"><rect width="100%" height="100%" fill="white"/>']
    def text(value,x,y,size=21,bold=False,anchor='mm',color='#17212b'):
        font=ImageFont.truetype(BOLD if bold else FONT,size)
        lines=value.split('\n')
        for j,line in enumerate(lines):
            yy=y+(j-(len(lines)-1)/2)*(size+5)
            tw,th=draw.textsize(line,font=font)
            dx=x-tw/2 if anchor=='mm' else x-tw if anchor=='rm' else x
            draw.text((dx,yy-th/2),line,font=font,fill=color)
            svg.append(f'<text x="{x}" y="{yy}" font-family="Noto Sans CJK KR,sans-serif" font-size="{size}" font-weight="{700 if bold else 400}" text-anchor="{"middle" if anchor=="mm" else "end" if anchor=="rm" else "start"}" dominant-baseline="central" fill="{color}">{html.escape(line)}</text>')
    def line(x1,y1,x2,y2,w=1,color='#bdc7ce'):
        draw.line((x1,y1,x2,y2),fill=color,width=w);svg.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{w}"/>')
    def box(x,y,w,h,color):
        draw.rectangle((x,y,x+w,y+h),fill=color);svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{color}"/>')
    titles={'table1':'Table 1. 전제 탐지 — 진행 상태','table2':'Table 2. 최종 답변 — 진행 상태','table3':'Table 3. 일반 QA 보존 — 진행 상태'}
    text(titles[name],left,31,29,True,'lm');text('기준 시각: '+snap['captured_at_kst'][:19].replace('T',' ')+' KST',width-left,64,17,False,'rm')
    box(left,top,width-2*left,head,'#edf2f5');text('방법',left+mw/2,top+head/2,23,True)
    for mi,(model,label) in enumerate(MODELS):
        x=left+mw+mi*len(columns)*cw;text(label,x+len(columns)*cw/2,top+23,25,True)
        offset=0
        for dataset,label,metrics in groups:
            gw=len(metrics)*cw;text(label,x+(offset+len(metrics)/2)*cw,top+65,20,True)
            for j,(metric,mlabel) in enumerate(metrics):text(mlabel,x+(offset+j+.5)*cw,top+101,18)
            offset+=len(metrics)
        line(x,top,x,top+head+rh*len(methods),2)
    line(left,top+head,width-left,top+head,2)
    csvrows=[['Method']+[f'{label} / {dataset} / {metric}' for _,label in MODELS for dataset,metric,_ in columns]]
    for ri,(method,label) in enumerate(methods):
        y=top+head+ri*rh;text(label,left+10,y+rh/2,21,method=='ours','lm');row=[label]
        for mi,(model,_) in enumerate(MODELS):
            for ci,(dataset,metric,_) in enumerate(columns):
                x=left+mw+(mi*len(columns)+ci)*cw;value=snap['cells']['|'.join([name,model,method,dataset,metric])]
                fill='#e7f3e8' if '완료' in value else '#fff3d8' if '채점만' in value else '#e9f2fc' if '생성 중' in value or '판정 중' in value else '#f5f6f7'
                box(x+2,y+2,cw-4,rh-4,fill);text(value,x+cw/2,y+rh/2,17 if len(value.split('\n')[0])>8 else 19)
                row.append(value.replace('\n',' '))
        csvrows.append(row);line(left,y+rh,width-left,y+rh)
    notes=['Cancer-Myth: 577 FPQ / 147 NFP. CREPE: 공통 test 500 FPQ / 500 NFP.' if name!='table3' else '분모: MedQA 1,273 / PubMedQA 1,000 / Medbullets 308 / MMLU 14,042 / GSM8K 1,319.',
           'Table 2는 Haiku 4점 이상 비율. 생성/채점 분자는 완료 문항 수이며 부분 성능을 뜻하지 않습니다.' if name=='table2' else 'Table 1은 정답 라벨로 직접 평가합니다. 별도 LLM 채점이 필요 없습니다.' if name=='table1' else '일반 QA는 정답 비교로 자동 채점. 2개 설정 = Cancer-Myth fold0 / CREPE 학습·예시 버전.',
           'TF-IDF는 모델 공통 판정입니다. 해당 없음 = Luna 내부 Probe 미지원.' if name=='table1' else '후속 실험 = Ours는 이번 baseline 실행 범위 밖. 해당 없음 = Luna 내부 Probe 미지원.']
    for i,note in enumerate(notes):text(note,left,top+head+rh*len(methods)+27+i*29,18,False,'lm')
    svg.append('</svg>');(OUT/(name+'.svg')).write_text('\n'.join(svg));im.save(OUT/(name+'.png'))
    with (OUT/(name+'.csv')).open('w',newline='') as f:csv.writer(f,lineterminator='\n').writerows(csvrows)
    return titles[name]


if __name__=='__main__':
    snap=capture();(OUT/'status_snapshot.json').write_text(json.dumps(snap,ensure_ascii=False,indent=2)+'\n')
    fpqa=[('cancer_myth','Cancer-Myth',[('1','FPQ 교정 ↑'),('0','NFP 보존 ↑')]),('crepe','CREPE',[('1','FPQ 교정 ↑'),('0','NFP 보존 ↑')])]
    render(snap,'table1',DETECT,[(d,l,[(m,'FPQ 탐지 ↑' if m=='1' else 'NFP 오탐 ↓') for m,_ in metrics]) for d,l,metrics in fpqa])
    render(snap,'table2',METHODS,fpqa)
    render(snap,'table3',METHODS,[(d,l,[('qa','정확도 ↑')]) for d,l in [('medqa','MedQA'),('pubmedqa','PubMedQA'),('medbullets','Medbullets'),('mmlu','MMLU'),('gsm8k','GSM8K')]])
    lines=['# 세 모델 baseline 진행 상태','',f"기준: {snap['captured_at_kst']}",'','빈칸 대신 자연어 상태를 표시했습니다. 완료된 셀만 성능 수치를 표시합니다.','']
    for n in (1,2,3):lines += [f'## Table {n}',f'![Table {n}](table{n}.png)',f'[CSV](table{n}.csv) · [SVG](table{n}.svg)','']
    (OUT/'README.md').write_text('\n'.join(lines).rstrip()+'\n')
    print(OUT)
