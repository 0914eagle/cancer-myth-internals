#!/usr/bin/env python3
"""Build a question/label-independent NCI corpus; no benchmark data is read."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen
from bs4 import BeautifulSoup

BASE = 'https://www.cancer.gov'
SEEDS = ['/types', '/about-cancer/treatment', '/about-cancer/treatment/side-effects',
         '/about-cancer/coping', '/about-cancer/causes-prevention',
         '/about-cancer/screening', '/about-cancer/diagnosis-staging/prognosis']

def fetch(url, out):
    key = hashlib.sha256(url.encode()).hexdigest()
    path = out/'pages'/f'{key}.json'
    if path.exists():
        return json.loads(path.read_text())
    req = Request(url, headers={'User-Agent': 'CancerMythResearch/1.0'})
    with urlopen(req, timeout=60) as r:
        raw = r.read(); final_url = r.url
    soup = BeautifulSoup(raw, 'html.parser')
    main = soup.find('main')
    if main is None:
        raise ValueError(f'No main element: {url}')
    links = sorted({urljoin(BASE, a['href']).split('#')[0] for a in main.select('a[href]')
                    if urlparse(urljoin(BASE, a['href'])).netloc == 'www.cancer.gov'})
    for tag in main.select('script,style,nav,footer,form'):
        tag.decompose()
    lines = [t.get_text(' ', strip=True) for t in main.select('h1,h2,h3,h4,p,li')]
    text = '\n'.join(dict.fromkeys(x for x in lines if x))
    data = dict(url=url, final_url=final_url, title=soup.title.get_text(strip=True) if soup.title else url,
                text=text, links=links, fetched_utc=datetime.now(timezone.utc).isoformat(),
                html_sha256=hashlib.sha256(raw).hexdigest())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')
    return data

def main():
    p=argparse.ArgumentParser(); p.add_argument('--out', type=Path, required=True); a=p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    if (a.out/'manifest.json').exists():
        print('Corpus already frozen'); return
    errors=[]; pages={}
    def batch(urls):
        def safe(url):
            try: return fetch(url,a.out)
            except Exception as e:
                return {'url':url,'error':f'{type(e).__name__}: {e}'}
        with ThreadPoolExecutor(max_workers=4) as pool:
            for r in pool.map(safe, sorted(urls)):
                if 'error' in r: errors.append(r)
                else: pages[r['url']]=r
        print(f'Fetched {len(pages)} pages; errors {len(errors)}',flush=True)
    batch({BASE+s for s in SEEDS})
    if BASE+'/types' not in pages: raise RuntimeError('NCI index unavailable')
    # All type links and supportive-care links from the fixed indices, not case-specific sources.
    level1={u for r in pages.values() for u in r['links']
            if urlparse(u).path.startswith(('/types/','/about-cancer/')) and not urlparse(u).query
            and not urlparse(u).path.endswith('.pdf')}
    batch(level1-set(pages))
    # Patient PDQ pages linked from those overview pages. Exclude professional variants.
    level2={u for r in list(pages.values()) for u in r['links']
            if urlparse(u).path.endswith('-pdq') and ('/patient/' in u or '/patient-' in u)
            and not urlparse(u).query}
    batch(level2-set(pages))
    chunks=[]
    for url,r in sorted(pages.items()):
        words=r['text'].split()
        for start in range(0,len(words),220):
            part=' '.join(words[start:start+280])
            if len(part.split())<30: continue
            chunks.append({'id':hashlib.sha256(f'{url}#{start}'.encode()).hexdigest()[:20],
                           'url':url,'title':r['title'],'text':part})
    if len(pages)<50 or len(chunks)<200: raise RuntimeError('Insufficient corpus; not freezing')
    target=a.out/'chunks.jsonl'
    target.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in chunks))
    manifest={'source':'NCI fixed indices, linked overview pages, linked patient PDQ pages',
              'seeds':SEEDS,'uses_questions_labels_annotations':False,'pages':len(pages),
              'chunks':len(chunks),'errors':errors,'urls':sorted(pages),
              'chunks_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
              'builder_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (a.out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:manifest[k] for k in ['pages','chunks','chunks_sha256']}))

if __name__=='__main__': main()
