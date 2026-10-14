"""Plot saved GPT candidate-dev scores, never test or main-table estimates."""
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/reader_gepa_analysis_mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

P = Path(__file__).resolve().parents[1] / 'results/nla_ao_parallel_20261012/gepa_reader_pilot/cpu_analysis_20261013'
data = json.loads((P / 'candidate_scores.json').read_text())
selected = json.loads((P / 'selected_dev_scores.json').read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none'})

fig,axes=plt.subplots(1,2,figsize=(12,4.8),sharey=True)
for ax,dataset in zip(axes,['cancer_myth','crepe']):
    seed=next(r for r in data if r['dataset']==dataset and r['arm']=='gepa' and r['starting_prompt'])
    rows=[seed]+[next(r for r in selected if r['dataset']==dataset and r['arm']==arm) for arm in ['gepa','text','nla']]
    x=np.arange(4)
    for offset,field,label,color in [(-.18,'fpq_mean','FPQ correction','#4879bb'),(.18,'nfp_mean','NFP preservation','#56a78d')]:
        bars=ax.bar(x+offset,[r[field] for r in rows],.34,label=label,color=color)
        ax.bar_label(bars,fmt='%.2f',padding=3,fontsize=10)
    ax.set_xticks(x,['Plain seed','GEPA','+ Text','+ NLA'])
    ax.set_title('Cancer-Myth' if dataset=='cancer_myth' else 'CREPE',fontweight='bold')
    ax.set_ylim(0,5.75)
    ax.grid(axis='y',alpha=.2)
    ax.set_axisbelow(True)
    ax.spines[['top','right']].set_visible(False)
axes[0].set_ylabel('GPT Well score (0–5)')
fig.legend(*axes[1].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.94),ncol=2,frameon=False)
fig.suptitle('Selected candidate-dev scores — pilot only',fontsize=16,fontweight='bold')
fig.text(.5,.02,'4 FPQ + 4 NFP per dataset; selection data, not held-out results. AO unfinished.',ha='center',fontsize=10,color='#526477')
fig.tight_layout(rect=[0,.06,1,.86])
for ext in ['png','pdf','svg']:fig.savefig(P/f'selected_dev_comparison.{ext}',dpi=180,bbox_inches='tight')
plt.close(fig)

fig,axes=plt.subplots(1,2,figsize=(12,4.5),sharey=True)
for ax,dataset in zip(axes,['cancer_myth','crepe']):
    rows=[r for r in data if r['dataset']==dataset and r['arm']=='nla' and r['dev_complete']]
    for field,label,color in [('fpq_mean','FPQ','#4879bb'),('nfp_mean','NFP','#56a78d')]:
        ax.plot(range(len(rows)),[r[field] for r in rows],marker='o',color=color,label=label)
    for i,r in enumerate(rows):
        if r['selected']:ax.axvline(i,color='#9b6d28',linestyle='--',alpha=.7,label='Selected prompt')
    ax.set_xticks(range(len(rows)),['Seed' if r['starting_prompt'] else f'Pool {r["engine_pool_index"]}' for r in rows])
    ax.set_ylim(0,5.5);ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
    ax.set_title('Cancer-Myth + NLA' if dataset=='cancer_myth' else 'CREPE + NLA')
    ax.set_xlabel('Candidates with complete dev evaluation')
axes[0].set_ylabel('GPT Well score (0–5)');axes[1].legend(frameon=False)
fig.suptitle('NLA-arm candidate scores are not monotonic',fontweight='bold',fontsize=15)
fig.text(.5,.02,'Search branches and rejected trials are not shown as an improvement trajectory. Dev only; n=4 per class.',ha='center',fontsize=9,color='#526477')
fig.tight_layout(rect=[0,.06,1,.92])
for ext in ['png','pdf','svg']:fig.savefig(P/f'nla_candidate_dev_scores.{ext}',dpi=180,bbox_inches='tight')
plt.close(fig)
print(P/'selected_dev_comparison.png')
