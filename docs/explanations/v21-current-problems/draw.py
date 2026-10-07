"""Chinese explanatory figure based only on sealed v21 numerical evidence."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import Rectangle, FancyBboxPatch

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
DATA=ROOT/'docs/results/lite-aniso-mainline/v21'
FONT=Path('/tmp/mpm-explain-NotoSansCJKsc-Regular.otf')
fm.fontManager.addfont(str(FONT))
plt.rcParams.update({'font.family':fm.FontProperties(fname=FONT).get_name(), 'font.size':12,
                     'axes.unicode_minus':False,'svg.fonttype':'path','savefig.facecolor':'#F4F6FA'})
a=json.loads((DATA/'final-spatial-acceptance.json').read_text())
r=json.loads((DATA/'reference-self-checks.json').read_text())
c=a['cases']['adaptive/gain/round6']
INK='#183047'; MUTED='#586B7B'; BLUE='#327BA3'; TEAL='#158C8C'; ORANGE='#D87827'; PURPLE='#8062AB'; RED='#BE514B'
fig=plt.figure(figsize=(12,13.3),facecolor='#F4F6FA')
def txt(x,y,s,size=12,color=INK,weight='normal',ha='left',**kw):
    return fig.text(x,y,s,fontsize=size,color=color,weight=weight,ha=ha,va='center',**kw)
def card(x,y,w,h):
    fig.add_artist(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.012,rounding_size=0.014',transform=fig.transFigure,facecolor='white',edgecolor='#DEE5EC',lw=.9,zorder=-1))

txt(.05,.964,'当前卡在哪里？',27,weight='bold')
txt(.05,.928,'F45 斜向纤维：静态一致性检查通过，空间应力仍不够准确',14,MUTED)
card(.045,.631,.91,.264)
txt(.064,.868,'① 夹具限制局部运动，斜向纤维需要协调变形',16,weight='bold')
ax=fig.add_axes([.063,.699,.874,.133]);ax.set_xlim(0,10);ax.set_ylim(0,2);ax.axis('off')
body=Rectangle((1,.5),8,1,facecolor='#E5F2F3',edgecolor=INK,lw=1.5);ax.add_patch(body)
for x in (1,7.667):ax.add_patch(Rectangle((x,.5),1.333,1,facecolor='#B6C1CD',edgecolor='none'))
for x in (2.333,7):ax.add_patch(Rectangle((x,.5),.667,1,facecolor='#FFE1B5',edgecolor='none'))
for x in np.arange(.4,9.4,.46):
    line,=ax.plot([x,x+1],[.5,1.5],color=TEAL,lw=1.1,alpha=.68);line.set_clip_path(body)
for x in (2.333,7.667):ax.plot([x,x],[.36,1.62],color=ORANGE,lw=2,ls='--')
for x,t in [(1.65,'左夹持：固定'),(5,'内部：允许收缩与剪切'),(8.3,'右夹持：向右拉')]:
    ax.text(x,1.89,t,ha='center',va='center',fontsize=11.4,color=INK)
ax.annotate('',xy=(9.84,1),xytext=(9.08,1),arrowprops=dict(arrowstyle='-|>',color=INK,lw=2))
for x in (4,5,6):
    ax.annotate('',xy=(x,1.5),xytext=(x,1.73),arrowprops=dict(arrowstyle='->',color=BLUE,lw=1.2))
    ax.annotate('',xy=(x,.5),xytext=(x,.27),arrowprops=dict(arrowstyle='->',color=BLUE,lw=1.2))
ax.text(2.67,.1,'夹持过渡',color=ORANGE,ha='center',fontsize=11)
ax.text(7.33,.1,'夹持过渡',color=ORANGE,ha='center',fontsize=11)
ax.text(5,.06,'斜线：45° 纤维方向',ha='center',fontsize=11,color=TEAL)
txt(.067,.682,r'$\varepsilon_{aa}=\frac{1}{2}(\varepsilon_{xx}+\varepsilon_{yy}+2\varepsilon_{xy})$',19)
txt(.535,.682,'纵横伸缩 + 剪切，都会改变纤维长度',12.4)
txt(.067,.650,'夹持过渡需要更丰富的局部变形表达；当前内部应力也仍有较大差异。',12,MUTED)

card(.045,.285,.438,.307); card(.517,.285,.438,.307)
txt(.064,.566,'② 局部应力仍不够准',16,weight='bold')
txt(.536,.566,'③ 当前网格自身也有局限',16,weight='bold')
txt(.064,.538,'v21 与最终 Q4 参考的区域应力差',11.5,MUTED)
txt(.536,.538,'不同空间与同一 Q4 参考的全域应力差',11.5,MUTED)

def bars(rect,labels,vals,colors,xmax,ticks):
    ax=fig.add_axes(rect)
    y=np.arange(len(vals))[::-1]
    ax.barh(y,vals,height=.5,color=colors,zorder=3)
    ax.set_yticks(y,labels,fontsize=11.2)
    ax.set_xlim(0,xmax);ax.set_ylim(-.65,len(vals)-.35)
    ax.set_xticks(ticks);ax.tick_params(axis='x',labelsize=10,color='#CCD4DD')
    ax.tick_params(axis='y',length=0,pad=7)
    for spine in ax.spines.values():spine.set_visible(False)
    ax.grid(axis='x',color='#E5EBF0',zorder=0)
    ax.axvline(2,color=RED,ls='--',lw=1.4,zorder=4)
    ax.set_xlabel('相对差异（%）',fontsize=10.5,color=MUTED,labelpad=5)
    for yy,v in zip(y,vals):ax.text(v+xmax*.024,yy,f'{v:.2f}%',va='center',fontsize=11.3,color=INK,weight='bold')
    return ax

bars([.145,.382,.294,.136],['夹持区','内部','全域'],[100*c['regions'][k]['stress_relative'] for k in ('grip','interior','global')],[ORANGE,BLUE,TEAL],62,[0,20,40,60])
bars([.66,.382,.265,.136],['v20：150 函数','v21：144 函数','Q2 全部自由度'],[100*a['cases'][k]['regions']['global']['stress_relative'] for k in ('v20-local150','adaptive/gain/round6','full-graded-Q2-limit')],['#92AAC1',TEAL,PURPLE],144,[0,40,80,120])
txt(.064,.321,f"反力差 {100*c['reaction_relative']:.2f}%，全域纤维应变差 {100*c['regions']['global']['fiber_strain_relative']:.2f}%。",11.8)
txt(.064,.297,'总力接近，不能替代局部应力验收。',11.8,MUTED)
txt(.536,.321,'局部函数和网格表达都还需要改善。',11.8)
txt(.536,.297,'三项差异不能相减来分摊误差来源。',11.2,MUTED)

card(.045,.081,.91,.161)
txt(.064,.216,'④ 参考解这把“尺子”，在夹持区也还没完全稳定',16,weight='bold')
for x,k,label,color in [(.215,'interior','内部',TEAL),(.5,'global','全域',ORANGE),(.79,'grip','夹持区',ORANGE)]:
    v=100*r['pairs']['p_q3_q4']['regions'][k]['stress_relative']
    txt(x,.165,f'{v:.2f}%',25,color,weight='bold',ha='center')
    txt(x,.130,label+'参考应力差',12,MUTED,ha='center')
txt(.064,.097,'以上为最细同网格 Q3/Q4 对照；内部通过本轮 h/p 门槛，全域与夹持区未通过。',11.2,MUTED)
txt(.051,.047,'上图是机制示意，不是应力云图。柱图红虚线为 2% 门槛；应力与应变差按体积范数计算。',10.2,MUTED)
txt(.051,.025,'来源：v21 封存验收数据。范围：线性静态 F45；新空间尚未接入移动动力学。',10.2,MUTED)
fig.savefig(OUT/'current-problems.png',dpi=160)
fig.savefig(OUT/'current-problems.svg')
plt.close(fig)
print(OUT/'current-problems.png')
