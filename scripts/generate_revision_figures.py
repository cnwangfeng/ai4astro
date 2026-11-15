"""Generate the six numerical/teaching figures added in revision 2.
Run from any directory. Uses only default Matplotlib colors; no external assets.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, Rectangle
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'figures'
OUT.mkdir(exist_ok=True)
fonts = {f.name for f in font_manager.fontManager.ttflist}
for candidate in ('Noto Sans CJK SC', 'Noto Sans CJK JP', 'Source Han Sans SC', 'SimHei'):
    if candidate in fonts:
        plt.rcParams['font.sans-serif'] = [candidate, 'DejaVu Sans']
        break
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 11

def save(fig, name):
    fig.savefig(OUT / (name+'.pdf'), bbox_inches='tight')
    fig.savefig(OUT / (name+'.png'), dpi=180, bbox_inches='tight')
    plt.close(fig)

def box(ax, x, y, w, h, text):
    ax.add_patch(FancyBboxPatch((x,y), w,h, boxstyle='round,pad=0.025', fill=False))
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=11)

def arrow(ax, a,b,label=None):
    ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','lw':1.3})
    if label:
        ax.text((a[0]+b[0])/2,(a[1]+b[1])/2+.12,label,ha='center',fontsize=9)

fig,ax=plt.subplots(figsize=(10.5,4.7)); ax.set(xlim=(0,11),ylim=(0,5));ax.axis('off')
box(ax,.2,3.2,2.5,1.15,'已有样本\n测量特征 + 已知类别')
box(ax,4.0,3.2,2.3,1.15,'训练算法\n利用配对记录建立规则')
box(ax,8,3.2,2.4,1.15,'已训练的模型\n保留学习得到的规则')
arrow(ax,(2.75,3.78),(3.95,3.78));arrow(ax,(6.35,3.78),(7.95,3.78))
box(ax,.2,.65,2.5,1.15,'新样本\n只提供测量特征')
box(ax,4,.65,2.3,1.15,'调用已训练的模型\n不重新学习参数')
box(ax,8,.65,2.4,1.15,'预测类别\n有参考答案时再核对')
arrow(ax,(2.75,1.22),(3.95,1.22));arrow(ax,(6.35,1.22),(7.95,1.22))
arrow(ax,(9.2,3.16),(6.15,1.9),'使用模型')
ax.text(5.5,4.75,'训练与应用是两个不同阶段',ha='center',fontsize=14)
save(fig,'v2_supervised_flow')

fig,ax=plt.subplots(figsize=(10,3.9))
for fold in range(3):
    for i in range(12):
        val=(i//4==fold)
        ax.add_patch(Rectangle((i,2-fold),.92,.7,fill=False,hatch='///' if val else None))
        ax.text(i+.46,2-fold+.35,str(i+1),ha='center',va='center')
    ax.text(-.3,2-fold+.35,f'第{fold+1}轮',ha='right',va='center')
ax.text(5.6,3.05,'斜线框：本轮验证记录；空白框：本轮训练记录',ha='center')
ax.text(5.6,-.5,'每条记录验证一次；每轮都重新拟合模型及需要学习参数的预处理。',ha='center',fontsize=10)
ax.set(xlim=(-1.5,12.2),ylim=(-.8,3.4)); ax.axis('off')
save(fig,'v2_cross_validation')

fig,ax=plt.subplots(figsize=(8.4,4))
x=np.arange(1,11);y=np.array([0,0,0,0,1,1,0,0,1,1])
for c,marker in [(0,'o'),(1,'^')]:
    m=y==c;ax.scatter(x[m],y[m],marker=marker,s=80,label=f'类别 {c}')
ax.axvline(4.5,ls='--',label='候选阈值 4.5')
ax.axvline(6.5,ls=':',label='候选阈值 6.5')
ax.set(xlabel='人工特征 x',ylabel='类别编号',xticks=x,yticks=[0,1],ylim=(-.2,1.5))
ax.legend(ncol=2,loc='upper center');fig.tight_layout()
save(fig,'v2_tree_split')

fig,ax=plt.subplots(figsize=(8.2,4.5))
x=np.arange(1,5); y=np.array([1,1,3,3])
ax.scatter(x,y,s=90,label='目标 y',zorder=5)
ax.plot(x,[2]*4,ls=':',marker='s',label='初始预测 F0；损失 0.5')
ax.plot(x,[1.5,1.5,2.5,2.5],ls='--',marker='o',label='第1轮 F1；损失 0.125')
ax.plot(x,[1.25,1.25,2.75,2.75],marker='^',label='第2轮 F2；损失 0.03125')
ax.set(xlabel='人工特征 x',ylabel='目标与预测',xticks=x,ylim=(.8,3.55))
ax.legend(loc='upper left',ncol=2,fontsize=9);fig.tight_layout()
save(fig,'v2_boost_steps')

fig,ax=plt.subplots(figsize=(11,4.4));ax.set(xlim=(0,12),ylim=(0,5));ax.axis('off')
box(ax,.1,1.9,1.3,1.0,'输入\nx = (1, 2)')
box(ax,2.25,2.9,2.2,1.1,'隐藏单元 1\nz1 = 1；h1 = 1')
box(ax,2.25,.8,2.2,1.1,'隐藏单元 2\nz2 = −1；h2 = 0')
arrow(ax,(1.45,2.45),(2.2,3.4));arrow(ax,(1.45,2.35),(2.2,1.35))
box(ax,6,1.9,2.2,1.0,'输出得分\ns = 2×1 − 1×0 − 0.5')
arrow(ax,(4.5,3.4),(5.95,2.6));arrow(ax,(4.5,1.35),(5.95,2.2))
box(ax,9.35,1.9,2.2,1.,'Sigmoid 输出\np ≈ 0.8176')
arrow(ax,(8.25,2.4),(9.3,2.4))
ax.text(3.35,4.45,'隐藏层使用 ReLU：负值变为0',ha='center')
ax.text(7.1,.6,'得分 s = 1.5',ha='center');fig.tight_layout()
save(fig,'v2_mlp_forward')

fig,ax=plt.subplots(figsize=(6.1,5.2))
p=np.array([[-2,-2],[-1,-1],[1,1],[2,2]])
ax.scatter(p[:,0],p[:,1],s=65,label='确定主方向的四个点')
ax.plot([-2.5,2.5],[-2.5,2.5],ls='--',label='主方向')
ax.scatter([1],[0],marker='s',s=80,label='查询点 (1, 0)')
ax.scatter([.5],[.5],marker='x',s=100,label='重构点 (0.5, 0.5)')
ax.plot([1,.5],[0,.5],ls=':')
ax.axhline(0,lw=.5);ax.axvline(0,lw=.5)
ax.set(xlabel='原特征 x1',ylabel='原特征 x2',xlim=(-2.8,3),ylim=(-2.8,3),aspect='equal')
ax.legend(loc='upper left',fontsize=9);fig.tight_layout()
save(fig,'v2_pca_projection')
print('Generated six figure pairs in',OUT)
