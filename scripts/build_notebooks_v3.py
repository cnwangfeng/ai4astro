#!/usr/bin/env python3
"""Build student/teacher notebooks from the auditable per-chapter experiment code."""
from pathlib import Path
import re,nbformat
ROOT=Path(__file__).resolve().parents[1]
TOPICS={
1:('从数据记录到任务','鸢尾花：一行一朵花，四项尺寸为输入，物种为目标；预先规定的简单阈值不是由数据训练出的算法。','比较多数类与给定阈值规则，解释读取数据、训练规则和预测的区别。','30条验证中的简单规则表现好，不等于所有类别问题都容易；本实验只把任务和评价对象讲清。'),
2:('表格、缺失与尺度','葡萄酒化学分析：178条、13项测量、三个来源类别；缺失只在副本上人为加入。','核对行身份、训练中位数和标准化。追踪哪一步学习了数据参数。','训练缺失指示、填补值和尺度均需保存；成功读入CSV不自动保证合并和标签对应正确。'),
3:('目标、梯度与稳定性','三点直线和单参数二次函数均为人工教学构造，不是观测发现。','比较解析梯度与有限差分，计算曲率与学习率上限，区分参数和损失轨迹。','二次目标的稳定上限依赖归一化及特征尺度；本例的边界不能直接复制到任意非凸网络。原生详细实验另在chapter03_material。'),
4:('评价的是新记录还是新对象','Wine用于基线评价；重复对象模拟给每个对象随机标签，明确构造不可向新对象学习的任务。','比较按行与按对象的验证，写清两个分数对应的部署问题。','按行接近满分是共享对象信息造成的，并不是算法发现了随机标签规律。交叉验证折间波动不是自动的置信区间。'),
5:('线性模型的解、复杂度和收缩','三点OLS、相关特征和300条非线性响应均为人工演示。住房实际数据为单独下载的扩展，不被此处替代。','核对OLS，比较多项式阶数和Ridge/Lasso单参数收缩，分析残差和验证表现。','高阶训练误差较低不保证验证最佳；相关变量可导致系数不稳而预测相似。目标函数约定必须与系数表一致。'),
6:('概率、损失与阈值','Wine为真实标准分类数据；八条概率—标签记录是为了完整统计阈值效果的人工表。','手算一次逻辑回归更新，逐阈值统计TP/FP/FN/TN，再做Wine统一验证。','固定概率下改阈值不重新训练；三阈值可以准确率相同但召回和精确率不同。类别概率的列顺序应核对classes_。'),
7:('距离是建模选择','Wine有不同量级的测量；高维距离集中实验采用明确的独立高斯生成点。','固定折比较标准化与K值，再改变维数检验平方距离的均值与相对波动。','标准化改善本实验不证明所有特征都应等权。高维结论受生成分布假设约束；不按最终测试挑K。'),
8:('把决策树完整算出来','十点分类、六点回归和九点网格是人工算例；真实鸢尾花的树图和表使用同一90/30开发划分。','全枚举分类阈值、核对回归SSE、验证贪心反例，再生成深度和剪枝结果。','最佳分类中点为2.9；两候选比较不能声称全局枚举最优。验证曲线平台是实际结果，不强行解释为已发生下降。'),
9:('为什么平均多个模型','Wine比较单树、Bagging和森林；预测方差公式通过受控相关误差模拟检查。','分开改变树数和特征随机性，报告各次种子结果、OOB及验证用途。','降低误差相关可提高平均收益，但也可能削弱单树。OOB是行级未采样，不自动满足新对象泛化。'),
10:('提升在逐步改变什么','四点残差例子与固定300条人工响应；不把它们写成真实房价或风洞实验。','追踪两轮预测，比较学习率—轮数曲线与森林/单树。','平方损失的残差拟合是负梯度的特例；选择最低验证轮数已属于模型选择。'),
11:('间隔与核的数值含义','二维人工点用于间隔示意，Wine用于C和gamma的少量联合验证。','核对显式二次映射，再解释参数组合的不同验证表现。','大gamma不等于更准确，C与gamma相互作用；图形演示与全特征分类的任务不同。'),
12:('网络表示、反向传播与训练诊断','四点XOR和单隐藏单元手算是人工构造；Wine网络与固定验证样本比较。','计算网络输出和梯度，检查有限差分；固定数据比较宽度、随机初值及训练曲线。','所有宽度准确率相同仍可有不同概率损失和迭代次数。表达一个函数不证明优化必然找到它。保留收敛警告并解释。'),
13:('从数据求主方向','非共线四点用于协方差与特征向量；Wine用于降维后分类；Digits用于训练拟合后的重构。','计算PCA方向、丢失SSE与解释方差，并比较后续分类而非只画投影。','解释方差与任务信息不同；均方误差的分母是样本还是像素须写清。PCA与尺度都放进每个训练折。'),
14:('聚类目标与自然分组','一维Lloyd例子、人工双月数据、已有标签隐藏后的标准数据分析。','核对局部稳定解，比较K均值/层次/DBSCAN及不同指标。','双月中轮廓系数较高的方法不一定更符合参考分组。噪声标签、簇号和科学身份不是同一个概念。'),
15:('异常排序与阈值','正常参考、正常查询和20条远处异常均由固定随机过程生成，注入位置已知。','分别计算异常分数、阈值、候选数、误报和检出；核对LOF局部密度例子。','远处人工异常易被找出，不能据此声称真实异常检出率100%。阈值预算不是自然界异常比例。'),
16:('结果解释的证据边界','固定人工响应模型的验证预测，用于分组残差、置换和条件bootstrap。','检查按输入区间的误差，区分指标区间、单对象预测区间与训练不确定性。','该bootstrap固定模型与预测表，不包括重训或仪器变化；置换重要性不是因果效应。'),
17:('完整物理标定项目','全部为人工物理情景：90个批次、每批8条、共享批次偏移；不是实验室观测。','按批次划分，开发比较基线与模型，冻结后评价18个未见批次。','二次模型在该生成机制中占优有明确背景，但不能据此认定真实仪器服从该式。不要将隐藏真值或偏移用于预测。'),
'ogle':('真实天文目录节选项目','OGLE OCVS大麦哲伦云基频造父变星cepF.dat前90条；取ID、I、V和周期，缺失保持。87条可用，69开发、18保留。','比较均值、原周期直线和对数周期直线，冻结后检查18条保留对象及最大残差。','固定节选不是总体随机代表样本；这里拟合视星等，不是给出完整巡天绝对光度定律。来源、字段与引用见data/OGLE_excerpt_README.md。')}
SETUP='''from pathlib import Path
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
root = next((p for p in [Path.cwd(), *Path.cwd().parents]
             if (p / "experiments/common.py").exists()
             and (p / "main.tex").exists()), None)
if root is None:
    raise RuntimeError("Open this notebook inside the extracted textbook project.")
sys.path.insert(0, str(root / "experiments"))
print("Project:", root)
'''
N=ROOT/'notebooks';N.mkdir(exist_ok=True)
# Remove superseded 34 top-level notebooks: their source is retained in the input release,
# but this release must not expose mutually inconsistent classroom instructions.
for p in N.glob('*.ipynb'):p.unlink()
for key,(topic,data,task,discussion) in TOPICS.items():
    stem=f'ch{key:02}' if isinstance(key,int) else 'project_ogle'
    source=(ROOT/'experiments'/f'{stem}.py').read_text()
    source=re.sub(r"^if __name__\s*==\s*['\"]__main__['\"]:.*$",'',source,flags=re.M).rstrip()
    chname=f'第{key}章' if isinstance(key,int) else '第17章天文项目'
    figure_prefix=f'v3_ch{key:02}_' if isinstance(key,int) else 'v3_project_ogle_'
    for role in ['student','teacher']:
        cells=[nbformat.v4.new_markdown_cell(f'# {chname}：{topic}\n\n**{"学生实验" if role=="student" else "教师参考"}｜全书深化稿**\n\n## 背景与数据\n{data}\n\n## 本次要回答的问题\n{task}\n\n先写下预判，再运行，最后用实际结果解释；不需要结果符合预期才算成功。')]
        cells.append(nbformat.v4.new_code_cell(SETUP))
        cells.append(nbformat.v4.new_markdown_cell('## 可检查的完整实验代码\n\n下方与对应`experiments`脚本一致。共享的`common.py`只提供固定数据划分、保存表图和绘图设置；模型、目标、参数及比较规则在此明确写出。阅读函数后，再运行下一单元执行实验。'))
        cells.append(nbformat.v4.new_code_cell(source))
        cells.append(nbformat.v4.new_code_cell('result = run()\nif result is not None:\n    print(result)'))
        cells.append(nbformat.v4.new_markdown_cell('## 检查结果文件与图\n\n下面显示该实验保存的图；所有坐标、样本与数值条件应与正文一致。数据表保存在`results/`。修改代码后，请另存结果或注明变化，不把新配置的输出与正文旧数值混用。'))
        cells.append(nbformat.v4.new_code_cell(f'''from IPython.display import display, Image
files = sorted((root / "figures").glob("{figure_prefix}*.png"))
for path in files:
    print(path.name)
    display(Image(filename=str(path), width=720))
if not files:
    print("This experiment uses tables rather than a new figure.")'''))
        if role=='student':
            cells.append(nbformat.v4.new_markdown_cell(f'## 运行后的分析\n\n解释一个与你的预判一致的结果，再解释一个需要进一步检查的现象。指出训练、验证和测试各使用哪些记录。\n\n{task}\n\n**提交要求：**保存可顺序运行的Notebook、一个关键结果表、必要图形，以及150—300字的机制与限制解释。完成正文D题中与本实验相关的计算。不得仅提交最高分截图。'))
        else:
            cells.append(nbformat.v4.new_markdown_cell(f'## 教师解释与核对\n\n{discussion}\n\n完整手算解答见`teacher_manual.pdf`。如果结果与这里的参考不同，先检查数据版本、划分、软件环境、目标归一化和随机状态，不要求学生人为调成相同分数。\n\n**评分顺序：**问题与字段角色、数据划分、算法机制、实际结果分析、证据边界。代码能运行但无法说明一项输出的含义，不算完成理解目标。'))
        nb=nbformat.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'},'textbook_version':'3.0-depth','role':role})
        nbformat.write(nb,N/f'{stem}_{role}.ipynb')
print('Created',len(list(N.glob('*.ipynb'))),'notebooks')
