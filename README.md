# 机器学习及其在天文与物理中的应用

广州大学天体物理中心 · 王锋

本教材面向物理学、天文学及相关理科专业本科生，系统介绍机器学习的基本概念、经典算法及其在天文与物理中的应用。全书共 17 章、119 个编号小节，按 64 学时课程设计，分为四个部分：

1. **机器学习基础**（第 1–4 章）：任务、数据表示与准备、损失函数与优化、模型评价与泛化；
2. **经典监督学习方法**（第 5–12 章）：线性回归、逻辑回归、近邻方法、决策树、随机森林、梯度提升、支持向量机、多层感知机；
3. **无标签数据的表示与结构**（第 13–15 章）：主成分分析、聚类、异常检测；
4. **模型解释与综合项目**（第 16–17 章）：模型解释、误差分析与天文物理综合项目。

每个案例从研究对象、数据含义和分析目标出发；定义、推导、可复算例题与实跑实验相互配合，算法学习始终与具体专业问题相联系。

## 仓库结构

```
main.tex                  全书编译入口（17 章，四个部分）
teacher_manual.tex        教师手册编译入口
chapters/                 17 章正文
appendices/               6 个附录（Python 速查、数据获取等）
figures/                  全部插图（正文挂载 64 幅）
data/                     实验数据、划分与字段说明
experiments/              17 章及 OGLE 节选项目脚本
results/                  实验实际生成的表格与 JSON 留档
notebooks/                36 份学生版/教师版 Notebook
docs/                     教师手册内容、参考来源、题解
scripts/                  审计、数据获取与实验辅助脚本
```

本仓库不存放编译产物 PDF；克隆后按下列方法编译生成 `main.pdf` 与 `teacher_manual.pdf`。

## 编译

需要 XeTeX 系引擎（xelatex 或 tectonic）与 CTeX 宏包：

```bash
# 方式一：tectonic（单文件引擎，宏包按需自动获取，推荐）
tectonic -X compile main.tex --outdir _build
tectonic -X compile teacher_manual.tex --outdir _build

# 方式二：TeX Live / latexmk
latexmk -xelatex -interaction=nonstopmode -halt-on-error main.tex
latexmk -xelatex -interaction=nonstopmode -halt-on-error teacher_manual.tex
```

无 latexmk 时可连跑两至三次 xelatex 使目录与交叉引用稳定。VS Code 用户安装 LaTeX Workshop 扩展后，打开 `main.tex` 直接构建即可。字体优先使用系统 Noto CJK SC 与 Linux Libertine O，缺失时自动回退到随包字体；不同系统分页可能略有差异。

## 实验环境与运行

测试环境为 Python 3.13.5、scikit-learn 1.8.0，精确依赖见 `requirements.txt` 与 `environment.lock.txt`：

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter lab
```

逐章实验脚本无隐式下载，可离线复现主线结果：

```bash
python experiments/ch08.py            # 决策树全部候选枚举与剪枝
python experiments/ch12.py            # 多层感知机数值实验
python experiments/project_ogle.py    # 真实 OGLE 节选：周期—星等项目
python scripts/run_deep_experiments.py  # 全部 17 章
```

结果写入 `results/`。核心数学与结构核查：

```bash
python scripts/audit_numerics_v3.py    # 66 项独立数值复核
python scripts/audit_structure_v3.py   # 17 章/119 节/D 题/引用完整性断言
```

## 数据的真实状态

1. 鸢尾花、葡萄酒、手写数字为 sklearn 公开标准数据，配有 CSV，可离线运行；
2. `data/ogle_lmc_f_excerpt.csv` 是真实 OGLE 目录（OCVS cepF.dat）前 90 条固定规则的少字段节选，来源、字段、缺失与限制见 `data/OGLE_excerpt_README.md`。87 条可用记录的模型比较与保留集评价已实际运行。**它不是完整巡天样本，也不是新的周期—光度标定**；
3. 教学中使用的生成数据均明确标注"人工构造/模拟"，不伪装成真实观测或仪器数据；
4. California Housing、完整 OGLE、SDSS、Gaia、Airfoil、Sonar、超导材料等外部数据不随仓库分发。需联网获取时运行 `python scripts/fetch_external_data.py --dataset <名称>`（失败即报错，不回退为其他数据）。相关章节提供背景、字段与任务设计。

## 深化要点与边界

- 全部 17 章包含机制解释、必要推导、可算例子与可检验任务；第 8 章决策树为重写版（候选全枚举、n·ΔV=SSE 恒等式、代价复杂度剪枝）；
- 每章 4 道共 68 道 D 编号深化题，教师手册（`docs/teacher_manual.tex`）逐题给出计算与分析参考；
- 由于作者水平有限，本书使用中如发现问题，欢迎提出。
