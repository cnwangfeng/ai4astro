# 机器学习及其在天文与物理中的应用
## 全书深化稿 v3.0（2026-11-15）

本工程是在“概念与案例教学修订版”基础上，对全部17章进行的统一深化。保留119个编号小节与64学时教学框架。正文不假定学生已有机器学习知识；先定义问题和数据，再介绍机制、必要推导、完整算例、实跑比较及科学解释。

**先阅读：** `main.pdf`；**编辑入口：** `main.tex`；**教师解答：** `teacher_manual.pdf`。不含Word文件，不附带字体文件。

## 一、编译教材

需要 XeLaTeX、CTeX、常见数学/图表宏包。推荐完整TeX Live或MiKTeX发行版。工程中的已生成图片足够直接编译，不需要先运行Python。

```bash
latexmk -xelatex -interaction=nonstopmode -halt-on-error main.tex
latexmk -xelatex -interaction=nonstopmode -halt-on-error teacher_manual.tex
```

或运行 `make all` / `bash build.sh`。没有latexmk时可分别执行两至三次xelatex，使目录与引用稳定。Overleaf将整个工程上传，选择XeLaTeX并以main.tex为主文件；教师手册另选teacher_manual.tex。

字体优先使用系统Noto CJK和Linux Libertine等，缺少时自动回退到TeX随带字体。包内不分发字体。若换系统后分页稍有变化，应以本机编译结果为准。

## 二、运行实验

本轮测试环境为 Python 3.13.5、scikit-learn 1.8.0；精确依赖在requirements.txt。建议新建环境，不在原有科研环境中强制覆盖依赖。

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
jupyter lab
```

`notebooks/`包含17章各一份学生版和教师版，以及真实OGLE节选项目的两份，共36份。另在`chapter03_material/notebooks/`保留两份详细优化实验，共38份；本轮均在新Jupyter内核中完整执行，无代码单元错误。源码中的实验结果已保存，可直接审阅。学生版保留可运行的基准流程，要求在此基础上解释结果、改变预定因素并完成D题；教师版增加结果解释和评分要点。

逐章脚本没有隐式下载：

```bash
python experiments/ch08.py
python experiments/ch12.py
python experiments/project_ogle.py
python scripts/run_deep_experiments.py
```

前两条分别重现树与MLP实验；最后一条执行17章和真实天文节选。表在results/，新图在figures/v3_*。核心数学和数据核查：

```bash
python scripts/audit_numerics_v3.py
python scripts/audit_structure_v3.py
```

重建Notebook会清除原输出；修改实验代码后才需要运行 `python scripts/build_notebooks_v3.py`，随后用 `python scripts/execute_notebooks_v3.py`在新内核重新执行。支持`--start`和`--stop`按文件排序分批执行，便于排查。

## 三、数据的真实状态

1. 鸢尾花、葡萄酒和手写数字为sklearn公开标准数据，配有CSV；新主线实验可离线运行。
2. `ogle_lmc_f_excerpt.csv` 是真实OGLE目录前90条固定记录的少字段节选。来源、行号、缺失、含义和限制见 `data/OGLE_excerpt_README.md`。87条可用记录的模型比较与最终保留评价已实际运行。**它不是完整巡天样本，也不是新的周期—光度标定。**
3. 优化、边界、残差提升、异常注入与批次物理标定中使用的生成数据都明确标注“人工构造/模拟”。第17章物理项目有90批、720条模拟测量，不伪装成真实仪器数据。
4. California Housing、完整OGLE、SDSS、Gaia、Airfoil、Sonar、超导材料等完整外部数据不全部随包提供。本轮未完成这些全量目录的下载复算。相关章节提供背景、字段、任务设计与必要代码；外部项目需要其真实输入后才能产生结果。**没有用别的数据填补或伪造这些实验分数。**

显式联网获取（失败即报错，不回退为其他数据）：

```bash
python scripts/fetch_external_data.py --dataset california
python scripts/fetch_external_data.py --dataset airfoil
python scripts/fetch_external_data.py --dataset sonar
python scripts/fetch_external_data.py --dataset ogle-full
python scripts/fetch_external_data.py --dataset ogle-verify
```

下载后可运行 `python scripts/run_external_project.py --dataset airfoil` 等。Airfoil的工况组是由已给条件元组构造的代理组，不是原始实验批次标识；Sonar缺少足够的设备/重复观测分组资料，不能宣称跨设备泛化。完整SDSS/Gaia访问需要按正文选择样本和使用相关官方服务，不能把本地模板视为已经完成的巡天处理。

## 四、本轮深化内容

- 全部17章增加了机制解释、必要的短推导、可算例子或可检验的任务分析；不是把每章拉到相同字数。
- 第8章重新编写，纠正参考稿中的最佳阈值、节点计数、剪枝目标口径、单调变换范围和多维预测图解释；保留分类、回归与贪心反例的价值。
- 逻辑回归连接概率、似然、交叉熵与参数更新；森林解释平均误差与相关性；提升区分残差和一般负梯度；MLP连接表示能力、梯度计算和优化诊断。
- PCA从非共线数据求协方差与主方向；聚类用局部最优与指标冲突说明限制；异常检测区分排序、阈值、注入恢复和真实身份。
- 增加68道D编号深化题，并在教师手册给出计算与分析参考。具体模型表现不能用来替代问题定义、验证设计和科学解释。

## 五、目录

```
main.tex / main.pdf             全书编译入口及预览
chapters/                      17章正文
appendices/                    6个附录
teacher_manual.tex/.pdf        教师手册入口及预览
docs/teacher_manual.tex         基础与深化题答案
docs/references.tex             参考来源
figures/                       原有图和新增图，PDF/PNG
experiments/                   17章及真实天文节选脚本
data/                          CSV、划分、来源、字段说明
results/                       本轮实际生成的表与JSON
notebooks/                     36份主线Notebook
chapter03_material/            优化章节详细原生实验（含2份Notebook）
scripts/                       编译辅助、核验、下载与外部流程
review/quality_report.md        本轮最终核验摘要
review_v3/                     详细核验记录
```

旧版生成脚本和历史审阅材料如保留，均不代表本轮独立审稿。当前的有效质量记录以`review/quality_report.md`及其指向的v3记录为准。

## 六、完成程度与边界

全书正文、附录、解答、配图、主线实验和两项综合示范已统一输出；数值、运行与版式完成本轮技术核验。**没有两名独立agent或外部学者完成交叉审稿，也没有真实课堂试教。** Windows/macOS实机、全部联网项目、完整巡天数据验证尚未完成。具体检查见审阅记录，避免把“38份Notebook运行通过”理解为已经证明所有科研结论或所有平台兼容性。
