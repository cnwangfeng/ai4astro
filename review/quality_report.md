# 全书深化稿 v3.0：最终质量记录

日期：2026-11-15。编写依据为既定17章计划、两轮初学者反馈以及用户上传第8章的深度参照。本文件说明实际完成内容及边界，不把作者自检称为独立双agent审稿。

## 1. 交付规模

| 项目 | 实际状态 |
|---|---|
| 正文 | 17章、119个编号小节 |
| 附录 | 6章；扩展内容纳入主文件 |
| 教材PDF | 223页 |
| 教师手册PDF | 25页 |
| 正文汉字计数 | 129,857；源码逐字计数，含标题和练习、不计lstlisting代码，不是出版字符总数 |
| 新增深化习题 | 68题，均有对应参考解答 |
| 配图资源 | 71组唯一文件主名，其中28组为v3新增；正文引用64个不同图文件 |
| Notebook | 36份主线＋2份原生优化材料，合计38份 |

## 2. 已完成的技术检查

**新内核执行：38/38通过。** 36份主线每份使用独立新Jupyter内核；两份详细第3章材料另行执行。记录在review_v3/notebook_execution.json和chapter03_legacy_notebook_execution.json。主线源码SHA256与执行时源码核对一致，未含错误输出。注意：Notebook主线不是对每个联网拓展和每段展示代码的穷举测试。

**独立计算路径：66/66通过。** scripts/audit_numerics_v3.py没有调用各章实验函数来给自己打分，而是另写小规模公式或数值路径，核对梯度、有限差分、最小二乘、阈值、树划分、剪枝、方差、提升、核映射、PCA、聚类、LOF及真实天文项目。结果载于numerical_audit.json。这里“独立”指计算路径，不指另一位审稿者。

**独立Python进程：18/18通过。** 17章脚本和真实OGLE节选脚本均重新运行；日志与时间在experiment_execution.json及各章run.log。没有依赖Notebook中遗留变量。固定环境下重新生成本轮结果表与新增图。

**结构检查通过。** 17章/119节、68道D题、输入文件、图路径、标签、引用、Notebook源码身份与执行记录一致；无正文待填结果或图占位。详见structure_audit.json。

**编译与版式检查通过。** 教材及手册XeLaTeX编译成功。最终日志：0个overfull box、0个缺字、0个未定义引用；所有页面文本和图像块均在页面范围内。223页教材已渲染并逐页接触表检查，关键树、网络、天文页及教师页放大检查未见遮挡或裁切。仍有xeCJK字体重定义/VerbAddon提示及若干underfull box；不是宣称零警告编译。

## 3. 数据和科学结果范围

- 鸢尾花、葡萄酒、手写数字：标准数据，本地可运行。
- 新增数值机制、边界、误差与标定项目：全部明确标为人工构造或模拟。标定项目90批720条；不声称真实仪器验证。
- OGLE：官方目录固定前90条少字段真实节选，87条有效，69开发/18保留。代码、字段、来源行号、划分及数据哈希随书。最终保留RMSE约0.1492mag，不代表完整巡天精度或新的绝对光度标定。
- 容器无法完成完整外部数据联网下载。完整Housing、SDSS、Gaia、Airfoil、Sonar、超导材料等项目没有全部重新运行；提供背景和显式获取/处理入口，不补造结果。
- OGLE节选由网页可读取源转录；本地交叉计算通过，但完整源文件逐字节和在线字段比较尚未完成。ogle-verify命令为联网环境准备，当前不标为执行通过。

## 4. 未完成且不能从上述通过项推导的事项

没有两名独立agent、外部学者交叉审稿或真实课堂试教；未做Windows/macOS实机验证。数值检查只覆盖列出的案例，不构成全部学术陈述的形式化证明。38份Notebook通过不代表所有专业扩展完成或所有软件版本兼容。交付为完整的深化书稿与工程，不标为已经出版定稿。

## 5. 重现入口

```bash
python scripts/audit_numerics_v3.py
python scripts/audit_structure_v3.py
python scripts/run_deep_experiments.py
python scripts/execute_notebooks_v3.py
latexmk -xelatex -interaction=nonstopmode -halt-on-error main.tex
latexmk -xelatex -interaction=nonstopmode -halt-on-error teacher_manual.tex
```

重跑会更新结果与图；修改配置时另存结果。历史v2记录位于archive_v2，不作为本轮通过证据。
