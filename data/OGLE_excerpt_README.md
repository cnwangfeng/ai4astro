# OGLE 真实目录节选：数据含义与来源

本文件说明 `ogle_lmc_f_excerpt.csv`。它不是模拟数据，也不是完整巡天样本。

**来源：** OGLE Collection of Variable Stars (OCVS), Classical Cepheids in the Large Magellanic Cloud, fundamental-mode table `cepF.dat`.

- 原表：https://www.astrouw.edu.pl/ogle/ogle4/OCVS/lmc/cep/cepF.dat
- 字段说明：https://www.astrouw.edu.pl/ogle/ogle4/OCVS/lmc/cep/README
- 项目主页：https://www.astrouw.edu.pl/ogle/ogle4/OCVS/lmc/cep/
- 应引用的目录论文：Soszyński et al. (2015), Acta Astronomica, 65, 297；完整引用按原项目要求核对。

## 节选规则

固定取本轮网页可读取版本的前90条数据记录（零起始 `source_row=0..89`），只保留来源行号、对象ID、I波段强度平均星等、V波段强度平均星等和主周期。不根据拟合残差或模型表现选取记录。该目录顺序并不是随机抽样，因此不能把此节选的分数推广成完整巡天精度。

字段由网页展示的原表转录。原表的 `-` 按缺失处理，CSV中留空，不变成0。当前90条中3条没有I目标；按当前任务排除后87条。V缺失不用于当前周期→I模型的删行。

| 列 | 含义 | 单位/类型 |
|---|---|---|
| source_row | 原表零起始位置，供追溯，不作为输入 | 整数 |
| source_id | OGLE目录对象标识 | 字符串 |
| I_mag | I波段强度平均视星等 | mag |
| V_mag | V波段强度平均视星等 | mag |
| period_days | 基频脉动主周期 | 天 |

## 运行与核验

`python experiments/project_ogle.py` 使用本地冻结节选，不联网。固定种子42将87条分成69条开发、18条保留；开发内部五折使用种子7。结果写入 `results/project_ogle_*`，划分在 `data/ogle_excerpt_split_v3.csv`，最终参数来自69条开发记录，不使用18条保留目标选择模型。

本轮已用独立 NumPy 最小二乘路径复核最终系数及测试RMSE。完整源文件的容器网络下载不可用，因此**没有声称完成源文件逐字节校验**。联网环境可运行：

```bash
python scripts/fetch_external_data.py --dataset ogle-verify
```

该命令逐列比较固定节选与在线前90条，发现目录顺序或数值更新时会报错，不静默覆盖。本轮没有执行这个需要联网的核验命令。

## 使用边界

保留OGLE署名与来源，不把第三方数据重新授权为本书原创数据。对外再发布或用于正式研究时，应检查项目的数据政策、引用要求及完整版本。当前节选只承担有限的教学建模任务；未完整处理消光、距离深度、拥挤、选择函数和目标测量误差，不产生新的绝对光度标定。
