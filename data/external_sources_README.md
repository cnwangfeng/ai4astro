# 专业数据来源、可复现与字段要求

本文件列出六个专业案例的官方数据来源与字段要求。数据版本与再分发许可请在使用机构网络环境内核验；书中示例均使用内置教学数据。

| 任务 | 推荐资料来源 | 推荐输入、输出、分组与注意事项 |
|---|---|---|
| 造父变星周期--星等 | OGLE-IV Classical Cepheids: https://www.astrouw.edu.pl/ogle/ogle4/OCVS/lmc/cep/ | P (day), mean I mag, pulsation mode; 需筛同区域/模式；目标是视星等，不是自动校正的绝对星等 |
| 恒星/星系/类星体分类 | SDSS DR17: https://www.sdss4.org/dr17/ | 模型可用测光量，类别标签来自光谱；需排除泄漏列，重复对象按唯一身份分组 |
| 星团候选识别 | Gaia DR3: https://gea.esac.esa.int/archive/documentation/GDR3/ | parallax, pmra, pmdec, color, spatial coords; 质量筛选及候选复核；无监督不等于已确认成员 |
| 风洞声学回归 | UCI Airfoil Self-Noise: https://archive.ics.uci.edu/dataset/291/airfoil+self-noise | 频率、攻角、弦长、来流速度和位移厚度；目标声压级dB，若无工况ID不可宣称跨工况验证 |
| 声呐分类 | UCI Sonar: https://archive.ics.uci.edu/dataset/151/connectionist+bench+sonar+mines+vs+rocks | 60个频带特征，标签是实验目标类型；208记录的小样本波动须说明 |
| 超导临界温度 | UCI Superconductivity: https://archive.ics.uci.edu/dataset/464/superconductivty+data | 原材料组成及描述符；目标临界温度；查重、按配方分组及对外推谨慎陈述 |

## 字段字典模板

`data/field_dictionary_template.csv` 包含列 `dataset,column,description,unit,role,dtype,missing_code,source`。`role` 应区分 input、target、group、id、quality、reference（事后核对）。原始CSV保留不修改的副本，处理脚本生成训练表格，记录版本、筛选、剔除原因及关联操作。

## 可复用CSV脚本

示例（用内置 Wine 数据检查脚本功能）：

```bash
python scripts/run_science_csv.py \
  --csv data/wine.csv --target target \
  --features "alcohol,malic_acid,ash" --task classification \
  --out results/example_model
```

真实科学数据应根据字段字典填写参数；若同一天体/同一材料有多次记录，请提供分组列，并检查程序输出的划分描述。无监督探索请使用对应 Notebook，不需要构造标签。
