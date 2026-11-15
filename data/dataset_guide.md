# 随书数据：先理解对象，再读取文件

## 四种基础案例与一个明确命名的辅助案例

|资料|一行代表什么|输入|目标|第一次完整介绍|随包状态|
|---|---|---|---|---|---|
|鸢尾花 Iris|一朵花的测量记录|四项尺寸，厘米|三个物种|正文1.2节|iris.csv，150行|
|葡萄酒 Wine|一个酒样的化学分析记录|13项指标；原来源未系统注明单位|三个葡萄栽培品种来源类别|正文2.1节|wine.csv，178行|
|加州住房 California Housing|1990年人口普查街区组|8项区域统计量|区域住房价值中位数；十万美元|正文1.4概览、5.1完整字段|只提供显式下载脚本；不暗中替换|
|手写数字 Digits|一幅8×8灰度图像|64个按行展开的像素，0–16|数字0–9的类别|正文12.8节|digits.csv，1797行|
|糖尿病随访 Diabetes|一名受试者的基线记录|10项基线测量；默认已缩放|一年后的疾病进展数值|正文5.6节的辅助案例|diabetes.csv，442行；不作医学推断|

所有本地CSV的目标列名以 `catalog.json` 为准；四个内置数据均为 `target`。标签不是额外输入。`field_dictionaries/`逐列解释中文含义、角色与尺度。保存的CSV来自原始工程；其值会与当前sklearn内置数据逐项核对，检查报告列出实际版本。

葡萄酒类别0/1/2对应59/71/48条；原UCI文本使用1/2/3。Wine与Wine Quality不是同一数据。鸢尾花代码0/1/2仅是物种编号，不表示尺寸。数字图像的类别编码虽然是数值，仍是识别类别而不是普通数值回归。

## 加州住房的显式获取

联网后从工程根目录运行：

```bash
python scripts/fetch_housing.py
```

该命令调用官方sklearn下载接口，建立sklearn缓存并导出真实住房CSV。第5章Notebook设置 `DATASET='california'` 才读取该缓存；无缓存时报错并给出说明。默认 `diabetes_demo` 是在正文已介绍的**独立辅助实验**。必须在报告标题写实际数据名；辅助结果不能记入住房实验结果。科学结论取决于实际输入，不能以数据规模或“程序能运行”为由替换。

## 人工生成数据

`synthetic_moons.csv`是400条人工双月形二维点；`teaching_label`记录生成时的组别。生成函数和随机种子见 `catalog.json`。第3章直线数据在 `data/ch03/`，原始配套在 `chapter03_material/data/`；40条中30条用于训练、10条保留。其他手算点在正文和Notebook中直接列出，含义和数值完全给定。

第15章的15条均匀注入点只表示生成来源。它们可能落在主体附近，不能把每条注入自动称为真实可检出的异常。程序的命中统计只对这个受控演示成立。

## 六个专业案例：不是一组不加说明的文件名

- **OGLE造父变星**：OGLE是光学引力透镜实验巡天；本书使用其变星目录中的造父变星作为任务设计。对象是亮度周期变化的恒星，输入可为周期，目标为平均I波段表观星等。需明确星系、脉动模式、消光与质量筛选。完整介绍在5.8节。
- **SDSS天体目录**：斯隆数字巡天提供测光与光谱资料。用u/g/r/i/z波段测光判断恒星、星系或类星体时，光谱类别可作参考标签，不把同一标签或仅在事后取得的光谱信息偷放入输入。正文6.7节。
- **Gaia星团邻域**：Gaia是空间天体测量任务。选定星团邻域内每行是天体，位置、视差和自行是不同测量量；DR3固定数据版本及单位详见正文14.7节。聚类只能给候选，需要额外证据。
- **Airfoil Self-Noise翼型自噪声**：风洞实验的1,503条条件记录，输入为频率Hz、攻角度、弦长m、自由流速度m/s和吸力侧位移厚度m；目标为声压级dB。正文5.8节。
- **Sonar声呐回波**：208条回波记录、60个归一化频带能量特征；111条金属目标、97条岩石。标签代表实验对象，不扩张为实际排雷性能；未取得完整频率元数据时不自行加Hz标签。正文6.7节。
- **Superconductivity超导材料**：21,263条记录，81个成分描述特征；目标是临界温度。物理样本分组需检查相同化学式、重复记录与材料家族。正文10.7节。

本包未宣称下载并重新计算上述六套全部专业数据。这里交付的是正文背景、任务设计、字段检查要求和CSV执行入口。真实数据获取后，应保存版本、筛选记录、字段字典和对象/材料分组，再运行 `scripts/run_science_csv.py`。不能使用人工数据冒充这些数据。

## 原始说明与许可核对

数据来源、版本与许可证是三个独立问题。已有数据说明不能自动作为所有资料可再分发的统一许可。内置数据及CSV使用时保留官方来源；专业数据在获取时逐项查看来源的使用、引用和再分发条件。

- sklearn标准小数据：https://scikit-learn.org/stable/datasets/toy_dataset.html
- Iris：https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html
- Wine原始来源：https://archive.ics.uci.edu/dataset/109/wine
- California Housing：https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html
- Digits：https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html
- Diabetes：https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html
- OGLE：https://www.astrouw.edu.pl/ogle/ogle4/OCVS/lmc/cep/
- SDSS DR17：https://www.sdss4.org/dr17/spectro/
- Gaia DR3：https://www.cosmos.esa.int/web/gaia/dr3
- Airfoil：https://archive.ics.uci.edu/dataset/291/airfoil+self-noise
- Sonar：https://archive.ics.uci.edu/dataset/151/connectionist+bench+sonar+mines+vs+rocks
- Superconductivity：https://archive.ics.uci.edu/dataset/464/superconductivty+data
