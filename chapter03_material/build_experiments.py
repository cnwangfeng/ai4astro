"""Generate the reproducible Chapter 3 teaching package.

Run with: python build_experiments.py.  All data are synthetic.  The reserved
subset is written to CSV but is never used to fit, scale, choose settings, or
evaluate an experiment in this chapter.
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import inspect
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import textwrap
import time
import uuid

try:
    import resource
except ImportError:  # Windows can run all experiments without RSS accounting.
    resource = None

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
FIGURES = ROOT / "figures"
NOTEBOOKS = ROOT / "notebooks"
for directory in (DATA, FIGURES, NOTEBOOKS):
    directory.mkdir(parents=True, exist_ok=True)

FONT_PATH = Path(os.environ.get(
    "CH03_FONT_PATH", "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"))
if FONT_PATH.exists():
    font_manager.fontManager.addfont(str(FONT_PATH))
    plt.rcParams["font.family"] = font_manager.FontProperties(
        fname=str(FONT_PATH)
    ).get_name()
else:
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    for font_name in ("Noto Sans CJK SC", "Microsoft YaHei", "SimHei", "PingFang SC"):
        if font_name in available_fonts:
            plt.rcParams["font.family"] = font_name
            break
plt.rcParams.update({
    "axes.unicode_minus": False,
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 120,
    "savefig.dpi": 200,
    "svg.fonttype": "none",
})
COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7"]


def half_mse(x, y, theta):
    """Training half mean squared error for theta=[slope, intercept]."""
    prediction = theta[0] * x + theta[1]
    return float(np.mean((prediction - y) ** 2) / 2)


def gradient(x, y, theta):
    """Both gradient components are computed at the same old theta."""
    error = theta[0] * x + theta[1] - y
    return np.array([np.mean(error * x), np.mean(error)])


def gradient_descent(x, y, lr, steps, threshold=1e12):
    """Full batch GD; index 0 records initialization before any update."""
    theta = np.zeros(2, dtype=float)
    rows = [{"step": 0, "a": 0.0, "b": 0.0,
             "loss": half_mse(x, y, theta)}]
    stop = "maximum_updates"
    for step in range(1, steps + 1):
        theta = theta - lr * gradient(x, y, theta)
        loss = half_mse(x, y, theta)
        rows.append({"step": step, "a": float(theta[0]),
                     "b": float(theta[1]), "loss": loss})
        if not np.isfinite(loss) or loss > threshold:
            stop = "divergence_guard"
            break
    return pd.DataFrame(rows), stop


def batch_descent(x, y, lr, epochs, batch_size, seed):
    """Same constant learning rate, with loss evaluated on all train rows.

    Epochs have an equal number of sample visits, not equal update counts.
    Loss evaluation itself is omitted from sample_visit accounting.
    """
    n = len(x)
    theta = np.zeros(2, dtype=float)
    rng = np.random.default_rng(seed)
    rows = [{"step": 0, "sample_visits": 0, "epoch_passes": 0.0,
             "a": 0.0, "b": 0.0, "loss": half_mse(x, y, theta)}]
    sample_visits = 0
    step = 0
    for epoch in range(epochs):
        order = np.arange(n) if batch_size == n else rng.permutation(n)
        for start in range(0, n, batch_size):
            index = order[start:start + batch_size]
            theta = theta - lr * gradient(x[index], y[index], theta)
            sample_visits += len(index)
            step += 1
            rows.append({"step": step, "sample_visits": sample_visits,
                         "epoch_passes": sample_visits / n,
                         "a": float(theta[0]), "b": float(theta[1]),
                         "loss": half_mse(x, y, theta)})
    return pd.DataFrame(rows)


def save_figure(fig, stem):
    fig.savefig(FIGURES / f"{stem}.png", bbox_inches="tight")
    fig.savefig(FIGURES / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def md_cell(source):
    return {"cell_type": "markdown", "id": uuid.uuid4().hex[:12],
            "metadata": {}, "source": textwrap.dedent(source).strip()}


def code_cell(source):
    return {"cell_type": "code", "id": uuid.uuid4().hex[:12],
            "metadata": {}, "source": textwrap.dedent(source).strip(),
            "execution_count": None, "outputs": []}


def write_notebook(path, cells):
    notebook = {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": platform.python_version()},
            "chapter": "03", "data_status": "synthetic",
        },
        "cells": cells,
    }
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding="utf-8")


def notebook_cells(teacher, results):
    role = "教师版" if teacher else "学生版"
    cells = [
        md_cell(f"""
        # 第3章 模型如何学习：损失函数与优化（{role}）

        本实验只研究**训练过程**。40条记录全部由程序人工生成，并非物理测量或天文观测。
        数据划分已经固定：30条训练记录用于本章；另外10条保留记录不参与计算或评价。
        `loss` 始终指训练数据上的**半均方误差**，不代表新样本预测能力。

        建议从本目录或上一层 `ch03` 目录启动。运行全部单元可得到完整基础流程。
        学生任务写在文字单元中，不会中断基础代码；请另加单元作答。
        目标是解释参数为何改变、学习率与特征尺度为何重要，以及一次更新和一轮数据遍历的差别。
        """),
        code_cell("""
        from pathlib import Path
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import StandardScaler

        ROOT = Path.cwd()
        if not (ROOT / "data" / "line_data.csv").exists():
            ROOT = ROOT.parent
        if not (ROOT / "data" / "line_data.csv").exists():
            raise FileNotFoundError("请从 ch03 或 ch03/notebooks 目录运行")
        plt.rcParams["axes.unicode_minus"] = False
        """),
        md_cell("""
        ## 1. 先判断：数据、预测和参数

        读取数据前，请写出直线模型的输入、目标和两个可学习参数。
        读取后只选择 `split == 'train'`。`sample_id` 和 `split` 不作为预测特征。
        """),
        code_cell("""
        table = pd.read_csv(ROOT / "data" / "line_data.csv")
        train = table.loc[table["split"].eq("train")].copy()
        x = train["x"].to_numpy()
        y = train["y"].to_numpy()
        print("训练样本数:", len(train))
        print(train.head().to_string(index=False))
        """),
        code_cell(inspect.getsource(half_mse) + "\n" + inspect.getsource(gradient)
                  + "\n" + inspect.getsource(gradient_descent)),
        md_cell("""
        ## 2. 单参数损失：方向与步长

        此处 $J(w)=(w-3)^2/2$ 是人为设置的示意目标，与直线CSV无关。
        从 $w=0$ 出发，先手算学习率0.5时的两次更新，再运行。
        比较0.1、0.5、1.5和2.1；记录哪些轨迹跨过了最小点，以及跨过后是否收敛。
        """),
        code_cell("""
        rates = [0.1, 0.5, 1.5, 2.1]
        scalar_records = []
        for rate in rates:
            w = 0.0
            for step in range(13):
                scalar_records.append({"rate": rate, "step": step,
                                       "w": w, "loss": (w - 3) ** 2 / 2})
                w = w - rate * (w - 3)
        scalar = pd.DataFrame(scalar_records)
        print(scalar.loc[(scalar["rate"] == 0.5) & (scalar["step"] <= 3)]
              .to_string(index=False))
        fig, ax = plt.subplots(figsize=(7, 4))
        for rate, group in scalar.groupby("rate"):
            ax.semilogy(group["step"], group["loss"], marker="o", label=str(rate))
        ax.set(xlabel="Parameter updates", ylabel="Illustrative loss J(w)")
        ax.legend(title="Learning rate")
        ax.grid(alpha=0.2)
        plt.show()
        """),
        md_cell("""
        **解释任务A**：为什么学习率1.5会左右交替，却仍接近最小点？
        把2.1改成2.0，先预测损失会怎样，再验证。不能把本题的学习率边界推广给所有目标函数。

        图例有四个学习率，但0.5和1.5的损失曲线完全重合：二者到最小点的距离每步都减半，
        仅参数偏差的符号变化不同。平方损失消除了符号，所以这不是漏画了一条曲线。
        """),
        md_cell("""
        ## 3. 同时学习斜率和截距

        保持训练数据、初始参数和目标函数不变，采用全批量梯度下降。
        每次同时计算两个偏导，再同时更新两个参数。第0步尚未进行更新。
        """),
        code_cell("""
        history, stop_reason = gradient_descent(x, y, lr=0.2, steps=120)
        print(history.loc[history["step"].isin([0, 1, 5, 20, 120])]
              .to_string(index=False))
        print("停止原因:", stop_reason)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].scatter(x, y, s=25, color="black", label="Synthetic train data")
        grid = np.linspace(x.min(), x.max(), 100)
        for step in [0, 1, 5, 20, 120]:
            row = history.loc[history["step"].eq(step)].iloc[0]
            axes[0].plot(grid, row["a"] * grid + row["b"], label=f"step {step}")
        axes[0].set(xlabel="x (dimensionless)", ylabel="y (dimensionless)")
        axes[0].legend(fontsize=8)
        axes[1].semilogy(history["step"], history["loss"])
        axes[1].set(xlabel="Parameter updates", ylabel="Training half MSE")
        axes[1].grid(alpha=0.2)
        plt.show()
        """),
        code_cell("""
        least_squares = LinearRegression().fit(train[["x"]], train["y"])
        reference_theta = np.array([least_squares.coef_[0], least_squares.intercept_])
        final_theta = history[["a", "b"]].iloc[-1].to_numpy()
        print("梯度下降参数:", final_theta)
        print("最小二乘参数:", reference_theta)
        print("最小二乘训练半MSE:", half_mse(x, y, reference_theta))
        print("参数最大绝对差:", np.max(np.abs(final_theta - reference_theta)))
        """),
        md_cell("""
        `LinearRegression` 在这里采用常规最小二乘数值求解；它不默认执行本实验的梯度下降循环。
        两种求解路径在同一目标上得到相近结果，有助于检查梯度实现。
        **解释任务B**：生成公式的斜率和截距是2.5与0.8，为什么拟合值不必恰好等于它们？
        为什么训练损失变小仍不能说明模型已经能处理新的观测？
        """),
        md_cell("""
        ## 4. 特征尺度：相同信息，不同更新过程

        把唯一特征改写成 `x_large = 1000 * x`，不要把x与x_large同时作为两列特征。
        标准化器只在训练记录上拟合。先预测直接沿用0.2学习率会发生什么。
        """),
        code_cell("""
        x_large = 1000.0 * x
        scale = StandardScaler().fit(x_large.reshape(-1, 1))
        z = scale.transform(x_large.reshape(-1, 1)).ravel()
        raw_fast, fast_stop = gradient_descent(x_large, y, 0.2, 120)
        raw_small, _ = gradient_descent(x_large, y, 1e-7, 120)
        standardized, _ = gradient_descent(z, y, 0.2, 120)
        print("大尺度、原学习率:", raw_fast.tail(1).to_string(index=False), fast_stop)
        print("大尺度、极小学习率:", raw_small.tail(1).to_string(index=False))
        print("标准化:", standardized.tail(1).to_string(index=False))
        az, bz = standardized[["a", "b"]].iloc[-1]
        original_a = 1000.0 * az / scale.scale_[0]
        original_b = bz - az * scale.mean_[0] / scale.scale_[0]
        print("标准化模型换回原x坐标:", original_a, original_b)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.semilogy(raw_small["step"], raw_small["loss"], label="Large x, rate 1e-7")
        ax.semilogy(standardized["step"], standardized["loss"], label="Scaled x, rate 0.2")
        ax.set(xlabel="Parameter updates", ylabel="Training half MSE")
        ax.legend()
        ax.grid(alpha=0.2)
        plt.show()
        """),
        md_cell("""
        损失超过$10^{12}$会触发保护性停止；这是发散保护，不是成功收敛。
        **解释任务C**：极小学习率下为什么可能很快学到大致斜率，但截距仍接近0？
        为什么直接比较标准化前后的斜率数字会产生误解？
        """),
        code_cell(inspect.getsource(batch_descent)),
        md_cell("""
        ## 5. 一次更新与一轮遍历

        使用同一训练标准化特征z、相同初值0、相同常数学习率0.1和80轮遍历。
        批量大小30、6、1分别对应每轮1、5、30次更新。这里控制的是样本遍历量，
        不是更新次数或运行时间。每个图中纵坐标都由全部30条训练记录计算。
        """),
        code_cell("""
        batch_histories = {}
        for label, size in [("Full batch", 30), ("Mini batch", 6), ("SGD", 1)]:
            run = batch_descent(z, y, lr=0.1, epochs=80, batch_size=size, seed=42)
            batch_histories[label] = run
            print(label, run.tail(1).to_string(index=False))
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for label, run in batch_histories.items():
            axes[0].semilogy(run["epoch_passes"], run["loss"], label=label)
            recent = run.loc[run["epoch_passes"] >= 70]
            axes[1].plot(recent["epoch_passes"], recent["loss"], label=label)
        axes[0].set(xlabel="Sample visits / training size", ylabel="Training half MSE")
        axes[1].set(xlabel="Sample visits / training size", ylabel="Training half MSE")
        for ax in axes:
            ax.legend(fontsize=8)
            ax.grid(alpha=0.2)
        plt.show()
        """),
        md_cell("""
        **解释任务D**：把随机种子改为0、1、2，比较末步结果。更换批量大小后，
        “80轮”为什么不再表示相同更新次数？观察随机更新的某一步使训练损失增加，
        能否立刻判断梯度代码错误？实验包含额外的全训练损失记录，不适合直接比较训练速度。

        ## 6. 提交内容

        提交两步手算、直线参数更新记录、学习率与尺度对照图，并回答任务A—D。
        另用150—250字说明本章的停止条件、结果能支持的判断，以及尚未回答的泛化问题。
        保留训练/保留划分，不利用保留记录调学习率或做本章的模型评价。
        """),
    ]
    if teacher:
        fit = results["linear_fit"]
        scale_result = results["scaling"]
        cells.extend([
            md_cell(fr"""
            ## 教师参考：解释任务

            **A**：误差满足$w_{{t+1}}-3=(1-\eta)(w_t-3)$。
            1.5对应倍率-0.5，符号交替而绝对值缩小；2.0对应-1，幅度不变；
            2.1对应-1.1，幅度增大。这个边界只针对本题二次函数。

            **B**：本次训练最小二乘解约为a={fit['reference_a']:.6f}、b={fit['reference_b']:.6f}，
            训练半MSE={fit['reference_loss']:.6f}。有限样本含随机噪声，训练最优值不等于生成参数。
            未接触的数据、划分方式和评价指标将在第4章讨论；训练目标下降不能替代这项评价。

            **C**：放大输入使斜率方向的梯度与曲率显著变大。为保证该方向稳定而选择极小学习率，
            会使截距方向更新缓慢。本次120步极小学习率的截距为
            {scale_result['large_small_lr']['b_original']:.8f}，训练半MSE仍为
            {scale_result['large_small_lr']['loss']:.6f}。
            标准化后的斜率对应z而非x，必须先按给定公式换回原坐标才可比较。

            **D**：30条训练记录对应每轮全批量1次、小批量5次、随机30次更新；80轮分别80、400、2400次。
            小批量梯度由局部样本决定，不保证每次都减小全部训练记录的损失。
            固定常数学习率时，最优值附近仍可能有波动；不同排列带来不同末步结果。
            不应从单个种子给出普遍排名。
            """),
            md_cell("""
            ## 教师扩展：逐轮减小学习率

            下例只改变随机梯度的学习率日程，以相同顺序与常数0.1比较。
            此例用于解释后期波动，不根据保留集选择日程。
            """),
            code_cell("""
            def scheduled_sgd(x, y, epochs=80, seed=42):
                theta = np.zeros(2)
                rng = np.random.default_rng(seed)
                records = []
                for epoch in range(epochs):
                    rate = 0.1 / (1.0 + epoch / 10.0)
                    for index in rng.permutation(len(x)):
                        theta -= rate * gradient(x[index:index + 1],
                                                 y[index:index + 1], theta)
                    records.append({"epoch": epoch + 1, "rate": rate,
                                    "loss": half_mse(x, y, theta)})
                return pd.DataFrame(records)
            scheduled = scheduled_sgd(z, y)
            fixed = batch_histories["SGD"]
            fixed_epoch = fixed.loc[(fixed["step"] % len(z) == 0)
                                    & fixed["epoch_passes"].between(60, 80)]
            scheduled_window = scheduled.loc[scheduled["epoch"].between(60, 80)]
            print(scheduled.tail().to_string(index=False))
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(fixed_epoch["epoch_passes"], fixed_epoch["loss"], label="Constant rate")
            ax.plot(scheduled_window["epoch"], scheduled_window["loss"],
                    label="Decreasing rate")
            ax.set(xlim=(60, 80), xlabel="Epoch", ylabel="Training half MSE")
            ax.legend()
            ax.grid(alpha=0.2)
            plt.show()
            """),
            md_cell("""
            ## 正文习题参考答案

            1. 对直线模型，x是输入、y是监督目标，ŷ是预测，a和b是模型参数。学习率、批量大小、
            最大更新次数和正则化强度是训练前设置的超参数。学习率0.2改为0.02改变优化过程，
            直线改为二次曲线则改变模型表达与参数集合。样本ID不作输入。
            2. 三点例子的残差为1、0、-1，平均残差为0，MSE为2/3，半均方误差为1/3；正负抵消掩盖误差。
            在a=1、b=1时梯度为(-2/3,0)。学习率0.3更新后a=1.2、b=1，
            半均方误差为0.64/3，约0.213333。
            3. 示意损失中从0出发、学习率0.5时w1=1.5、w2=2.25，损失分别1.125和0.28125。
            学习率2.0时w在0和6之间交替，损失始终4.5。
            4. 使用已更新的a计算b的梯度不再是本章定义的同步全梯度更新；应先算完整梯度向量。
            即使某次仍下降，也不能据此说程序实现了同一种更新规则。
            5. 30条样本、80轮，三种批量大小30、6、1对应80、400、2400次更新和相同2400次样本访问。
            若某数据集样本数不能被批量大小整除，应保留末尾较小批次并按其实际大小取平均。
            6. 尺度实验参考任务C。仅将步数增加不能保证挽救过大学习率造成的发散；
            标准化要在训练数据上学习统计量，再用于其他数据。
            7. 改变打乱种子会改变随机/小批量的参数轨迹。应报告固定的比较条件和多个种子的结果，
            不挑选最好的一次。全批量在本程序里不依赖排列。
            8. 开放题，答案不唯一。例如传感器标定：样本为特定工况的一次读数，x为参考输入信号，
            y为传感器输出，可在有限量程内先尝试ŷ=ax+b。需检查量程、系统偏差、重复读数相关性、
            测量不确定性、异常记录和划分，再评价新条件下的预测。训练直线与已知物理公式形似，
            不足以验证机制；残差结构和外推范围仍需检查。

            ### 常见课堂问题

            学生把1e12保护停止称作收敛时，先让其比较初始与末步损失；把标准化斜率直接与2.5比较时，
            让其写出z和x的关系；把训练曲线称作测试结果时，要求指出哪一行代码使用了新样本。
            不要求学生从零推导矩阵求导，也不在本章引入MLP反向传播的链式计算。
            """),
        ])
    return cells


EXECUTE_NOTEBOOK = r'''
import base64, contextlib, io, json, os, pathlib, sys, time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
path = pathlib.Path(sys.argv[1]).resolve()
notebook = json.loads(path.read_text(encoding="utf-8"))
os.chdir(path.parent)
scope = {"__name__": "__main__"}
count = 0
start = time.perf_counter()
for cell in notebook["cells"]:
    if cell["cell_type"] != "code":
        continue
    count += 1
    capture = io.StringIO()
    with contextlib.redirect_stdout(capture), contextlib.redirect_stderr(capture):
        exec(compile(cell["source"], f"{path.name}:cell{count}", "exec"), scope)
    cell["execution_count"] = count
    cell["outputs"] = []
    if capture.getvalue():
        cell["outputs"].append({"output_type": "stream", "name": "stdout",
                                "text": capture.getvalue()})
    for number in plt.get_fignums():
        fig = plt.figure(number)
        image = io.BytesIO()
        fig.savefig(image, format="png", dpi=120, bbox_inches="tight")
        cell["outputs"].append({"output_type": "display_data", "metadata": {},
            "data": {"image/png": base64.b64encode(image.getvalue()).decode("ascii"),
                     "text/plain": "Matplotlib figure"}})
    plt.close("all")
notebook["metadata"]["execution_verification"] = {
    "method": "fresh Python process; sequential exec of code cells; no Jupyter kernel",
    "code_cells": count,
    "all_cells_completed": True,
}
path.write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({"file": path.name, "code_cells": count,
                  "seconds": round(time.perf_counter() - start, 4),
                  "all_cells_completed": True}))
'''


def main():
    started = time.perf_counter()
    n = 40
    noise_seed = 2026
    split_seed = 20261114
    rng = np.random.default_rng(noise_seed)
    full_x = np.linspace(-2.0, 2.0, n)
    full_y = 2.5 * full_x + 0.8 + rng.normal(0.0, 0.3, n)
    chosen = np.random.default_rng(split_seed).permutation(n)[:30]
    is_train = np.zeros(n, dtype=bool)
    is_train[chosen] = True
    table = pd.DataFrame({"sample_id": [f"S{i + 1:03d}" for i in range(n)],
                          "x": full_x, "y": full_y,
                          "split": np.where(is_train, "train", "reserved")})
    table.to_csv(DATA / "line_data.csv", index=False, float_format="%.17g")
    # All experimental work begins again from CSV and selects train only.
    table = pd.read_csv(DATA / "line_data.csv")
    train = table.loc[table["split"].eq("train")].copy()
    x = train["x"].to_numpy()
    y = train["y"].to_numpy()
    reference = LinearRegression().fit(train[["x"]], train["y"])
    theta_ref = np.array([reference.coef_[0], reference.intercept_])
    ref_loss = half_mse(x, y, theta_ref)

    scalar_rows = []
    for lr in [0.1, 0.5, 1.5, 2.1]:
        w = 0.0
        for step in range(13):
            scalar_rows.append({"learning_rate": lr, "step": step,
                                "w": w, "loss": 0.5 * (w - 3.0) ** 2})
            w -= lr * (w - 3.0)
    scalar = pd.DataFrame(scalar_rows)
    scalar.to_csv(DATA / "scalar_trajectories.csv", index=False)

    fit, fit_stop = gradient_descent(x, y, lr=0.2, steps=120)
    fit.to_csv(DATA / "linear_training_trace.csv", index=False)
    scaled_x = 1000.0 * x
    scaler = StandardScaler().fit(scaled_x.reshape(-1, 1))
    z = scaler.transform(scaled_x.reshape(-1, 1)).ravel()
    large_fast, fast_stop = gradient_descent(scaled_x, y, lr=0.2, steps=120)
    large_small, small_stop = gradient_descent(scaled_x, y, lr=1e-7, steps=120)
    standardized, std_stop = gradient_descent(z, y, lr=0.2, steps=120)
    scale_runs = [("large_original_lr", large_fast, fast_stop, 0.2),
                  ("large_small_lr", large_small, small_stop, 1e-7),
                  ("standardized", standardized, std_stop, 0.2)]
    scale_result = {}
    for name, trace, reason, lr in scale_runs:
        trace.to_csv(DATA / f"scaling_{name}.csv", index=False)
        last = trace.iloc[-1]
        a, b = float(last["a"]), float(last["b"])
        if name == "standardized":
            a_orig = 1000.0 * a / scaler.scale_[0]
            b_orig = b - a * scaler.mean_[0] / scaler.scale_[0]
        else:
            a_orig, b_orig = 1000.0 * a, b
        scale_result[name] = {
            "learning_rate": lr, "updates": int(last["step"]),
            "loss": float(last["loss"]), "a_fitted_coordinate": a,
            "b_fitted_coordinate": b, "a_original": a_orig,
            "b_original": b_orig, "stop_reason": reason,
        }
    scale_result["training_scaler_mean"] = float(scaler.mean_[0])
    scale_result["training_scaler_scale"] = float(scaler.scale_[0])
    scale_result["divergence_guard_threshold"] = 1e12

    batch_sizes = {"full": 30, "mini": 6, "sgd": 1}
    batch_runs = {}
    batch_result = {}
    for name, size in batch_sizes.items():
        trace = batch_descent(z, y, 0.1, 80, size, 42)
        batch_runs[name] = trace
        trace.to_csv(DATA / f"batch_{name}_trace.csv", index=False)
        last = trace.iloc[-1]
        repetitions = []
        for seed in range(5):
            repeated = batch_descent(z, y, 0.1, 80, size, seed)
            repeated_last = repeated.iloc[-1]
            repetitions.append({"seed": seed, "loss": float(repeated_last["loss"]),
                                "a_z": float(repeated_last["a"]),
                                "b_z": float(repeated_last["b"])})
        pd.DataFrame(repetitions).to_csv(DATA / f"batch_{name}_seeds.csv", index=False)
        batch_result[name] = {
            "batch_size": size, "learning_rate": 0.1, "epochs": 80,
            "seed": 42, "updates": int(last["step"]),
            "sample_visits": int(last["sample_visits"]),
            "final_loss": float(last["loss"]),
            "a_z": float(last["a"]), "b_z": float(last["b"]),
            "loss_range_seeds_0_to_4": [min(r["loss"] for r in repetitions),
                                         max(r["loss"] for r in repetitions)],
            "repeated_runs": repetitions,
        }

    # A tiny, exact example used for hand calculation in the text.
    toy_x = np.array([-1.0, 0.0, 1.0])
    toy_y = np.array([-1.0, 1.0, 3.0])
    toy_theta = np.array([1.0, 1.0])
    toy_grad = gradient(toy_x, toy_y, toy_theta)
    toy_updated = toy_theta - 0.3 * toy_grad

    # The regularized solutions illustrate a training objective, not validation.
    design = np.column_stack([x, np.ones(len(x))])
    regularized = []
    for strength in [0.0, 0.1, 1.0]:
        theta = np.linalg.solve(design.T @ design / len(x) + np.diag([strength, 0]),
                                design.T @ y / len(x))
        loss = half_mse(x, y, theta)
        penalty = strength * theta[0] ** 2 / 2
        regularized.append({"lambda": strength, "a": float(theta[0]),
                            "b": float(theta[1]), "empirical_loss": loss,
                            "penalty": float(penalty),
                            "objective": float(loss + penalty)})
    pd.DataFrame(regularized).to_csv(DATA / "regularization_demo.csv", index=False)

    # 3-1: model family and candidate parameters.
    grid = np.linspace(-2.1, 2.1, 200)
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    ax.scatter(x, y, color="#303030", s=29, label="人工生成训练数据（n=30）", zorder=4)
    for a, b, label, color in [(0, 0, "a=0，b=0", COLORS[1]),
                               (1.5, 0.4, "a=1.5，b=0.4", COLORS[2]),
                               (theta_ref[0], theta_ref[1], "训练最小二乘解", COLORS[0])]:
        ax.plot(grid, a * grid + b, lw=2, label=label, color=color)
    ax.set(xlabel="输入 x（无量纲）", ylabel="目标 y（无量纲）")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(alpha=0.16)
    save_figure(fig, "fig3_1_candidate_lines")

    # 3-2: a deliberately defined scalar objective, no observational data.
    fig = plt.figure(figsize=(9.5, 7.1))
    layout = fig.add_gridspec(2, 2)
    axes = [fig.add_subplot(layout[0, 0]), fig.add_subplot(layout[0, 1]),
            fig.add_subplot(layout[1, :])]
    w_grid = np.linspace(-0.5, 6.5, 300)
    axes[0].plot(w_grid, 0.5 * (w_grid - 3.0) ** 2, color="#444444")
    path = scalar.loc[(scalar.learning_rate == 0.5) & (scalar.step <= 5)]
    axes[0].plot(path.w, path.loss, "o-", color=COLORS[0], ms=5)
    for _, row in path.iloc[:4].iterrows():
        axes[0].annotate(f"{int(row.step)}", (row.w, row.loss),
                         xytext=(5, 6), textcoords="offset points", fontsize=9)
    axes[0].set(xlabel="参数 w", ylabel="示意损失 J(w)", title="η=0.5 的更新位置")
    for (rate, group), color in zip(scalar.groupby("learning_rate"), COLORS):
        axes[1].plot(group.step, group.w, marker="o", ms=3, color=color, label=f"η={rate}")
        axes[2].semilogy(group.step, group.loss, marker="o", ms=3, color=color, label=f"η={rate}")
    axes[1].axhline(3.0, color="#555555", ls="--", lw=1)
    axes[1].set(xlabel="参数更新次数", ylabel="参数 w", title="参数轨迹")
    axes[2].set(xlabel="参数更新次数", ylabel="示意损失 J(w)（对数轴）", title="学习率与下降过程")
    axes[2].legend(fontsize=10, ncol=4)
    for ax in axes:
        ax.grid(alpha=0.17)
    fig.tight_layout()
    save_figure(fig, "fig3_2_scalar_learning_rates")

    # 3-3: full-batch line training.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].scatter(x, y, color="#303030", s=24, zorder=5, label="训练记录")
    line_colors = ["#999999", "#E69F00", "#CC79A7", "#009E73", "#0072B2"]
    for step, color in zip([0, 1, 5, 20, 120], line_colors):
        row = fit.loc[fit.step == step].iloc[0]
        axes[0].plot(grid, row.a * grid + row.b, color=color, lw=1.9, label=f"第{step}步")
    axes[0].set(xlabel="输入 x（无量纲）", ylabel="目标 y（无量纲）")
    axes[0].legend(fontsize=9, ncol=2)
    axes[1].semilogy(fit.step, fit.loss, color=COLORS[0], lw=2, label="梯度下降，η=0.2")
    axes[1].axhline(ref_loss, color="#555555", ls="--", lw=1.3, label="训练最小二乘参考值")
    axes[1].set(xlabel="参数更新次数", ylabel="训练半均方误差（对数轴）")
    axes[1].legend(fontsize=9)
    for ax in axes:
        ax.grid(alpha=0.17)
    fig.tight_layout()
    save_figure(fig, "fig3_3_line_training")

    # 3-4: separate panels keep the divergence from hiding stable trajectories.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].semilogy(large_fast.step, large_fast.loss, "o-", color="#D55E00", lw=2)
    axes[0].axhline(1e12, color="#555555", ls="--", label="发散保护阈值")
    axes[0].set(xlabel="参数更新次数", ylabel="训练半均方误差（对数轴）",
                title="输入放大1000倍，η=0.2")
    axes[0].set_xticks(large_fast.step)
    axes[0].legend(fontsize=9)
    axes[1].semilogy(large_small.step, large_small.loss, color=COLORS[1], lw=2,
                     label=r"放大输入，$\eta=10^{-7}$")
    axes[1].semilogy(standardized.step, standardized.loss, color=COLORS[0], lw=2,
                     label="训练集标准化，η=0.2")
    axes[1].axhline(ref_loss, color="#555555", ls="--", lw=1, label="最小二乘参考值")
    axes[1].set(xlabel="参数更新次数", ylabel="训练半均方误差（对数轴）",
                title="稳定更新仍可能非常缓慢")
    axes[1].legend(fontsize=9)
    for ax in axes:
        ax.grid(alpha=0.17)
    fig.tight_layout()
    save_figure(fig, "fig3_4_feature_scaling")

    # 3-5: common sample-pass axis with explicit update-count accounting.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    labels = {"full": "全批量（30）", "mini": "小批量（6）", "sgd": "随机（1）"}
    for (name, trace), color in zip(batch_runs.items(), COLORS):
        axes[0].semilogy(trace.epoch_passes, trace.loss, color=color, lw=1.5, label=labels[name])
        recent = trace.loc[trace.epoch_passes >= 70]
        axes[1].plot(recent.epoch_passes, recent.loss, color=color, lw=1.2, label=labels[name])
    for ax in axes:
        ax.axhline(ref_loss, color="#555555", ls="--", lw=1, label="最小二乘参考值")
        ax.set_xlabel("累计参与更新的样本数 / 30")
        ax.grid(alpha=0.17)
    axes[0].set_ylabel("训练半均方误差（对数轴）")
    axes[1].set_ylabel("训练半均方误差")
    axes[0].set_title("相同80轮遍历，更新次数不同")
    axes[1].set_title("第70—80轮的局部放大")
    axes[0].legend(fontsize=9)
    fig.tight_layout()
    save_figure(fig, "fig3_5_batch_methods")

    last_fit = fit.iloc[-1]
    results = {
        "chapter": 3,
        "data": {"kind": "synthetic; not observed or measured", "n_total": 40,
                 "n_train": 30, "n_reserved": 10,
                 "generation": "x=linspace(-2,2,40); y=2.5*x+0.8+N(0,0.3^2)",
                 "noise_seed": noise_seed, "split_seed": split_seed,
                 "train_sample_ids": train.sample_id.tolist(),
                 "reserved_used_for_fitting_scaling_selection_or_evaluation": False,
                 "csv_sha256": hashlib.sha256((DATA / "line_data.csv").read_bytes()).hexdigest()},
        "objective": "training half MSE = mean((a*x+b-y)^2)/2",
        "scalar": {"objective": "0.5*(w-3)^2; illustrative, no data",
                   "initial_w": 0.0, "steps": 12,
                   "last_step_by_rate": scalar.loc[scalar.step == 12].to_dict("records"),
                   "rate_0_5_first_steps": scalar.loc[(scalar.learning_rate == 0.5)
                                                       & (scalar.step <= 3)].to_dict("records")},
        "hand_calculation": {"x": toy_x, "y": toy_y, "initial_theta": toy_theta,
                             "residuals": toy_theta[0] * toy_x + toy_theta[1] - toy_y,
                             "initial_loss": half_mse(toy_x, toy_y, toy_theta),
                             "gradient": toy_grad, "learning_rate": 0.3,
                             "updated_theta": toy_updated,
                             "updated_loss": half_mse(toy_x, toy_y, toy_updated)},
        "linear_fit": {"learning_rate": 0.2, "updates": 120,
                       "initial_loss": float(fit.iloc[0].loss),
                       "final_a": float(last_fit.a), "final_b": float(last_fit.b),
                       "final_loss": float(last_fit.loss),
                       "reference_a": float(theta_ref[0]), "reference_b": float(theta_ref[1]),
                       "reference_loss": ref_loss,
                       "max_absolute_parameter_difference": float(np.max(np.abs(
                           np.array([last_fit.a, last_fit.b]) - theta_ref))),
                       "final_gradient_norm": float(np.linalg.norm(gradient(x, y,
                           np.array([last_fit.a, last_fit.b])))),
                       "stop_reason": fit_stop,
                       "selected_steps": fit.loc[fit.step.isin([0, 1, 5, 20, 120])].to_dict("records")},
        "regularization_demo": regularized,
        "scaling": scale_result,
        "batch_methods": batch_result,
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "pandas": pd.__version__, "matplotlib": matplotlib.__version__,
                        "scikit_learn": sklearn.__version__, "platform": platform.platform()},
    }
    results = json_safe(results)
    for teacher in (False, True):
        name = "chapter03_teacher.ipynb" if teacher else "chapter03_student.ipynb"
        write_notebook(NOTEBOOKS / name, notebook_cells(teacher, results))

    execution_results = []
    for name in ("chapter03_student.ipynb", "chapter03_teacher.ipynb"):
        complete = subprocess.run([sys.executable, "-c", EXECUTE_NOTEBOOK,
                                   str(NOTEBOOKS / name)], check=True,
                                  capture_output=True, text=True)
        execution_results.append(json.loads(complete.stdout.strip()))
    results["notebook_execution"] = {
        "method": "independent Python process for each notebook; sequential execution of every code cell",
        "jupyter_kernel_used": False, "notebooks": execution_results,
    }
    rss_divisor = 1024 * 1024 if platform.system() == "Darwin" else 1024
    parent_rss = (round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / rss_divisor, 2)
                  if resource is not None else None)
    child_rss = (round(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / rss_divisor, 2)
                 if resource is not None else None)
    results["runtime"] = {
        "build_and_notebook_execution_seconds": round(time.perf_counter() - started, 4),
        "parent_process_peak_rss_mib": parent_rss,
        "child_process_max_peak_rss_mib": child_rss,
        "measurement_scope": "local run including figure rendering and notebook outputs; not a speed benchmark",
    }
    (ROOT / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    (ROOT / "requirements.txt").write_text(
        f"numpy=={np.__version__}\npandas=={pd.__version__}\n"
        f"matplotlib=={matplotlib.__version__}\nscikit-learn=={sklearn.__version__}\n",
        encoding="utf-8")
    (ROOT / "environment.txt").write_text(
        "第3章实验环境\n" + json.dumps(results["environment"], ensure_ascii=False, indent=2)
        + "\n\n运行方式：python build_experiments.py\n"
        + "建议用独立Python环境，先运行 python -m pip install -r requirements.txt。\n"
        + "全部图表为Matplotlib计算绘图；本次中文字体：Noto Sans CJK。\n"
        + "其他平台可安装Noto Sans CJK、微软雅黑或黑体；也可通过CH03_FONT_PATH\n"
        + "指定本机中文字体文件路径。若没有可用中文字体，计算仍可执行，但图中文字可能缺失。\n"
        + "Notebook为nbformat 4.5。当前环境不含Jupyter内核，已分别在独立Python进程\n"
        + "中按顺序执行全部代码单元，并保存输出；尚未在Jupyter前端逐项试用。\n"
        + "Jupyter Notebook/Lab为交互打开所需的可选环境，不属于本次已安装依赖；\n"
        + "需要时可另行安装，例如 python -m pip install notebook。\n"
        + "内存统计使用可选的resource模块；Windows上该项记为null，不影响实验。\n"
        + "生成CSV使用两个独立种子；Notebook只读CSV，不重新划分。\n"
        + json.dumps(results["runtime"], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    (DATA / "data_dictionary.txt").write_text(textwrap.dedent(f"""
        # 第3章人工生成数据说明

        `line_data.csv` 是本教材生成的教学数据，**不是实验测量或天文观测**。
        生成式：x在[-2,2]上等间距取40点，y=2.5x+0.8+ε，ε为均值0、标准差0.3的独立正态随机数。
        NumPy default_rng噪声种子为{noise_seed}；独立划分种子为{split_seed}。
        固定随机排列的前30个索引为训练记录，其余10个保留。本章不使用保留记录拟合、缩放、选择设置或评价。
        x与y均为无量纲教学变量。无缺失值、无实体分组、无真实测量误差标记。
        生成代码由本教材提供，可随本章教学材料一起分发；不存在第三方数据下载或再分发限制。

        |字段|类型|含义|模型用途|
        |---|---|---|---|
        |sample_id|字符串|固定记录标识S001—S040|身份字段，不作特征|
        |x|浮点数|人工生成的输入坐标，无量纲|唯一输入特征|
        |y|浮点数|人工生成目标，无量纲|监督目标|
        |split|字符串|train或reserved|划分字段，不作特征|

        `scalar_trajectories.csv`记录独立的示意损失J(w)=(w-3)^2/2，不来自line_data。
        其余CSV为训练过程记录，不是新增观测。`step`从0开始；第0步为初始化。
        `loss`统一为训练半均方误差，唯scalar文件为其明确给定的示意损失。
        `a`、`b`属于各实验当前输入坐标；尺度实验的原坐标换算结果见results.json。
        批量实验使用训练标准化坐标z；`epoch_passes=sample_visits/30`，`step`为参数更新次数。
        `sample_visits`不计额外记录整批训练损失所需的计算，不能当作墙钟计时。
        `regularization_demo.csv`用原始x坐标、截距不惩罚，目标为半MSE+lambda*a^2/2。
        `batch_*_seeds.csv`为种子0—4的末步训练结果；正文曲线种子为42。

        输入CSV SHA-256：`{results['data']['csv_sha256']}`。
    """).strip() + "\n", encoding="utf-8")
    print(json.dumps({"linear_fit": results["linear_fit"],
                      "scaling": results["scaling"],
                      "batch_methods": {k: {q: v[q] for q in
                          ["batch_size", "updates", "final_loss", "loss_range_seeds_0_to_4"]}
                          for k, v in results["batch_methods"].items()},
                      "notebook_execution": results["notebook_execution"],
                      "runtime": results["runtime"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
