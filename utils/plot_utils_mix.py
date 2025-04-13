import itertools
import numpy as np
import matplotlib.pyplot as plt
import pickle
import uproot
import os
import sys
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
import pickle

#latex
plt.rc('text', usetex=True)


# 定义类别名称字典（用于 ROC/Rejection 等图中的标注）
name = {0: "b-jet", 1: "bbar-jet", 2: "c-jet", 3: "cbar-jet",
        4: "s-jet", 5: "sbar-jet", 6: "u-jet", 7: "ubar-jet",
        8: "d-jet", 9: "dbar-jet", 10: "gluon"}


def plot_confusion_matrix(cm,
                          classes,
                          cmerror=None,
                          normalize=False,
                          title="",
                          filename=None,
                          precision=4,
                          integer=False,
                          logColor=False,
                          fontsize=6,
                          cmap=plt.cm.Blues):
    """
    绘制混淆矩阵。

    参数：
      - cm: 混淆矩阵数组
      - classes: 类别标签列表
      - cmerror: 混淆矩阵误差（可选）
      - normalize: 是否归一化
      - title: 图像标题
      - filename: 如果提供，则保存图像至该文件
      - precision: 数值显示精度
      - integer: 是否显示整数
      - logColor: 是否对颜色映射进行对数变换
      - fontsize: 文字大小
      - cmap: 色图
    """
    plt.figure(figsize=(5, 5))
    plt.title(title)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    def text_at(i_true, j_pred):
        if integer:
            return "%d" % cm[i_true, j_pred]
        if cmerror is None:
            return "{:.{}f}".format(cm[i_true, j_pred], precision)
        else:
            s = "{:.{}f}\n({})".format(cm[i_true, j_pred], precision,
                                       int(round(np.power(10, precision) * cmerror[i_true, j_pred])))
            return s[1:]

    args = {"interpolation": "nearest", "cmap": cmap}
    if logColor:
        plt.imshow(np.power(cm, 0.5), **args)
    else:
        plt.imshow(cm, **args)

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, text_at(i, j), fontsize=fontsize,
                 horizontalalignment="center",
                 verticalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True')
    plt.xlabel('Predicted')
    if filename:
        plt.savefig(filename)
    # plt.show()


# 定义一个空类用于存储数据
class Data:
    pass


def load_merged_data(filepath):
    """
    从合并 ROOT 文件中加载数据。

    要求 ROOT 文件中包含标签分支和得分分支，分支名称如下：
      - 标签：["label_b", "label_bbar", "label_c", "label_cbar",
               "label_d", "label_dbar", "label_u", "label_ubar",
               "label_s", "label_sbar", "label_g"]
      - 得分：["score_label_b", "score_label_bbar", "score_label_c", "score_label_cbar",
               "score_label_d", "score_label_dbar", "score_label_u", "score_label_ubar",
               "score_label_s", "score_label_sbar", "score_label_g"]

    返回一个 Data 对象，其中包含：
      - Xtotal: 得分二维数组 (n_events, n_classes)
      - class_names: 类别名称列表（LaTeX 格式）
      - y_true: 真实标签（整数数组）
      - y_pred: 预测标签（整数数组，由最大得分确定）
      - Num: 每个类别的事件数
    """
    file = uproot.open(filepath)
    tree = file["Events"]

    # 定义分支名称（根据你的 ROOT 文件实际情况调整顺序）
    labels = ["label_b", "label_bbar", "label_c", "label_cbar",
              "label_d", "label_dbar", "label_u", "label_ubar",
              "label_s", "label_sbar", "label_g"]
    scores = ["score_label_b", "score_label_bbar", "score_label_c", "score_label_cbar",
              "score_label_d", "score_label_dbar", "score_label_u", "score_label_ubar",
              "score_label_s", "score_label_sbar", "score_label_g"]

    data_arrays = tree.arrays(labels + scores, library="np")
    n_events = len(data_arrays[labels[0]])
    y_true = np.zeros(n_events, dtype=int)
    # 假定每个事件采用 one-hot 编码（只有一个分支为 1）
    for i, lab in enumerate(labels):
        y_true[data_arrays[lab] == 1] = i

    # 组合得分数组，shape 为 (n_events, n_classes)
    scores_arr = np.array([data_arrays[score] for score in scores]).T
    y_pred = np.argmax(scores_arr, axis=1)

    # 计算每个类别的事件数
    Num = np.array([np.sum(y_true == i) for i in range(len(labels))])

    # 定义类别名称（注意顺序应与分支对应）
    class_names = ["$b$", r"$\overline{b}$", "$c$", r"$\overline{c}$",
                   "$d$", r"$\overline{d}$", "$u$", r"$\overline{u}$",
                   "$s$", r"$\overline{s}$", "$G$"]

    data = Data()
    data.Xtotal = scores_arr
    data.class_names = class_names
    data.y_true = y_true
    data.y_pred = y_pred
    data.Num = Num
    return data


def convert_files(rootpath, piclepath, filelist):
    """
    将多个 ROOT 文件转换为 pickle 文件（适用于原来分散的情况）。
    """
    os.system("mkdir -p " + piclepath)
    for filename in filelist:
        with uproot.open(rootpath + filename) as file:
            arrays = file["Events"].arrays(library="np")
            with open(piclepath + filename[:-5] + ".pickle", 'wb') as handle:
                pickle.dump(arrays, handle)


def plot_confusion_matrices(y_true, y_pred, class_names, Num=None, save=False, output_dir="Figures"):
    """
    绘制一系列混淆矩阵图。

    参数：
      - y_true, y_pred: 真实与预测标签
      - class_names: 类别标签列表
      - Num: 每个类别的事件数（用于计算统计误差）
      - save: 是否保存图片
      - output_dir: 保存图片的目录
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 绘制未归一化的混淆矩阵
    cnf_matrix = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cnf_matrix,
                          integer=True,
                          classes=class_names, normalize=False,
                          title='Unnormalized confusion matrix')

    # 绘制归一化混淆矩阵
    cnf_matrix_norm = confusion_matrix(y_true, y_pred, normalize="true")
    print("Normalized confusion matrix:")
    print("Trace:", np.trace(cnf_matrix_norm))
    print(cnf_matrix_norm)
    print("Row sums:", np.sum(cnf_matrix_norm, axis=1))

    cnf_matrix_error = None
    if Num is not None:
        # 计算统计误差
        cnf_matrix_std = np.sqrt(cnf_matrix_norm * (1 - cnf_matrix_norm) * Num[:, np.newaxis])
        cnf_matrix_error = cnf_matrix_std / Num[:, np.newaxis]

    plot_confusion_matrix(cnf_matrix_norm, cmerror=cnf_matrix_error,
                          classes=class_names, normalize=True,
                          title=('Normalized confusion matrix\n'
                                 'uncertainty in () in unit of $10^{-4}$\n'
                                 'arising from finite statistics'),
                          filename=os.path.join(output_dir, "base_ConfusionMatrixWithUncertainty.pdf") if save else None)

    plot_confusion_matrix(cnf_matrix_norm,
                          precision=3,
                          fontsize=8,
                          logColor=True,
                          classes=class_names, normalize=True,
                          filename=os.path.join(output_dir, "base_ConfusionMatrix.pdf") if save else None)

    # 绘制非对称矩阵：$M_{11} - M_{11}^T$
    aymetric = cnf_matrix_norm - cnf_matrix_norm.T
    plot_confusion_matrix(aymetric,
                          precision=4,
                          classes=class_names, normalize=False,
                          title='$M_{11}$ - transpose($M_{11}$)')

    # 绘制归一化后（按不对称误差归一化）的矩阵
    if cnf_matrix_error is not None:
        aymetric_norm = (cnf_matrix_norm - cnf_matrix_norm.T) / np.sqrt(cnf_matrix_error ** 2 + cnf_matrix_error.T ** 2)
        plot_confusion_matrix(aymetric_norm,
                              precision=1,
                              classes=class_names, normalize=False,
                              title='($M_{11}$ - transpose($M_{11}$))/Uncertainty')


def data_confusion_matrix(data, save=False, output_dir="Figures"):
    """
    利用 Data 对象绘制混淆矩阵（调用上面函数）。
    """
    return plot_confusion_matrices(data.y_true, data.y_pred, data.class_names, data.Num,
                                   save=save, output_dir=output_dir)


def data_plot_ROC(data, SigIdx, BkgIdx, label=None, show=True, fig=None, savePath=None):
    """
    绘制某一对信号与背景的 ROC 曲线。

    参数：
      - SigIdx: 信号类别索引
      - BkgIdx: 背景类别索引（可为单个索引或列表）
      - label: 曲线标签
      - show: 是否显示图像
      - fig: 指定的 matplotlib figure 对象（可选）
      - savePath: 保存路径（可选）
    """
    if fig is None:
        fig = plt.figure(figsize=(5, 5))

    if np.isscalar(BkgIdx):
        sel = (data.y_true == SigIdx) | (data.y_true == BkgIdx)
    else:
        sel = (data.y_true == SigIdx) | np.isin(data.y_true, BkgIdx)

    y_score = data.Xtotal[sel, SigIdx]
    y_true_sel = data.y_true[sel]

    fpr, tpr, _ = roc_curve(y_true_sel, y_score, pos_label=SigIdx)
    plt.plot(fpr, tpr, label=label)

    if savePath is not None:
        plt.savefig(savePath)
    if show:
        plt.show()


def datas_plot_ROC(datas, SigIdx, BkgIdx, labels=None, show=False, savePath=None):
    """
    绘制多个 Data 对象的 ROC 曲线。
    """
    fig = plt.figure(figsize=(5, 5))
    if labels is None:
        labels = [None] * len(datas)

    for data, label in zip(datas, labels):
        data_plot_ROC(data, SigIdx=SigIdx, BkgIdx=BkgIdx, label=label, show=False, fig=fig)

    plt.legend()
    if savePath is not None:
        BkgIdxStr = f"{BkgIdx}".replace("[", "(").replace("]", ")")
        fig.savefig(os.path.join(savePath, f"ROC_{SigIdx}_{BkgIdxStr}.pdf"))
    if show:
        plt.show()


def data_plot_Rej(data, SigIdx, BkgIdx, minRej=1, maxRej=100, logY=True, label=None, show=True, fig=None, savePath=None):
    """
    绘制拒识（Rejection）曲线。
    """
    if fig is None:
        fig = plt.figure(figsize=(5, 5))

    if np.isscalar(BkgIdx):
        sel = (data.y_true == SigIdx) | (data.y_true == BkgIdx)
    else:
        sel = (data.y_true == SigIdx) | np.isin(data.y_true, BkgIdx)

    y_score = data.Xtotal[sel, SigIdx]  # 使用信号得分
    y_true_sel = data.y_true[sel]

    fpr, tpr, _ = roc_curve(y_true_sel, y_score, pos_label=SigIdx)
    plt.plot(tpr, 1 / (fpr + 1e-10), label=label, alpha=0.7, linewidth=0.7)

    if logY:
        plt.yscale("log")

    plt.ylabel("Rejection Power")
    plt.xlabel(name[SigIdx] + " efficiency")
    plt.ylim((minRej, maxRej))

    if savePath is not None:
        plt.savefig(savePath)
    if show:
        plt.show()


def datas_plot_Rej(datas, SigIdx, BkgIdx, minRej=1, maxRej=100, logY=True, labels=None, show=False, savePath=None):
    """
    绘制多个 Data 对象的拒识曲线。
    """
    fig = plt.figure(figsize=(5, 5))
    if labels is None:
        labels = [None] * len(datas)

    for data, label in zip(datas, labels):
        data_plot_Rej(data, SigIdx=SigIdx, BkgIdx=BkgIdx, minRej=minRej,
                      maxRej=maxRej, logY=logY, label=label, show=False, fig=fig)

    plt.legend()
    if savePath is not None:
        BkgIdxStr = f"{BkgIdx}".replace("[", "(").replace("]", ")")
        fig.savefig(os.path.join(savePath, f"Rej_{name[SigIdx]}_rest_ir.pdf"))
    if show:
        plt.show()


def data_plot_Eff(data, SigIdx, minRej=1e-5, maxRej=1, logY=True, show=False, label=None, fig=None, savePath=None):
    """
    绘制标记效率（Efficiency）曲线（多类别 ROC）。
    对于每个对比类别，计算 ROC 曲线并绘制。
    {0: "B-jet", 1: "Bbar-jet", 2: "C-jet", 3: "Cbar-jet",
        4: "S-jet", 5: "Sbar-jet", 6: "U-jet", 7: "Ubar-jet",
        8: "D-jet", 9: "Dbar-jet", 10: "Gluon"}
    """
    if fig is None:
        fig = plt.figure(figsize=(5, 5))

    # 定义对比类别与颜色
    categories = ['b', 'c', 's', 'u', 'd', 'G']
    # categories = ['u', 'G', 's', 'd', 'c', 'b']
    colors = ['orange', 'green', 'blue', 'red', 'purple', 'grey']
    # colors = ['red', 'grey', 'blue', 'purple', 'green', 'orange']
    for idx, cat in enumerate(categories):  #
        # 选择信号和背景（这里假定信号和它的互补分支为 SigIdx 与 SigIdx+1）
        if idx * 2 == SigIdx:  # 如果 idx * 2 与 SigIdx 相同，则跳过
            continue
        sel = np.isin(data.y_true, [SigIdx, SigIdx + 1]) | np.isin(data.y_true, [idx * 2, idx * 2 + 1])  #
        y_true_sel = data.y_true[sel]
        # 将标签二值化：事件是否为信号（True）或背景（False）
        y_true_binary = (y_true_sel == SigIdx) | (y_true_sel == SigIdx + 1)

        xscore = data.Xtotal[sel, SigIdx]
        # 对于标记效率，若存在配对分支，则合并两者得分
        if SigIdx < 10:
            xbar_score = data.Xtotal[sel, SigIdx + 1]
        else:
            xbar_score = xscore
        y_score = xscore + xbar_score

        fpr, tpr, _ = roc_curve(y_true_binary, y_score, pos_label=1)
        plt.plot(tpr, fpr, label=f'{cat} as {name[SigIdx]}', color=colors[idx], linewidth=2)
        if logY:
            plt.yscale("log")

    # plt.title("ROC Curves for Multi-class Classification")
    plt.xlabel("jet tagging efficiency")
    plt.ylabel("jet misid probability")
    plt.ylim((minRej, maxRej))
# new order of legend
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [2, 4, 3, 1, 0]
    new_handles = [handles[i] for i in order]
    new_labels = [labels[i] for i in order]
    plt.legend(loc="lower right",handles=new_handles, labels=new_labels, fontsize=14)
    plt.grid(True)
    if logY:
        plt.yscale("log")
    if savePath is not None:
        fig.savefig(savePath)
    if show:
        plt.show()


def datas_plot_Eff(data, SigIdx, label=None, minRej=1e-4, maxRej=1, logY=True, show=False, savePath=None):
    """
    绘制单个 Data 对象的标记效率曲线（封装上面函数）。
    """
    fig = plt.figure(figsize=(5, 5))
    data_plot_Eff(data, SigIdx=SigIdx, minRej=minRej, maxRej=maxRej,
                  logY=logY, show=False, label=label, fig=fig)
    #
    if savePath is not None:
        fig.savefig(os.path.join(savePath, f"{label}_{name[SigIdx]}_eff.png"))
        print(f"Saving figure: {label}_{name[SigIdx]}_eff.png")
        plt.close(fig)
    if show:
        plt.show()


# ----------------------------

if __name__ == "__main__":
    # 示例：更新以下路径为实际合并 ROOT 文件的路径和输出目录
    merged_file_path = ".../pred.root"
    output_directory = merged_file_path.replace("pred.root", "figures")

    data = load_merged_data(merged_file_path)
    data_confusion_matrix(data, save=True, output_dir=output_directory)


    SigIdxs = [0, 2, 4, 6, 8, 10]
    for SigIdx in SigIdxs:
        datas_plot_Eff(data, SigIdx=SigIdx, label="Eff_example", show=False, savePath=output_directory)


