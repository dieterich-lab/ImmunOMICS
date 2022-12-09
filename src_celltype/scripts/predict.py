from sklearn.preprocessing import LabelEncoder, minmax_scale
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib.backends.backend_pdf import PdfPages
import sklearn.metrics as skm
import matplotlib.pyplot as plt
from collections import defaultdict
import shap
from numba import njit, prange
import seaborn as sns
from statannot import add_stat_annotation

# For SHAP package should disable v2 tensorflow
tf.compat.v1.disable_v2_behavior()

x_exp = pd.DataFrame([])
for elem in snakemake.input["GE"]:
    x_ = pd.read_csv(elem, index_col=0)
    x_exp = pd.concat([x_exp, x_], axis=0)

# Load data and models
model_e_f = snakemake.input["model_e"]
svm_e_f = snakemake.input["svm_e"]
LogReg_e_f = snakemake.input["LogReg_e"]
RF_e_f = snakemake.input["RF_e"]
out_fig = snakemake.output["fig"]


def confidence_interval(values):
    mean = np.mean(values)
    alpha = 0.95
    p = ((1.0 - alpha) / 2.0) * 100
    bottom = max(0.0, np.percentile(values, p))
    p = (alpha + ((1.0 - alpha) / 2.0)) * 100
    top = min(1.0, np.percentile(values, p))
    return mean, bottom, top


def eval_box(metrics, tit):
    f = plt.figure()
    val_vec = list()
    bt_vec = list()
    tp_vec = list()
    x_vec = list()
    for d in metrics:
        val, bt, tp = confidence_interval(metrics[d])
        val_vec.append(val)
        bt_vec.append(val - bt)
        tp_vec.append(tp - val)
        x_vec.append(d)

    with plt.rc_context({"figure.figsize": (4, 3), "figure.dpi": 300, "font.size": 16}):
        plt.errorbar(
            x_vec,
            val_vec,
            yerr=(bt_vec, tp_vec),
            fmt="o",
            capsize=5,
            ecolor="k",
            lw=3,
            ls=":",
            color="blue",
        )
        plt.title(tit)
        plt.ylim(0.4, 1.05)
        plt.plot([], c="k", label="CI 95%")
        plt.plot([], c="blue", label="mean")
        plt.legend(loc="lower right")

    return f


def eval_box_sbn(metrics, tit):
    f = plt.figure()
    data = pd.DataFrame([], columns=["modality", "value"])
    for d in metrics:
        data_ = pd.DataFrame(
            [np.full((len(metrics[d])), d), metrics[d]], index=["modality", "value"]
        )
        data = pd.concat([data, data_.transpose()], ignore_index=True)
    data.value = data.value.astype("float")
    mod = data.modality.unique()
    if len(mod) == 3:
        pairs = [("CC", "GE"), ("CC", "MLP"), ("GE", "MLP")]
    else:
        pairs = [("MLP", "SVM"), ("MLP", "RF"), ("MLP", "LogisticR")]
    with plt.rc_context({"figure.figsize": (10, 8), "figure.dpi": 96, "font.size": 16}):

        g = sns.boxplot(x="modality", y="value", data=data)

        add_stat_annotation(
            g,
            x="modality",
            y="value",
            data=data,
            box_pairs=pairs,
            test="Mann-Whitney",
            text_format="star",
            loc="inside",
            verbose=2,
        )
        plt.ylabel(tit)
        if len(mod) == 3:
            plt.xlabel("modality")
        else:
            plt.xlabel("model")

    return f


def compute_metrics(y):
    metrics = defaultdict(list)
    col_len = y.shape[1]
    for i in range(col_len):
        l = y.columns[i]
        metrics["auc"].append(skm.roc_auc_score(Ytest, y[l]))
        metrics["acc"].append(skm.accuracy_score(Ytest, y[l] >= 0.5))
        metrics["f1"].append(skm.f1_score(Ytest, y[l] >= 0.5))
        metrics["rec"].append(skm.recall_score(Ytest, y[l] >= 0.5))
        metrics["prc"].append(skm.precision_score(Ytest, y[l] >= 0.5))
        metrics["auprc"].append(skm.average_precision_score(Ytest, y[l]))
    return metrics


def predict_loop(model_, data):
    y_score1 = pd.DataFrame([])
    model_len = len(model_)
    for i in range(model_len):
        model = model_[i]
        y_score1["sampling" + str(i)] = model.predict(data).flatten()
    return y_score1


def shap_loop(model_j, training_set, dim_exp, x_exp):
    model_len = len(model_j)
    for i in range(model_len):
        model_joint = model_j[i]
        x_ref_exp = training_set[i][:, :dim_exp]
        explainer = shap.DeepExplainer(model_joint, x_ref_exp)
        shap_values = explainer.shap_values(np.array(x_exp))
        if model_joint == model_j[0]:
            shap_values_all_exp = shap_values[0]
        else:
            shap_values_all_exp = shap_values_all_exp + shap_values[0]
    return shap_values_all_exp


if __name__ == "__main__":
    if x_exp.shape[0] > 0:

        with open(model_e_f, "rb") as b:
            model_e = pickle.load(b)
        with open(svm_e_f, "rb") as b:
            svm_e = pickle.load(b)
        with open(LogReg_e_f, "rb") as b:
            LogReg_e = pickle.load(b)
        with open(RF_e_f, "rb") as b:
            RF_e = pickle.load(b)

        # prepare the data
        x_exp = x_exp.loc[x_exp["condition"].isin(["Mild", "Severe"]), :]

        label = x_exp.iloc[:, -2].values
        x_exp = x_exp.drop("condition", axis=1)
        x_exp = x_exp.drop("who_score", axis=1)

        genes = x_exp.columns

        le = LabelEncoder()
        Ytest = le.fit_transform(label)

        x_exp = x_exp / 15

        # perform predictions
        y_score2_RF = predict_loop(RF_e, x_exp)

        y_score2 = predict_loop(model_e, x_exp)

        y_score2_svm = predict_loop(svm_e, x_exp)

        #     print(LogReg_c.values())
        y_score2_LogReg = predict_loop(LogReg_e, x_exp)

        # compute metrics and plot results
        all_yscores = {
            out_fig: [y_score2],
            "svm_fig": [y_score2_svm],
            "LogReg_fig": [y_score2_LogReg],
            "RF_fig": [y_score2_RF],
        }
        comp = {}
        for key in all_yscores.keys():
            #         if key==out_fig:
            all_yscores[key][0].to_csv(key.replace(".pdf", "_GE.csv"))
            res_e = compute_metrics(all_yscores[key][0])
            comp[key] = res_e
            # compute mean and CI
            all_met = pd.DataFrame([])
            for d in res_e:
                val, bt, tp = confidence_interval(res_e[d])
                all_met[d] = [val, bt, tp]
            all_met.index = ["mean", "lower CI", "upper CI"]
            all_met.transpose().to_csv(key.replace(".pdf", "_GE.txt"))

        fig14 = eval_box_sbn(
            {
                "LogisticR": comp["LogReg_fig"]["auc"],
                "SVM": comp["svm_fig"]["auc"],
                "RF": comp["RF_fig"]["auc"],
                "MLP": comp[out_fig]["auc"],
            },
            "AUC",
        )
        fig15 = eval_box_sbn(
            {
                "LogisticR": comp["LogReg_fig"]["auprc"],
                "SVM": comp["svm_fig"]["auprc"],
                "RF": comp["RF_fig"]["auprc"],
                "MLP": comp[out_fig]["auprc"],
            },
            "Average Precision",
        )
        fig16 = eval_box_sbn(
            {
                "LogisticR": comp["LogReg_fig"]["acc"],
                "SVM": comp["svm_fig"]["acc"],
                "RF": comp["RF_fig"]["acc"],
                "MLP": comp[out_fig]["acc"],
            },
            "Accuracy",
        )
        fig17 = eval_box_sbn(
            {
                "LogisticR": comp["LogReg_fig"]["prc"],
                "SVM": comp["svm_fig"]["prc"],
                "RF": comp["RF_fig"]["prc"],
                "MLP": comp[out_fig]["prc"],
            },
            "Precision",
        )
        fig18 = eval_box_sbn(
            {
                "LogisticR": comp["LogReg_fig"]["rec"],
                "SVM": comp["svm_fig"]["rec"],
                "RF": comp["RF_fig"]["rec"],
                "MLP": comp[out_fig]["rec"],
            },
            "Recall",
        )
        fig19 = eval_box_sbn(
            {
                "LogisticR": comp["LogReg_fig"]["f1"],
                "SVM": comp["svm_fig"]["f1"],
                "RF": comp["RF_fig"]["f1"],
                "MLP": comp[out_fig]["f1"],
            },
            "F1-Score",
        )

        all_yscores_ = defaultdict(list)
        for key in all_yscores.keys():
            if key == out_fig:
                all_yscores_[key.split("/")[-1] + "GE"] = all_yscores[key][0].mean(
                    axis=1
                )

        fig20 = plt.figure()
        with plt.rc_context(
            {"figure.figsize": (6, 5), "figure.dpi": 300, "font.size": 10}
        ):
            for key in all_yscores_.keys():
                ns_auc = skm.roc_auc_score(Ytest, all_yscores_[key])
                # calculate roc curves
                lr_fpr, lr_tpr, _ = skm.roc_curve(Ytest, all_yscores_[key])
                # plot the roc curve for the model
                plt.plot(
                    lr_fpr, lr_tpr, marker=".", label=key + ": " + str(round(ns_auc, 2))
                )
                # axis labels
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                # show the legend
                plt.legend()
                # show the plot
        #         plt.show()

        fig21 = plt.figure()
        with plt.rc_context(
            {"figure.figsize": (6, 5), "figure.dpi": 300, "font.size": 10}
        ):
            for key in all_yscores_.keys():

                lr_precision, lr_recall, _ = skm.precision_recall_curve(
                    Ytest, all_yscores_[key]
                )
                lr_auc = skm.auc(lr_recall, lr_precision)
                avr_prec = skm.average_precision_score(Ytest, all_yscores_[key])
                # plot the precision-recall curves
                plt.plot(
                    lr_recall,
                    lr_precision,
                    marker=".",
                    label=key
                    + ": "
                    + str(round(lr_auc, 2))
                    + "_"
                    + str(round(avr_prec, 2)),
                )
                # axis labels
                plt.xlabel("Recall")
                plt.ylabel("Precision")
                # show the legend
                plt.legend()
                # show the plot
        #         plt.show()

        pp = PdfPages(out_fig)
        pp.savefig(fig14, bbox_inches="tight")
        pp.savefig(fig15, bbox_inches="tight")
        pp.savefig(fig16, bbox_inches="tight")
        pp.savefig(fig17, bbox_inches="tight")
        pp.savefig(fig18, bbox_inches="tight")
        pp.savefig(fig19, bbox_inches="tight")
        pp.savefig(fig20, bbox_inches="tight")
        pp.savefig(fig21, bbox_inches="tight")

        pp.close()

        # Compute and plot SHAP values
        dim_exp = x_exp.shape[1]

        with open(snakemake.input["training"], "rb") as b:
            training_set = pickle.load(b)
        shap_values_all_exp = shap_loop(list(model_e), training_set, dim_exp, x_exp)

        nb = len(model_e)
        #         f1 = plt.figure()
        #         with plt.rc_context({'figure.figsize': (4, 3), 'figure.dpi':300}):
        #             shap.summary_plot(shap_values_all_exp/nb, plot_type= 'violin',features=np.array(x_exp)
        #                               , feature_names =genes,color_bar_label='Feature value',show=False)
        f2 = plt.figure()

        with plt.rc_context({"figure.figsize": (4, 3), "figure.dpi": 300}):
            shap.summary_plot(
                shap_values_all_exp / nb,
                plot_type="bar",
                features=np.array(x_exp),
                feature_names=genes,
                color_bar_label="Feature value",
                show=False,
            )
        #         f3 = plt.figure()

        #         with plt.rc_context({'figure.figsize': (4, 3), 'figure.dpi':300}):
        #             shap.summary_plot(shap_values_all_cell/nb, plot_type= 'violin',features=x_cell
        #                               , feature_names =selected_cols,color_bar_label='Feature value',show=False, max_display=15)
        #         f4 = plt.figure()
        #         with plt.rc_context({'figure.figsize': (4, 3), 'figure.dpi':300}):
        #             shap.summary_plot(shap_values_all_cell/nb, plot_type= 'bar',features=np.array(x_cell)
        #                               , feature_names =selected_cols,color_bar_label='Feature value',show=False, max_display=15)

        pp = PdfPages(out_fig + "shap.pdf")
        #         pp.savefig(f1, bbox_inches='tight')
        pp.savefig(f2, bbox_inches="tight")
        #         pp.savefig(f3, bbox_inches='tight')
        #         pp.savefig(f4, bbox_inches='tight')
        pp.close()
    else:
        pp = PdfPages(out_fig)
        pp.close()
#         pp = PdfPages(out_fig)
#         pp.close()
#         pp = PdfPages(svm_fig)
#         pp.close()
#         pp = PdfPages(LReg_fig)
#         pp.close()
#         pp = PdfPages(RF_fig)
#         pp.close()
#         pp = PdfPages(LogReg_fig)
#         pp.close()
