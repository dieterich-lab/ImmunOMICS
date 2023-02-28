from sklearn.preprocessing import LabelEncoder, minmax_scale
import pickle
import numpy as np
import pandas as pd
import random
import os
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from sklearn.utils import resample
from keras.models import load_model
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.feature_selection import SelectKBest, f_classif
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from collections import defaultdict
# from hyperopt import SparkTrials, STATUS_OK, tpe, fmin, hp
from joblib import Parallel, delayed

# Load data
x_exp = pd.read_csv(snakemake.input["GE"], index_col=0)
# Outputs
train_set_f = snakemake.output["train_set"]
val_set_f = snakemake.output["val_set"]
model_e_f = snakemake.output["model_e"]
svm_e_f = snakemake.output["svm_e"]
LogReg_e_f = snakemake.output["LogReg_e"]
RF_e_f = snakemake.output["RF_e"]
path = snakemake.params[0]


def reset_random_seeds():
    os.environ["PYTHONHASHSEED"] = str(1)
    tf.random.set_seed(0)
    np.random.seed(1234)


def training(model, trainDataOne, y, valid_set, ref):
    reset_random_seeds()
    # Optimizer setting
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    # Model compiling settings
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=["accuracy"],
    )

    # A mechanism that stops training if the validation loss is not improving for more than n_idle_epochs.
    n_idle_epochs = 100
    earlyStopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=n_idle_epochs, min_delta=0.001
    )
    mc = ModelCheckpoint(
        model_e_f + ref + ".h5", monitor="val_loss", mode="min", save_best_only=True
    )

    # Creating a custom callback to print the log after a certain number of epochs
    class NEPOCHLogger(tf.keras.callbacks.Callback):
        def __init__(self, per_epoch=100):
            """
            display: Number of batches to wait before outputting loss
            """
            self.seen = 0
            self.per_epoch = per_epoch

        def on_epoch_end(self, epoch, logs=None):
            if epoch % self.per_epoch == 0:
                print(
                    "Epoch {}, loss {:.2f}, val_loss {:.2f}, accuracy {:.2f}, val_accuracy {:.2f}".format(
                        epoch,
                        logs["loss"],
                        logs["val_loss"],
                        logs["accuracy"],
                        logs["val_accuracy"],
                    )
                )

    log_display = NEPOCHLogger(per_epoch=100)
    # Training loop
    n_epochs = 2000
    history = model.fit(
        trainDataOne,
        y,
        batch_size=16,
        epochs=n_epochs,
        validation_data=valid_set,
        verbose=0,
        callbacks=[log_display, earlyStopping, mc],
    )
    return model, history


def build_1modality_classifier(inp1):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Dropout(0.2, seed=0, input_shape=(inp1,)),
            tf.keras.layers.Dense(
                2,
                activation="relu",
                kernel_initializer=tf.keras.initializers.GlorotNormal(seed=1234),
            ),
            tf.keras.layers.Dense(
                1,
                activation="sigmoid",
                kernel_initializer=tf.keras.initializers.GlorotNormal(seed=1234),
            ),
        ]
    )
    return model


# sampling taking into account WHO score (all modalities should be included in training validation sets for all samplings)
def provide_stratified_bootstap_sample_indices(bs_sample, percent):
    strata = pd.DataFrame(who).value_counts()
    bs_index_list_stratified = []
    for idx_stratum_var, n_stratum_var in strata.iteritems():
        data_index_stratum = list(np.where(who == idx_stratum_var[0])[0])
        kk = round(len(data_index_stratum) * percent)
        bs_index_list_stratified.extend(random.sample(data_index_stratum, k=kk))
    return bs_index_list_stratified


# training task
def task(bs_list_stratified):
    bs_index_list_stratified = bs_list_stratified
    res_task = {}
    train = sets[bs_index_list_stratified, :]
    res_task["train"] = train
    test = np.array([x for x in sets if x.tolist() not in train.tolist()])
    res_task["test"] = test
    ref = str(random.random())
    y = train[:, -1]

    # build and train and load best 1 modality classifier for gene expression GE
    train_set = train[:, :dim_exp]
    model = build_1modality_classifier(dim_exp)
    model, res_task["history"] = training(
        model, train_set, y, (test[:, :dim_exp], test[:, -1]), ref
    )
    res_task["model_e"] = load_model(model_e_f + ref + ".h5")

    return res_task


# run 30 tasks (samplings) in Parallel
def train_loop():
    n_iterations = 30
    random.seed(1234)
    seeds = random.sample(range(0, 1000), n_iterations)
    stratified_all = list()
    # startify 80% training 20% validation
    for i in range(n_iterations):
        random.seed(seeds[i])
        stratified_all.append(provide_stratified_bootstap_sample_indices(sets, 0.8))
    res_ = defaultdict(list)
    results = Parallel(n_jobs=10)(delayed(task)(i) for i in stratified_all)
    for result in results:
        for key in result.keys():
            res_[key].append(result[key])
    return res_



# Evaluate baseline models
def task_comaparative(bs_list_stratified):
    bs_index_list_stratified = bs_list_stratified
    res_task = {}
    train = sets[bs_index_list_stratified, :]
    test = np.array([x for x in sets if x.tolist() not in train.tolist()])
    ref = str(random.random())
    y = train[:, -1]
    # evaluate GE model
    train_set = train[:, :dim_exp]
    models = {
        "LogReg_e": LogisticRegression(),
        "svm_e": svm.SVC(),
        "RF_e": RandomForestClassifier(),
    }
    for key in models.keys():
        res_task[key] = models[key].fit(train_set, y)

    return res_task


# run 30 comaparative tasks (samplings) in Parallel
def compare():
    n_iterations = 30
    random.seed(1234)
    seeds = random.sample(range(0, 1000), n_iterations)
    stratified_all = list()
    for i in range(n_iterations):
        random.seed(seeds[i])
        stratified_all.append(provide_stratified_bootstap_sample_indices(sets, 0.8))

    res_ = defaultdict(list)
    results = Parallel(n_jobs=6)(delayed(task_comaparative)(i) for i in stratified_all)
    for result in results:
        for key in result.keys():
            res_[key].append(result[key])
    return res_


if __name__ == "__main__":
    if x_exp.shape[0] > 0:
        res = defaultdict(list)
        # Format matrices
        x_exp = x_exp.loc[x_exp["condition"].isin(["Mild", "Severe"]), :]
        label = x_exp.iloc[:, -2].values
        who = x_exp.iloc[:, -1].values
        x_exp = x_exp.drop("condition", axis=1)
        x_exp = x_exp.drop("who_score", axis=1)
        genes = x_exp.columns
        le = LabelEncoder()
        Ytrain = le.fit_transform(label)
        # scale GE data to 0-1 knowing that the Gene expession is between 0 and 15
        x_exp = x_exp / 15
        sets = np.column_stack((x_exp, Ytrain))
        dim_exp = x_exp.shape[1]
        # train the models CC&GE, GE and CC with hyperopt optimization tool
        res = train_loop()
        # plot loss history
        pp = PdfPages(model_e_f + "history_loss.pdf")
        for his in res["history"]:
            f = plt.figure()
            # summarize history for loss
            plt.plot(his.history["loss"])
            plt.plot(his.history["val_loss"])
            plt.title("model loss")
            plt.ylabel("loss")
            plt.xlabel("epoch")
            plt.legend(["train", "test"], loc="upper left")
            pp.savefig(f, bbox_inches="tight")
        pp.close()

        pp = PdfPages(model_e_f + "history_acc.pdf")
        for his in res["history"]:
            f = plt.figure()
            # summarize history for loss
            plt.plot(his.history["accuracy"])
            plt.plot(his.history["val_accuracy"])
            plt.title("model accuracy")
            plt.ylabel("accuracy")
            plt.xlabel("epoch")
            plt.legend(["train", "test"], loc="upper left")
            pp.savefig(f, bbox_inches="tight")
        pp.close()
        # train baseline models Linear, logistic, SVM and RF
        res_comparative = compare()
        # concatenate all results in one dict
        res = {**res, **res_comparative}
        # Save results
        map_name_file = {
            "test": val_set_f,
            "train": train_set_f,
            "model_e": model_e_f,
            "svm_e": svm_e_f,
            "RF_e": RF_e_f,
            "LogReg_e": LogReg_e_f,
        }
        for key in map_name_file.keys():
            with open(map_name_file[key], "wb") as b:
                pickle.dump(res[key], b)
        # remove tmp files
        l_h5 = os.listdir(path)
        for item in l_h5:
            if item.endswith(".h5"):
                os.remove(os.path.join(path, item))

    else:
        map_name_file = {
            "test": val_set_f,
            "train": train_set_f,
            "model_e": model_e_f,
            "svm_e": svm_e_f,
            "RF_e": RF_e_f,
            "LogReg_e": LogReg_e_f,
        }
        for key in map_name_file.keys():
            with open(map_name_file[key], "wb") as b:
                pickle.dump(None, b)
