##########################################################
#########################################################


# Imports:
import numpy as np
import pandas as pd
import functools, operator

import matplotlib.pyplot as plt
from matplotlib import style
from seaborn import heatmap



from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_classif
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedShuffleSplit as SSS
from sklearn.model_selection import ParameterGrid
from sklearn.decomposition import TruncatedSVD

from progressbar import ProgressBar

import matplotlib.pyplot as plt
from matplotlib import style
from seaborn import heatmap, kdeplot




################################################################
################################################################
# BASIC ML STARTER FNS:


def fn_tr_ts_split_clf(df_Xy_, ts_size=0.2, rand_state=63):
    df_Xy = df_Xy_
    df_X, y = df_Xy.iloc[:, :-1], df_Xy.iloc[:, -1].values

    sss = SSS(n_splits=1, test_size=ts_size, random_state=rand_state).split(df_X, y)
    tr_idxs, ts_idxs = list(sss)[0]

    return tr_idxs, ts_idxs


def fn_tr_eval_ts_split_clf(df_Xy_, eval_size=0.2, ts_size=0.2):
    idxs_tr, idxs_ts_ = fn_tr_ts_split_clf(df_Xy_, ts_size=ts_size + eval_size)

    df_tr = df_Xy_.iloc[idxs_tr]
    df_ts_ = df_Xy_.iloc[idxs_ts_]

    idxs_eval, idxs_ts = fn_tr_ts_split_clf(df_ts_, ts_size=ts_size / (ts_size + eval_size))

    df_eval = df_ts_.iloc[idxs_eval]
    df_ts = df_ts_.iloc[idxs_ts]

    return df_tr, df_eval, df_ts


def fn_distr_labels_clf(labels_, legend=None, width=0.5):
    df_y = pd.DataFrame(labels_, columns=['labels'])
    dictO_labels = dict(df_y.labels.value_counts())
    d = dictO_labels

    z = [f'CLASS_{k} : {d[k]} ( {np.round((100 * d[k] / len(df_y)), 2)} %)\n' for k in d]
    title = functools.reduce(operator.add, z)

    kwargs = dict(kind='bar', alpha=0.6, label=legend, width=width)
    df_y.labels.value_counts().plot(**kwargs)
    plt.title(title)

    if legend != None:
        plt.legend(loc='upper right')
    plt.xticks(size=15)


def fn_standardize_df(df_tr_, to_transform=[]):
    def fn_X(df): return df.iloc[:, :-1].values

    def fn_y(df): return df.iloc[:, -1].values

    def fn_df(X, y):
        cols = df_tr_.columns[:-1]
        return pd.DataFrame(X, columns=cols).assign(labels=y)

    Xs = [fn_X(df) for df in [df_tr_, *to_transform]]
    ys = [fn_y(df) for df in [df_tr_, *to_transform]]

    scaler = StandardScaler().fit(Xs[0])
    transformed_Xs = [scaler.transform(X) for X in Xs]

    dfs = [fn_df(X, y) for X, y in zip(transformed_Xs, ys)]
    return [*dfs, scaler]


##################################################################
##################################################################
# DIMENSIONALITY REDUCTION:

def fn_pca_df(df_tr_, reduce_dims=3, to_transform=[], n_iter=7):
    def fn_X(df): return df.iloc[:, :-1].values

    def fn_y(df): return df.iloc[:, -1].values

    def fn_df(X, y):
        cols = ['f' + str(i) for i in range(X.shape[1])]
        return pd.DataFrame(X, columns=cols).assign(labels=y)

    Xs = [fn_X(df) for df in [df_tr_, *to_transform]]
    ys = [fn_y(df) for df in [df_tr_, *to_transform]]

    pca_tranformer = TruncatedSVD(n_components=reduce_dims, n_iter=n_iter, random_state=42).fit(Xs[0])
    transformed_Xs = [pca_tranformer.transform(X) for X in Xs]

    dfs = [fn_df(X, y) for X, y in zip(transformed_Xs, ys)]
    return [*dfs, pca_tranformer]


##################################################################
# 3D SCATTER IN 4 PERSPECTIVES:

def fn_plot_3d_clf(df_Xy_, class_sample_fr_vec=None,
                   alpha=0.4, figsize=(13, 10),
                   listO_views=[(10, 5), (-40, 0), (70, 0), (10, 80)]):
    '''
    class_sample_fr_vec = fracO_each_class_2B_sampled (default 100%).
    '''
    n_views = len(listO_views)
    z1, z2 = n_views / 2, n_views // 2
    z3 = z1 - z2
    n_rows = z2 if (z3 == 0) else z2 + 1
    subplot_grid = (n_rows, 2)

    df_3X = df_Xy_.iloc[:, :3]
    labels = df_Xy_.iloc[:, -1]
    df = df_3X.assign(labels=labels)
    labels_unique = df.labels.unique()
    df.index = range(len(df))
    cols = df_3X.columns

    fig = plt.figure(figsize=figsize)
    z = class_sample_fr_vec
    class_sample_fr_vec = [1 for i in labels_unique] if z == None else z
    for idx, view in enumerate(listO_views):

        ax = fig.add_subplot(*subplot_grid, idx + 1, projection='3d')
        for klass, frac in zip(labels_unique, class_sample_fr_vec):
            dff = df[df.labels == klass].sample(frac=frac)
            x1, x2, x3 = dff.iloc[:, 0].values, dff.iloc[:, 1].values, dff.iloc[:, 2].values
            ax.scatter(x1, x2, x3, marker='o', s=30, alpha=alpha, label=f'{klass}, n_pts: {len(dff)}')

        ax.set_xlabel(cols[0], fontsize=10, labelpad=10)
        ax.set_ylabel(cols[1], fontsize=10, labelpad=10)
        ax.set_zlabel(cols[2], fontsize=10, labelpad=10)
        ax.view_init(*view)

    plt.legend(loc='upper right', prop=dict(size=12, weight='bold'))
    plt.tight_layout()


#################################################################
#################################################################
# CORRELATION/FEATURE SELECTION/VISUALIZTION:


def fn_feat_importance(df_X_tr, y_tr):
    f_ratios, zzz = f_classif(df_X_tr, y_tr)
    f_ratios = pd.Series(f_ratios, index=df_X_tr.columns)

    return f_ratios.sort_values(ascending=False)


def fn_corr_matrix(df_X_tr, f_ratios):
    zz = list(f_ratios.index)
    df_corr_mat = df_X_tr.corr(method='spearman').abs()
    df_corr = df_corr_mat.loc[zz]

    return df_corr.loc[:, zz]


def fn_filter_feats(df_corr, thresh=0.8):
    dff = df_corr.copy()
    collect_good_feats = []
    cols = list(dff.columns)

    while True:

        col = cols[0]
        s = dff.loc[:, col]
        ss = s[s.values >= thresh].index
        ss = list(ss)

        dff = dff.drop(ss)
        dff = dff.drop(ss, axis=1)

        [cols.remove(i) for i in ss]
        collect_good_feats.append(col)
        if len(cols) < 2:
            break

    return df_corr.loc[collect_good_feats].loc[:, collect_good_feats]


def fn_plot_corr_numerical(f_ratios, df_corr, figsize=(15, 7), fontsize=15):
    plt.figure(figsize=figsize)

    plt.subplot(1, 2, 1)
    f_ratios.sort_values().plot(kind='barh', alpha=0.6)
    plt.title('FEATURE_LABEL_CORRELATIONS (anova - f - values)')
    plt.xlabel('Degree_of_Correlation', fontsize=fontsize)
    plt.xticks(fontsize=fontsize - 2)
    plt.yticks(fontsize=fontsize - 2)

    plt.subplot(1, 2, 2)
    heatmap(df_corr, annot=False, cmap="YlGnBu")
    plt.title('FEATURE_FEATURE_CORRELATIONS (SPEARMANS)')
    plt.xticks(fontsize=0)
    plt.yticks(fontsize=0, rotation=0)
    plt.tight_layout()
    plt.show()


def fn_feat_select_clfn(df_tr, n_feats=None, plot=True,
                        thresh_feat_label=None,
                        thresh_feat_feat=None,
                        figsize=(15, 7)):
    n_feats = -1 if n_feats == None else n_feats
    df_X, y = df_tr.iloc[:, :n_feats], df_tr.iloc[:, -1].values
    f_ratios = fn_feat_importance(df_X, y)
    df_corr = fn_corr_matrix(df_X, f_ratios)

    if thresh_feat_label == None and thresh_feat_feat == None:
        best_feats = list(f_ratios.index)

    else:
        df_corr = fn_filter_feats(df_corr, thresh=thresh_feat_feat)

        f_ratios = f_ratios[df_corr.index]
        f_ratios = f_ratios[f_ratios.values >= thresh_feat_label]
        best_feats = list(f_ratios.index)

        df_corr = df_corr.loc[best_feats].loc[:, best_feats]

    if plot: fn_plot_corr_numerical(f_ratios, df_corr, figsize=figsize)

    return f_ratios, df_corr, best_feats


def fn_distr_feats_labels(df_tr_, n_top_feats=4, figsize=(15, 12)):
    def fn_plot_dist_feat(feat, labels, idx):

        dictO_xs = {}
        for label in labels:
            x = df_tr[df_tr.labels == label][feat].values
            label = str(label)
            dictO_xs[feat + '_' + label] = x

        for k in dictO_xs:
            title_, label_ = k.split('_')[0], k.split('_')[-1]
            kdeplot(dictO_xs[k], shade=True, label=label_)
            plt.title(title_)
            if idx == 0:
                plt.legend(bbox_to_anchor=(1.125, 1), ncol=1, fancybox=True, fontsize=12)

    df_tr = df_tr_.dropna(axis=1)
    labels = df_tr.iloc[:, -1].unique()
    idx = 0
    plt.figure(figsize=figsize)

    for feat in df_tr.columns[:n_top_feats]:
        plt.subplot(n_top_feats, 1, idx + 1)
        fn_plot_dist_feat(feat, labels, idx)
        idx += 1
    plt.tight_layout()


####################################################################
####################################################################
# TRAINING/EVALUATION/VISUALIZATION:

def fn_param_grid(param_grid_):
    return ParameterGrid(param_grid_)


def fn_train_models(X_std, y, model_class, param_grid):
    X = X_std
    trained_models = []
    pbar = ProgressBar()
    for hyp_params in pbar(param_grid):
        trained_model = model_class(**hyp_params).fit(X, y)
        trained_models.append(trained_model)
    trained_models = pd.Series(trained_models)
    return trained_models


def fn_pred_proba(model, X):
    if hasattr(model, "predict_proba"):
        prob_pos = model.predict_proba(X)[:, 1]
    else:  # For model without pred_proba
        prob_pos = model.decision_function(X)
        prob_pos = (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())

    return prob_pos


def fn_acc_prec_rec(y, y_proba, thresh):
    y_pred = np.array([1 if i > thresh else 0 for i in y_proba])
    dff = pd.DataFrame().assign(y=y, y_pred=y_pred)

    TP_1 = sum([1 for i in dff[dff.y == 1].y_pred if i == 1]) + 1e-6
    FP_1 = sum([1 for i in dff[dff.y == 0].y_pred if i != 0]) + 1e-6
    FN_1 = sum([1 for i in dff[dff.y == 1].y_pred if i != 1]) + 1e-6
    prec_1, rec_1 = TP_1 / (TP_1 + FP_1), TP_1 / (TP_1 + FN_1)

    TP_0 = sum([1 for i in dff[dff.y == 0].y_pred if i == 0]) + 1e-6
    FP_0 = sum([1 for i in dff[dff.y == 1].y_pred if i != 1]) + 1e-6
    FN_0 = sum([1 for i in dff[dff.y == 0].y_pred if i != 0]) + 1e-6
    prec_0, rec_0 = TP_0 / (TP_0 + FP_0), TP_0 / (TP_0 + FN_0)

    acc = (TP_1 + TP_0) / len(y_pred)

    return acc, prec_0, prec_1, rec_0, rec_1


def fn_performance_metrics(y, y_proba, listO_thresholds):
    listO_metrics = []
    for thresh in listO_thresholds:
        acc, prec_0, prec_1, rec_0, rec_1 = fn_acc_prec_rec(y, y_proba, thresh)
        listO_metrics.append([acc, prec_0, prec_1, rec_0, rec_1, thresh])

    df_performance_metrics = pd.DataFrame(np.array(listO_metrics))
    df_performance_metrics.columns = ['acc', 'prec_0', 'prec_1', 'rec_0', 'rec_1', 'thresh']
    df_performance_metrics.sort_values(by='thresh')

    return df_performance_metrics


def fn_pr_rec_tr_eval(y_tr, y_tr_proba, y_eval, y_eval_proba,
                      class_, listO_thresholds=np.linspace(0, 1, 10).round(2)):
    df_metrics_tr = fn_performance_metrics(y_tr, y_tr_proba, listO_thresholds)
    df_metrics_eval = fn_performance_metrics(y_eval, y_eval_proba, listO_thresholds)

    if class_ == 1:
        tr_prec, tr_rec = df_metrics_tr.prec_1, df_metrics_tr.rec_1
        eval_prec, eval_rec = df_metrics_eval.prec_1, df_metrics_eval.rec_1
    if class_ == 0:
        tr_prec, tr_rec = df_metrics_tr.prec_0, df_metrics_tr.rec_0
        eval_prec, eval_rec = df_metrics_eval.prec_0, df_metrics_eval.rec_0

    acc_tr, acc_eval = df_metrics_tr.acc, df_metrics_eval.acc

    subplot_grid = (1, 2)
    fig, axes = plt.subplots(*subplot_grid, figsize=(12, 4), sharey=True)
    axes = axes.ravel()

    axes[0].plot(listO_thresholds, tr_prec, label='precision')
    axes[0].plot(listO_thresholds, tr_rec, label='recall')
    axes[0].plot(listO_thresholds, acc_tr, linestyle='--', label='accuracy')
    axes[0].set_xlabel('thresholds')
    axes[0].set_title('TRAIN - ' + 'class_' + str(class_))

    axes[1].plot(listO_thresholds, eval_prec, label='precision')
    axes[1].plot(listO_thresholds, eval_rec, label='recall')
    axes[1].plot(listO_thresholds, acc_eval, linestyle='--', label='accuracy')
    axes[1].set_xlabel('thresholds')
    axes[1].set_title('EVAL - ' + 'class_' + str(class_))

    plt.legend()
    plt.tight_layout()


##########################################################################
##########################################################################


def fn_compare_feat_imp(f_ratios, feat_imp, df_tr):
    fig, axes = plt.subplots(1, 2, figsize=(10, 6), sharey=True)
    axes = axes.flatten()

    f = f_ratios.sort_values()

    feat_imp = abs(feat_imp.flatten())
    s = pd.Series(feat_imp, index=df_tr.columns[:-1])
    s = s[f.index]

    axes[0].barh(f.index, f.values, alpha=0.7)
    axes[0].set_xlabel('CORR_FEAT_IMP')

    axes[1].barh(s.index, s.values, alpha=0.7)
    axes[1].set_xlabel('MODEL_FEAT_IMP')
    plt.tight_layout()


##########################################################################
##########################################################################


# TESTING:

def fn_test_model_binary_clf(df_Xy_, model_, threshold_class_1=0.5):
    df, model = df_Xy_, model_
    X, y = df.iloc[:, :-1].values, df.iloc[:, -1].values.ravel()
    y_proba = fn_pred_proba(model, X)

    logloss = log_loss(y, y_proba, labels=model_.classes_)
    acc, prec_0, prec_1, rec_0, rec_1 = fn_acc_prec_rec(y, y_proba, threshold_class_1)

    df = pd.DataFrame().assign(prec=(prec_0, prec_1), rec=(rec_0, rec_1))
    df.index = ['class_' + str(i) for i in range(len(df))]

    print('--------------------')
    print(f'LOGLOSS : {round(logloss, 4)}')
    print(f'ACCURACY: {round(acc, 3)}')
    print('--------------------')
    print()

    return df.round(3)


##############################################################################
##############################################################################


def fn_pr_rec_tr_eval_multi(y_tr, y_tr_pred, y_eval, y_eval_pred):
    def fn_prec_rec_multicls(y, y_pred):
        dff = pd.DataFrame().assign(y=y, y_pred=y_pred)
        classes = dff.y.unique()
        d = {}
        collect_TPs = []

        for cls in classes:
            preds_current_class = dff[dff.y == cls].y_pred
            preds_other_classes = dff[dff.y != cls].y_pred

            TP = sum([1 for i in preds_current_class if i == cls]) + 1e-6
            FP = sum([1 for i in preds_current_class if i != cls]) + 1e-6
            FN = sum([1 for i in preds_other_classes if i == cls]) + 1e-6
            prec, rec = TP / (TP + FP), TP / (TP + FN)
            collect_TPs.append(TP)

            d[cls] = [prec, rec]

        dff = pd.DataFrame(d).T
        dff.columns = 'prec rec'.split()

        acc = 100 * np.array(collect_TPs).sum() / len(y)
        return dff, acc.round(4)

    dff_tr, tr_acc = fn_prec_rec_multicls(y_tr, y_tr_pred)
    dff_eval, eval_acc = fn_prec_rec_multicls(y_eval, y_eval_pred)

    dff = pd.concat([dff_tr, dff_eval], axis=1) * 100
    dff.columns = 'tr_prec tr_rec eval_prec eval_rec'.split()

    prec_diff = (dff.tr_prec - dff.eval_prec).round(4)
    rec_diff = (dff.tr_rec - dff.eval_rec).round(4)
    dff = dff.assign(prec_diff=prec_diff, rec_diff=rec_diff)
    avg = pd.DataFrame(dff.mean(axis=0), columns=['avg']).T
    df_performance = pd.concat([dff, avg])
    print(f'tr_acc = {tr_acc}, eval_acc = {eval_acc}')
    return df_performance.style.background_gradient()


def fn_pr_rec_tr_ts_multi(y_tr, y_tr_pred, y_ts, y_ts_pred):
    def fn_prec_rec_multicls(y, y_pred):
        dff = pd.DataFrame().assign(y=y, y_pred=y_pred)
        classes = dff.y.unique()
        d = {}
        collect_TPs = []

        for cls in classes:
            preds_current_class = dff[dff.y == cls].y_pred
            preds_other_classes = dff[dff.y != cls].y_pred

            TP = sum([1 for i in preds_current_class if i == cls]) + 1e-6
            FP = sum([1 for i in preds_current_class if i != cls]) + 1e-6
            FN = sum([1 for i in preds_other_classes if i == cls]) + 1e-6
            prec, rec = TP / (TP + FP), TP / (TP + FN)
            collect_TPs.append(TP)

            d[cls] = [prec, rec]

        dff = pd.DataFrame(d).T
        dff.columns = 'prec rec'.split()

        acc = 100 * np.array(collect_TPs).sum() / len(y)
        return dff, acc.round(4)

    dff_tr, tr_acc = fn_prec_rec_multicls(y_tr, y_tr_pred)
    dff_eval, eval_acc = fn_prec_rec_multicls(y_ts, y_ts_pred)

    dff = pd.concat([dff_tr, dff_eval], axis=1) * 100
    dff.columns = 'tr_prec tr_rec ts_prec ts_rec'.split()

    prec_diff = (dff.tr_prec - dff.ts_prec).round(4)
    rec_diff = (dff.tr_rec - dff.ts_rec).round(4)
    dff = dff.assign(prec_diff=prec_diff, rec_diff=rec_diff)
    avg = pd.DataFrame(dff.mean(axis=0), columns=['avg']).T
    df_performance = pd.concat([dff, avg])
    print(f'tr_acc = {tr_acc}, ts_acc = {eval_acc}')
    return df_performance.style.background_gradient()