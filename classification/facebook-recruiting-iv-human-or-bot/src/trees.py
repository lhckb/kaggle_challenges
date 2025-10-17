import marimo

__generated_with = "0.13.11"
app = marimo.App(width="full")


@app.cell
def _(mo):
    mo.md(r"""# Tree-based Models""")
    return


@app.cell
def _():
    # %load_ext autoreload
    # %reload_ext autoreload
    # %autoreload 2

    # system
    import os
    from pathlib import Path
    import sys
    sys.path.append(str(Path().resolve().parents[2]))  # workaround to resolve my helper module above

    # dataframe
    import polars as pl
    import polars.selectors as cs

    # modules
    from modules.plotting import multiple_barplot, multiple_boxplots, compute_and_plot_pr_roc_auc
    from modules.polars_helpers import pivot_table_on_column

    # math
    from scipy.stats import pearsonr
    import numpy as np

    # machine learning
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.compose import ColumnTransformer
    # from sklearn.pipeline import Pipeline
    from sklearn.model_selection import train_test_split, KFold, cross_validate, cross_val_predict
    from sklearn.metrics import precision_recall_curve, roc_curve, auc
    from xgboost import XGBClassifier, XGBRFClassifier
    from sklearn.svm import SVC
    from skopt import BayesSearchCV
    from skopt.space import Real, Integer
    from sklearn.linear_model import LogisticRegression
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline

    parent_dir = Path.cwd().parent

    DATA_PATH = str(parent_dir / "data") + "/"
    print(DATA_PATH)

    RANDOM_SEED = 42
    return (
        BayesSearchCV,
        ColumnTransformer,
        DATA_PATH,
        Integer,
        KFold,
        LabelEncoder,
        LogisticRegression,
        Pipeline,
        RANDOM_SEED,
        RandomForestClassifier,
        Real,
        SMOTE,
        SVC,
        StandardScaler,
        XGBClassifier,
        XGBRFClassifier,
        compute_and_plot_pr_roc_auc,
        cross_val_predict,
        cs,
        np,
        pl,
    )


@app.cell
def _(DATA_PATH, pl):
    train = pl.read_csv(DATA_PATH + "train.csv")
    test = pl.read_csv(DATA_PATH + "test.csv")

    bids = pl.read_csv(DATA_PATH + "bids.csv")
    return bids, test, train


@app.cell
def _(mo):
    mo.md(r"""The goal of this notebook is to model the data using tree based models such as Random Forests, XGBoost, perhaps Catboost too.""")
    return


@app.cell
def _(mo):
    mo.md(r"""Considering all models here will be tree-based and use an sklearn-like API, we can define pipelines to label encode data. We can label encode data considering the magnitude or ordinal importance of categories is of very low value for the model, because it will just split on the numbers representing categories""")
    return


@app.cell
def _(cs, pl):
    def create_features(bidders_df: pl.DataFrame, bids: pl.DataFrame) -> pl.DataFrame:
        bids = bids.drop_nulls()

        # devices per bidder
        n_devices_per_bidder = bids.group_by("bidder_id").n_unique().select(["bidder_id", "device"])
        bidders_df = bidders_df.join(n_devices_per_bidder, on="bidder_id", how="left")

        # auction count per bidder
        auction_count_per_bidder = bids.group_by("bidder_id").n_unique().select(["bidder_id", "auction"])
        bidders_df = bidders_df.join(auction_count_per_bidder, on="bidder_id", how="left")

        # merchandise per bidder
        max_kind_of_merch_per_bidder = bids.group_by("bidder_id").max().select(["bidder_id", "merchandise"])
        bidders_df = bidders_df.join(max_kind_of_merch_per_bidder, on="bidder_id", how="left")

        # number of ips per bidder
        ip_count_per_bidder = bids.group_by("bidder_id").n_unique().select(["bidder_id", "ip"])
        bidders_df = bidders_df.join(ip_count_per_bidder, on="bidder_id", how="left")

        # different URLs per bidder
        url_count_per_bidder = bids.group_by("bidder_id").n_unique().select(["bidder_id", "url"])
        bidders_df = bidders_df.join(url_count_per_bidder, on="bidder_id", how="left")

        # most occurring country per bidder
        country_counts = (
            bids
            .group_by(["bidder_id", "country"])
            .agg(pl.count().alias("country_count"))
        )

        max_country_per_bidder = (
            country_counts
            .sort(["bidder_id", "country_count"], descending=[False, True])
            .group_by("bidder_id")
            .agg([
                pl.first("country").alias("most_frequent_country")
            ])
        )

        bidders_df = bidders_df.join(max_country_per_bidder, on="bidder_id", how="left")

        # time delta between bids
        bids = bids.sort(["bidder_id", "time"], descending=False).with_columns(
            pl.col("time").diff().over("bidder_id").fill_null(0).alias("diff_last_bid")
        )

        mean_time_diff_bids = bids.group_by("bidder_id").agg(
            pl.col("diff_last_bid").mean().alias("mean_diff_time_bids")
        )

        bidders_df = bidders_df.join(mean_time_diff_bids, on="bidder_id", how="left")

        # num of bids per bidder
        bid_counts = bids.group_by("bidder_id").agg(
            pl.n_unique("bid_id").alias("bid_count")
        )

        bidders_df = bidders_df.join(bid_counts, on="bidder_id", how="left")

        # handling missing values after feature creation
        numerical_cols = bidders_df.select(cs.by_dtype(pl.NUMERIC_DTYPES)).columns
        numerical_cols = filter(lambda x: x != "outcome", numerical_cols)

        bidders_df = bidders_df.with_columns([
            pl.col("most_frequent_country").fill_null("unknown"),
            pl.col("merchandise").fill_null("unknown"),
            pl.col("bidder_id").is_in(bids["bidder_id"]).alias("has_bids"),
        ]).with_columns(
            pl.col(num_col).fill_null(0) for num_col in numerical_cols
        )

        return bidders_df
    return (create_features,)


@app.cell
def _(bids, create_features, test, train):
    train_1 = create_features(train, bids)
    test_processed = create_features(test, bids)
    return test_processed, train_1


@app.cell
def _(train_1):
    train_1.schema
    return


@app.cell
def _(train_1):
    train_1.head(3)
    return


@app.cell
def _():
    numeric_features = ["mean_diff_time_bids", "bid_count", "auction", "device", "ip", "url"]
    cat_features = [
        "has_bids",
        "merchandise",
        "most_frequent_country",
    ]
    return cat_features, numeric_features


@app.cell
def _(
    ColumnTransformer,
    LabelEncoder,
    StandardScaler,
    cat_features,
    numeric_features,
):
    col_transformer = ColumnTransformer(
        transformers=[
            ("cat", LabelEncoder(), cat_features),
            ("num", StandardScaler(), numeric_features)
    ])
    return


@app.cell
def _(KFold):
    kfold = KFold(n_splits=5)
    return (kfold,)


@app.cell
def _(train_1):
    X = train_1.drop(['payment_account', 'address', 'bidder_id', 'outcome'])
    y = train_1['outcome']
    return X, y


@app.cell
def _(LabelEncoder, X, pl):
    X_label_enc = X.with_columns(
        pl.lit(LabelEncoder().fit_transform(X["has_bids"])).alias("has_bids"),
        pl.lit(LabelEncoder().fit_transform(X["merchandise"])).alias("merchandise"),
        pl.lit(LabelEncoder().fit_transform(X["most_frequent_country"])).alias("most_frequent_country"),
    )

    X_label_enc
    return (X_label_enc,)


@app.cell
def _(mo):
    mo.md(r"""## Random Forest Classifier""")
    return


@app.cell
def _(mo):
    mo.md(r"""### Baseline""")
    return


@app.cell
def _(RANDOM_SEED, RandomForestClassifier):
    rf = RandomForestClassifier(random_state=RANDOM_SEED)
    return (rf,)


@app.cell
def _(X_label_enc, cross_val_predict, kfold, rf, y):
    y_prob_rf = cross_val_predict(rf, X_label_enc, y, cv=kfold, n_jobs=-1, method="predict_proba")
    y_scores = y_prob_rf[:, 1]
    return (y_scores,)


@app.cell
def _(compute_and_plot_pr_roc_auc, y, y_scores):
    compute_and_plot_pr_roc_auc(y, y_scores)
    return


@app.cell
def _(mo):
    mo.md(r"""## XGBoost Classifier""")
    return


@app.cell
def _(np, y):
    scale_pos_weight = np.sqrt((y == 0).sum() / (y == 1).sum())
    scale_pos_weight
    return (scale_pos_weight,)


@app.cell
def _(mo):
    mo.md(r"""### Baseline""")
    return


@app.cell
def _(RANDOM_SEED, XGBClassifier, scale_pos_weight):
    xgb_clf = XGBClassifier(scale_pos_weight = scale_pos_weight, random_state=RANDOM_SEED)
    return (xgb_clf,)


@app.cell
def _(X_label_enc, cross_val_predict, kfold, xgb_clf, y):
    y_prob_xgb = cross_val_predict(xgb_clf, X_label_enc, y, cv=kfold, n_jobs=-1, method="predict_proba")
    y_scores_xgb = y_prob_xgb[:, 1]
    return (y_scores_xgb,)


@app.cell
def _(compute_and_plot_pr_roc_auc, y, y_scores_xgb):
    compute_and_plot_pr_roc_auc(y, y_scores_xgb)
    return


@app.cell
def _(mo):
    mo.md(r"""## XGBoost Random Forest""")
    return


@app.cell
def _(mo):
    mo.md(r"""### Baseline""")
    return


@app.cell
def _(RANDOM_SEED, XGBRFClassifier, scale_pos_weight):
    xgbrf_clf = XGBRFClassifier(objective="binary:logistic", scale_pos_weight = scale_pos_weight, random_state=RANDOM_SEED)
    return (xgbrf_clf,)


@app.cell
def _(X_label_enc, cross_val_predict, kfold, xgbrf_clf, y):
    y_prob_xgbrf = cross_val_predict(xgbrf_clf, X_label_enc, y, cv=kfold, n_jobs=-1, method="predict_proba")
    y_scores_xgbrf = y_prob_xgbrf[:, 1]
    return (y_scores_xgbrf,)


@app.cell
def _(compute_and_plot_pr_roc_auc, y, y_scores_xgbrf):
    compute_and_plot_pr_roc_auc(y, y_scores_xgbrf)
    return


@app.cell
def _(mo):
    mo.md(r"""XGBoost Random Forest improved slightly on sklearn's random forest, but the precision-recall curve is still very much unstable. That is most likely a dataset issue and not something the model can compensate for by itself""")
    return


@app.cell
def _(mo):
    mo.md(r"""### Hyperparameter Search""")
    return


@app.cell
def _(Integer, Real):
    search_space = {
        "n_estimators": Integer(5, 500),
        "max_depth": Integer(2, 20),
        "learning_rate": Real(0.0001, 0.2, prior="log-uniform"),
        "subsample": Real(0.5, 1.0),
        "colsample_bynode": Real(0.4, 1.0)
    }
    return (search_space,)


@app.cell
def _(
    BayesSearchCV,
    RANDOM_SEED,
    XGBRFClassifier,
    X_label_enc,
    cross_val_predict,
    kfold,
    scale_pos_weight,
    search_space,
    y,
):
    xgbrf = XGBRFClassifier(
        n_jobs=-1,
        random_state=RANDOM_SEED,
        scale_pos_weight=scale_pos_weight
    )

    search = BayesSearchCV(
        estimator=xgbrf,
        search_spaces=search_space,
        n_iter=100,
        scoring="roc_auc",
        cv=kfold,
        verbose=0,
        n_jobs=-1,
        random_state=RANDOM_SEED
    )

    search.fit(X_label_enc, y)

    xgbrf_best = search.best_estimator_
    print("Best parameters:", search.best_params_)

    y_prob_xgbrf_bcv = cross_val_predict(
        xgbrf_best, X_label_enc, y, cv=kfold, n_jobs=-1, method="predict_proba"
    )
    y_scores_xgbrf_bcv = y_prob_xgbrf_bcv[:, 1]
    return (y_scores_xgbrf_bcv,)


@app.cell
def _(compute_and_plot_pr_roc_auc, y, y_scores_xgbrf_bcv):
    compute_and_plot_pr_roc_auc(y, y_scores_xgbrf_bcv)
    return


@app.cell
def _(mo):
    mo.md(r"""### Oversampled Minority Class""")
    return


@app.cell
def _(RANDOM_SEED, SMOTE, X_label_enc, y):
    X_resampled, y_resampled = SMOTE(random_state=RANDOM_SEED).fit_resample(X_label_enc.to_numpy(), y.to_numpy())
    X_resampled.shape, (y_resampled == 1).sum()
    return X_resampled, y_resampled


@app.cell
def _(
    RANDOM_SEED,
    XGBRFClassifier,
    X_resampled,
    cross_val_predict,
    kfold,
    y_resampled,
):
    xgbrf_clf_os = XGBRFClassifier(
        n_jobs=-1,
        random_state=RANDOM_SEED
    )

    y_prob_xgbrf_os = cross_val_predict(xgbrf_clf_os, X_resampled, y_resampled, cv=kfold, n_jobs=-1, method="predict_proba")
    y_scores_xgbrf_os = y_prob_xgbrf_os[:, 1]
    return xgbrf_clf_os, y_scores_xgbrf_os


@app.cell
def _(compute_and_plot_pr_roc_auc, y_resampled, y_scores_xgbrf_os):
    compute_and_plot_pr_roc_auc(y_resampled, y_scores_xgbrf_os)
    return


@app.cell
def _(mo):
    mo.md(r"""### Oversampling Minority + Bayesian Search""")
    return


@app.cell
def _(Integer, Real):
    search_space_pipeline = {
        "clf__n_estimators": Integer(5, 500),
        "clf__max_depth": Integer(2, 20),
        "clf__learning_rate": Real(0.0001, 0.2, prior="log-uniform"),
        "clf__subsample": Real(0.5, 1.0),
        "clf__colsample_bynode": Real(0.4, 1.0)
    }
    return (search_space_pipeline,)


@app.cell
def _(
    BayesSearchCV,
    Pipeline,
    RANDOM_SEED,
    SMOTE,
    XGBRFClassifier,
    X_label_enc,
    kfold,
    search_space_pipeline,
    y,
):
    xgbrf_os_pipe = Pipeline([
        ('smote', SMOTE()),
        ('clf', XGBRFClassifier(random_state=RANDOM_SEED, n_jobs=-1))
    ])

    xgbrf_os_search = BayesSearchCV(
        xgbrf_os_pipe, 
        search_space_pipeline, 
        scoring='roc_auc', 
        cv=kfold, 
        random_state=RANDOM_SEED,
        n_jobs=-1, 
        verbose=0, 
        n_iter=50
    )

    xgbrf_os_search.fit(X_label_enc.to_numpy(), y.to_numpy())
    return (xgbrf_os_search,)


@app.cell
def _(xgbrf_os_search):
    xgbrf_os_search.best_params_
    return


@app.cell
def _(
    X_label_enc,
    compute_and_plot_pr_roc_auc,
    cross_val_predict,
    kfold,
    xgbrf_os_search,
    y,
):
    xgbrf_os_best = xgbrf_os_search.best_estimator_.named_steps["clf"]

    y_prob_xgbrf_os_bcv = cross_val_predict(
        xgbrf_os_best, X_label_enc, y, cv=kfold, n_jobs=-1, method="predict_proba"
    )
    y_scores_xgbrf_os_bcv = y_prob_xgbrf_os_bcv[:, 1]

    compute_and_plot_pr_roc_auc(y, y_scores_xgbrf_os_bcv)
    return (xgbrf_os_best,)


@app.cell
def _(mo):
    mo.md(r"""# Linear Models""")
    return


@app.cell
def _(X, cat_features, pl):
    onehots = X.select(cat_features).to_dummies()

    X_onehot_enc = pl.concat([X.drop(cat_features), onehots], how="horizontal")
    X_onehot_enc.schema
    return (X_onehot_enc,)


@app.cell
def _(StandardScaler, X_onehot_enc, numeric_features, pl):
    scaler = StandardScaler()
    num_feats = X_onehot_enc.select(numeric_features)
    scaled_feats = scaler.fit_transform(num_feats)
    scaled_feats_df = pl.DataFrame(scaled_feats, schema=numeric_features)
    X_onehot_enc_1 = X_onehot_enc.with_columns([scaled_feats_df[col].alias(col) for col in numeric_features])
    X_onehot_enc_1
    return (X_onehot_enc_1,)


@app.cell
def _(mo):
    mo.md(r"""## Support Vector Machines""")
    return


@app.cell
def _(mo):
    mo.md(r"""### Baseline""")
    return


@app.cell
def _(RANDOM_SEED, SVC):
    svm_clf = SVC(random_state=RANDOM_SEED, class_weight="balanced", kernel="rbf", C=1)
    return (svm_clf,)


@app.cell
def _(X_onehot_enc_1, cross_val_predict, kfold, svm_clf, y):
    y_prob_svm = cross_val_predict(svm_clf, X_onehot_enc_1, y, cv=kfold, n_jobs=-1, method='decision_function')
    y_scores_svm = y_prob_svm
    return (y_scores_svm,)


@app.cell
def _(compute_and_plot_pr_roc_auc, y, y_scores_svm):
    compute_and_plot_pr_roc_auc(y, y_scores_svm)
    return


@app.cell
def _(mo):
    mo.md(r"""The baseline SVM seems less stable than a baseline Random Forest for this dataset""")
    return


@app.cell
def _(mo):
    mo.md(r"""## Logistic Regression""")
    return


@app.cell
def _(mo):
    mo.md(r"""### Baseline""")
    return


@app.cell
def _(LogisticRegression, RANDOM_SEED):
    lr_clf = LogisticRegression(random_state=RANDOM_SEED, class_weight="balanced")
    return (lr_clf,)


@app.cell
def _(X_onehot_enc_1, cross_val_predict, kfold, lr_clf, y):
    y_prob_lr = cross_val_predict(lr_clf, X_onehot_enc_1, y, cv=kfold, n_jobs=-1, method='predict_proba')
    y_scores_lr = y_prob_lr[:, 1]
    return (y_scores_lr,)


@app.cell
def _(compute_and_plot_pr_roc_auc, y, y_scores_lr):
    compute_and_plot_pr_roc_auc(y, y_scores_lr)
    return


@app.cell
def _(mo):
    mo.md(r"""# Submissions""")
    return


@app.cell
def _(mo):
    mo.md(r"""## XGBRFClassifier - undefeated best""")
    return


@app.cell
def _(LabelEncoder, pl, test_processed):
    test_processed_1 = test_processed.drop(['bidder_id', 'payment_account', 'address'])
    test_processed_1 = test_processed_1.with_columns(pl.lit(LabelEncoder().fit_transform(test_processed_1['has_bids'])).alias('has_bids'), pl.lit(LabelEncoder().fit_transform(test_processed_1['merchandise'])).alias('merchandise'), pl.lit(LabelEncoder().fit_transform(test_processed_1['most_frequent_country'])).alias('most_frequent_country'))
    test_processed_1
    return (test_processed_1,)


@app.cell
def _(DATA_PATH, X_label_enc, pl, test, test_processed_1, xgbrf_clf, y):
    xgbrf_clf.fit(X_label_enc, y)
    _y_probas = xgbrf_clf.predict_proba(test_processed_1)
    pl.DataFrame([_y_probas[:, 1], test['bidder_id']], schema=['prediction', 'bidder_id']).write_csv(DATA_PATH + 'xgbrf.csv')
    return


@app.cell
def _(mo):
    mo.md(r"""## XGBRFClassifier with minority class upsampled""")
    return


@app.cell
def _(
    DATA_PATH,
    X_resampled,
    pl,
    test,
    test_processed_1,
    xgbrf_clf_os,
    y_resampled,
):
    xgbrf_clf_os_1 = xgbrf_clf_os.fit(X_resampled, y_resampled)
    _y_probas = xgbrf_clf_os_1.predict_proba(test_processed_1)
    pl.DataFrame([_y_probas[:, 1], test['bidder_id']], schema=['prediction', 'bidder_id']).write_csv(DATA_PATH + 'xgbrf_upsampled.csv')
    return


@app.cell
def _(mo):
    mo.md(r"""## XGBRFClassifier + minority oversampled + hyperparameter search""")
    return


@app.cell
def _(
    DATA_PATH,
    X_resampled,
    pl,
    test,
    test_processed_1,
    xgbrf_os_best,
    y_resampled,
):
    xgbrf_os_best.fit(X_resampled, y_resampled)
    _y_probas = xgbrf_os_best.predict_proba(test_processed_1)
    pl.DataFrame([_y_probas[:, 1], test['bidder_id']], schema=['prediction', 'bidder_id']).write_csv(DATA_PATH + 'xgbrf_upsampled_tuned.csv')
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
