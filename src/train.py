
def temporal_split(df, split_ratio=0.7):
    split_time = df["Time"].quantile(split_ratio)

    train = df[df["Time"] <= split_time]
    test  = df[df["Time"] > split_time]

    return train, test

from sklearn.linear_model import LogisticRegression

def train_logistic(X, y):
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000
    )
    model.fit(X, y)
    return model

from xgboost import XGBClassifier
import mlflow
import mlflow.sklearn

def train_xgb(X, y):
    scale_pos_weight = (y == 0).sum() / (y == 1).sum()

    with mlflow.start_run():
        model = XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric="aucpr"
        )

        model.fit(X, y)

        mlflow.log_param("n_estimators", 200)
        mlflow.log_param("max_depth", 5)
        mlflow.log_param("learning_rate", 0.05)
        mlflow.log_param("scale_pos_weight", scale_pos_weight)

        mlflow.sklearn.log_model(model, "model")

    return model

# mlflow ui