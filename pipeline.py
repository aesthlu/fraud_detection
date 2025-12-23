from src.ingestion import load_data
from src.preprocessing import preprocess
from src.features import build_features
from src.train import temporal_split, train_xgb
from src.evaluate import evaluate_model, business_threshold

df = load_data("data/creditcard.csv")
df, scaler = preprocess(df)
df = build_features(df)

train, test = temporal_split(df)

X_train = train.drop("Class", axis=1)
y_train = train["Class"]
X_test  = test.drop("Class", axis=1)
y_test  = test["Class"]

model = train_xgb(X_train, y_train)
roc, pr, probs = evaluate_model(model, X_test, y_test)

threshold = business_threshold(y_test, probs)

print("ROC-AUC:", roc)
print("PR-AUC:", pr)
print("Optimal threshold:", threshold)
#

import joblib

if __name__ == "__main__":
    df = load_data("data/creditcard.csv")
    df, scaler = preprocess(df)
    df = build_features(df)
    FEATURES = df.drop("Class", axis=1).columns.tolist()
    train, test = temporal_split(df)

    model = train_xgb(train[FEATURES], train["Class"])

    joblib.dump(model, "models/fraud_model.pkl")
    joblib.dump(scaler, "models/amount_scaler.pkl")
    joblib.dump(FEATURES, "models/features.pkl")
    joblib.dump(threshold, "models/optimal_threshold.pkl")