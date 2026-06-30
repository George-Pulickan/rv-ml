from pathlib import Path
import pandas as pd 
from preprocess import RVDataset
from parse_and_label import parse_tbl 
from time_series_features import spectral_features 
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def create_dataset(split="train"):
    dataset = RVDataset(split, normalize=False)
    rows = []

    for i in range(len(dataset)):
        x, lsp, theta, info = dataset.get_numpy(i)

        if not info["valid"]:
            continue

        path = Path("data/rv_raw")/info["file"]
        meta, t, rv, rv_err = parse_tbl(path)

        features = spectral_features(t, rv)
        row = {}

        i = 1
        for feature in features: 
            row[f"f{i}"] = feature
            i += 1

        row["log10_P"] = theta[0]
        row["log10_K"] = theta[1]
        row["e"] = theta[2]
        row["cos_omega"] = theta[3]
        row["sin_omega"] = theta[4]
        rows.append(row)

    frame = pd.DataFrame(rows)
    return frame

# test create dataset
train = create_dataset("train")
print(train.head(10))
print(train.shape)

# train 
feature_columns = []
for column in train.columns:
    if column.startswith("f"):
        feature_columns.append(column)

X = train[feature_columns]
y = train["log10_P"]

rf = RandomForestRegressor()
rf.fit(X, y)

# test
test = create_dataset("test")
X_test = test[feature_columns]
y_test = test["log10_P"]

# predict
y_pred = rf.predict(X_test)

# metrics 
print("MEAN ABSOLUTE ERROR:", mean_absolute_error(y_test, y_pred))
print("MEAN SQUARED ERROR:", mean_squared_error(y_test, y_pred))
print("R2 SCORE:", r2_score(y_test, y_pred)) 