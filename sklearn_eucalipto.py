# %%

from xgboost import XGBRegressor
from sklearn import metrics
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle

# %%

training = pd.read_excel('data/train.xlsx', skiprows=1)
testing = pd.read_excel('data/test.xlsx', skiprows=1)

df = training.append(testing)

X = df.filter(items=["DBH", "H", "TX", "d"]).values.reshape(-1, 4)
y = df.filter(items=["V"])

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.25, 
                                                    random_state=1504,
                                                    stratify=df.Cod_Volume)

model = XGBRegressor(
    objective='reg:squarederror',
    n_estimators=2000,
    max_depth=7,
    learning_rate=0.0367,
    n_jobs=10,
    gamma=0.0000806,
    booster="gbtree",
    min_child_weight=20
    )


model.fit(X_train, y_train)

predicted_vol = model.predict(X_test)

print("Mean Absolute Error:", metrics.r2_score(y_test, predicted_vol))

model_results = pd.DataFrame(
    {"actual": y_test["V"].array, "predicted": predicted_vol}
)


model_results["residual_abs"] = model_results["actual"] - \
    model_results["predicted"]

# %%
plt.figure(figsize=(8, 7), dpi=300)
plt.scatter(model_results["actual"], 
            model_results["predicted"],
            s=15, 
            edgecolors='black', 
            linewidth=0.4, 
            alpha=0.6)
plt.title("Volumetria Eucalipto - Modelo XGBoost")
plt.xlabel("Volume observado (m³)")
plt.ylabel("Volume predito (m³)")
z = np.polyfit(model_results["actual"], 
               model_results["predicted"], 
               1)
p = np.poly1d(z)
plt.plot(model_results["predicted"], 
         p(model_results["predicted"]), 
         "r--", 
         color="black")

# %%
model.fit(X, y)

# %%
