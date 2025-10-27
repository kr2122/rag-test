from xgboost import XGBRegressor
import numpy as np
def train_xgboost(X_train, y_train, X_test, y_test):
    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, y_pred
# if __name__ == "__main__":
#     # 示例数据
#     X_train = np.random.rand(100, 10, 5)
#     y_train = np.random.rand(100)
#     X_test = np.random.rand(20, 10, 5)
#     y_test = np.random.rand(20)
#     model, y_pred = train_xgboost(X_train, y_train, X_test, y_test)
#     print("✅ XGBoost模型训练完成！")
#     print(y_pred)
