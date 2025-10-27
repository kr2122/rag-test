from sklearn.ensemble import RandomForestRegressor

def train_random_forest(X_train, y_train, X_test, y_test):
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, y_pred
