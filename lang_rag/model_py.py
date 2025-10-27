import numpy as np
# from model.lstm_model import train_lstm
# from model.random_forest_model import train_random_forest
# from model.xgboost_model import train_xgboost
# from model.rnn_model import train_rnn
# from model.lightgbm_model import train_lightgbm
# from model.mlp_model import train_mlp
# 文件：lang_rag/train_interface.py
# from model.lstm_model import train_lstm
import importlib

def train_model(model_name: str, df):
    """
    根据模型名称调用对应模型文件进行训练。
    各模型文件应包含一个 train(df) 函数。
    """
    model_name = model_name.lower()
    try:
        if "lstm" in model_name or "LSTM" in model_name:
            module = importlib.import_module("model.lstm_model")
        elif "rnn" in model_name or "RNN" in model_name:
            module = importlib.import_module("model.rnn_model")
        # elif "cnn" or "CNN" in model_name:
        #     module = importlib.import_module("model.cnn_model")
        elif "xgboost" in model_name or "xgb" in model_name:
            module = importlib.import_module("model.xgboost_model")
        # elif "lightgbm" in model_name or "lgbm" in model_name:
        #     module = importlib.import_module("model.lightgbm_model")
        elif "randomforest" in model_name or "rf" in model_name:
            module = importlib.import_module("model.random_forest_model")
        else:
            raise ValueError(f"暂不支持的模型类型: {model_name}")

        print(f"🚀 开始训练模型：{model_name}")
        return module.train(df)

    except Exception as e:
        print(f"❌ 模型加载或训练失败: {e}")
        return None, None













# def train_model(model_name, X_train, y_train, X_test, y_test, **kwargs):
#     model_name = model_name.lower()
#     if model_name == "lstm" or model_name == "LSTM":
#         return train_lstm(X_train, y_train, X_test, y_test, input_size=X_train.shape[2], **kwargs)
#     elif model_name == "random_forest":
#         return train_random_forest(X_train, y_train, X_test, y_test)
#     elif model_name == "xgboost":
#         return train_xgboost(X_train, y_train, X_test, y_test)
#     elif model_name == "rnn" or model_name == "RNN":
#         return train_rnn(X_train, y_train, X_test, y_test, input_size=X_train.shape[2], **kwargs)
#     # elif model_name == "lightgbm":
#     #     return train_lightgbm(X_train, y_train, X_test, y_test)
#     # elif model_name == "mlp":
#     #     return train_mlp(X_train, y_train, X_test, y_test)
#     else:
#         raise ValueError(f"❌ 未知模型类型: {model_name}")


