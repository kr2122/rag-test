# 文件路径：lang_rag/models/rnn_model.py
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import numpy as np
# from sklearn.metrics import mean_squared_error, r2_score

# # ====================== #
# #       模型定义
# # ====================== #
# class RNNModel(nn.Module):
#     def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=1, dropout=0.2):
#         super(RNNModel, self).__init__()
#         self.rnn = nn.RNN(
#             input_size=input_size,
#             hidden_size=hidden_size,
#             num_layers=num_layers,
#             batch_first=True,
#             dropout=dropout,
#             nonlinearity='tanh'
#         )
#         self.fc = nn.Linear(hidden_size, output_size)

#     def forward(self, x):
#         # x: [batch, seq_len, input_size]
#         out, _ = self.rnn(x)
#         out = self.fc(out[:, -1, :])  # 取最后一个时间步输出
#         return out


# # ====================== #
# #       训练函数
# # ====================== #
# def train_rnn(X_train, y_train, X_test, y_test, input_size, epochs=50, lr=0.001):
#     """
#     训练 RNN 模型
#     参数:
#         X_train, y_train, X_test, y_test: numpy 数据
#         input_size: 特征数量
#         epochs: 训练轮数
#         lr: 学习率
#     """
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # 数据格式调整
#     X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
#     y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
#     X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
#     y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1).to(device)

#     # 构建模型
#     model = RNNModel(input_size=input_size).to(device)
#     criterion = nn.MSELoss()
#     optimizer = optim.Adam(model.parameters(), lr=lr)

#     # ====================== #
#     #       开始训练
#     # ====================== #
#     print(f"🚀 开始训练 RNN 模型，共 {epochs} 轮")

#     for epoch in range(epochs):
#         model.train()
#         optimizer.zero_grad()
#         outputs = model(X_train)
#         loss = criterion(outputs, y_train)
#         loss.backward()
#         optimizer.step()

#         if (epoch + 1) % 10 == 0:
#             print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.6f}")

#     # ====================== #
#     #       模型评估
#     # ====================== #
#     model.eval()
#     with torch.no_grad():
#         y_pred = model(X_test).cpu().numpy().flatten()
#         y_true = y_test.cpu().numpy().flatten()
#         mse = mean_squared_error(y_true, y_pred)
#         r2 = r2_score(y_true, y_pred)

#     print("✅ RNN 训练完成")
#     print(f"📉 MSE: {mse:.6f}, R²: {r2:.4f}")

#     return model, y_pred


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


# ====================== #
#       模型定义
# ====================== #
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=1, dropout=0.2):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            nonlinearity='tanh'
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])  # 取最后一个时间步输出
        return out


# ====================== #
#       训练函数
# ====================== #
def train(df: pd.DataFrame, target_col: str = None, epochs=50, lr=0.001):
    """
    训练 RNN 模型（直接输入 DataFrame）

    参数:
        df: pandas DataFrame（包含特征和目标）
        target_col: 目标列名（若不指定，则默认最后一列）
        epochs: 训练轮数
        lr: 学习率
    """

    print("📊 [RNN] 开始数据预处理...")

    # 确定目标列
    if target_col is None:
        target_col = "air_temperature"
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                # 尝试转换日期时间列为时间戳
                df[col] = pd.to_datetime(df[col]).astype('int64') // 10**9
            except:
                # 其他字符串列用均值填充（或根据实际情况处理）
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col].fillna(df[col].mean(), inplace=True)

    X = df.drop(columns=[target_col]).values
    y = df[target_col].values

    # 数据归一化（建议对 RNN 使用）
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

    # 拆分训练集与测试集
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

    # RNN 输入要求: [samples, timesteps, features]
    X_train = np.expand_dims(X_train, axis=1)
    X_test = np.expand_dims(X_test, axis=1)

    input_size = X_train.shape[2]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 转换为 tensor
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1).to(device)

    # 构建模型
    model = RNNModel(input_size=input_size).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print(f"🚀 开始训练 RNN 模型，共 {epochs} 轮")

    # ====================== #
    #       开始训练
    # ====================== #
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.6f}")

    # ====================== #
    #       模型评估
    # ====================== #
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test).cpu().numpy().flatten()
        y_true = y_test.cpu().numpy().flatten()

        # 反归一化
        y_pred_inv = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()
        y_true_inv = scaler_y.inverse_transform(y_true.reshape(-1, 1)).flatten()

        mse = mean_squared_error(y_true_inv, y_pred_inv)
        r2 = r2_score(y_true_inv, y_pred_inv)

    print("✅ RNN 训练完成")
    print(f"📉 MSE: {mse:.6f}, R²: {r2:.4f}")

    return  y_pred_inv, {"mse": mse, "r2": r2}
# if __name__ == "__main__":
#     # 读取数据（替换为你的数据路径）
#     data_path = "./lang_rag/data/environment_data_export_2025-10-16_164127.csv"  # 你自己的Excel文件路径
#     df = pd.read_csv(data_path)
#     print(train(df))

