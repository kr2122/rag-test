# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader, TensorDataset
# import numpy as np

# class LSTMModel(nn.Module):
#     def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=1, dropout=0.2):
#         super(LSTMModel, self).__init__()
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
#         self.fc = nn.Linear(hidden_size, output_size)

#     def forward(self, x):
#         out, _ = self.lstm(x)
#         out = self.fc(out[:, -1, :])  # 取最后一个时间步的输出
#         return out


# def train_lstm(X_train, y_train, X_test, y_test, input_size, epochs=50, lr=0.001, batch_size=32):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # 转换为Tensor
#     X_train = torch.tensor(X_train, dtype=torch.float32)
#     y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
#     X_test = torch.tensor(X_test, dtype=torch.float32)
#     y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

#     train_dataset = TensorDataset(X_train, y_train)
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

#     model = LSTMModel(input_size=input_size).to(device)
#     criterion = nn.MSELoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)

#     model.train()
#     for epoch in range(epochs):
#         for X_batch, y_batch in train_loader:
#             X_batch, y_batch = X_batch.to(device), y_batch.to(device)
#             optimizer.zero_grad()
#             output = model(X_batch)
#             loss = criterion(output, y_batch)
#             loss.backward()
#             optimizer.step()

#         if (epoch+1) % 10 == 0:
#             print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}")

#     model.eval()
#     with torch.no_grad():
#         y_pred = model(X_test.to(device)).cpu().numpy()
#         mse = np.mean((y_pred - y_test.numpy())**2)
#         print(f"✅ LSTM Test MSE: {mse:.6f}")

#     return model, y_pred






# # 文件路径：lang_rag/models/lstm_model.py
# import numpy as np
# import pandas as pd
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error, r2_score

# class LSTMModel(nn.Module):
#     def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=1):
#         super(LSTMModel, self).__init__()
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
#         self.fc = nn.Linear(hidden_size, output_size)

#     def forward(self, x):
#         out, _ = self.lstm(x)
#         return self.fc(out[:, -1, :])

# def train(df):
#     # 1️⃣ 数据准备
#     X = df.iloc[:, :-1].values
#     y = df.iloc[:, -1].values.reshape(-1, 1)

#     # 归一化
#     scaler_x, scaler_y = MinMaxScaler(), MinMaxScaler()
#     X = scaler_x.fit_transform(X)
#     y = scaler_y.fit_transform(y)

#     # 划分训练测试集
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     # 转为3D输入 [samples, timesteps, features]
#     X_train = np.expand_dims(X_train, axis=1)
#     X_test = np.expand_dims(X_test, axis=1)

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     model = LSTMModel(input_size=X_train.shape[2]).to(device)
#     criterion = nn.MSELoss()
#     optimizer = optim.Adam(model.parameters(), lr=0.001)

#     X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
#     y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
#     X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
#     y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

#     print("🚀 LSTM 开始训练")
#     for epoch in range(50):
#         model.train()
#         optimizer.zero_grad()
#         outputs = model(X_train)
#         loss = criterion(outputs, y_train)
#         loss.backward()
#         optimizer.step()

#         if (epoch + 1) % 10 == 0:
#             print(f"Epoch [{epoch+1}/50], Loss: {loss.item():.6f}")

#     # 测试集预测
#     model.eval()
#     with torch.no_grad():
#         y_pred = model(X_test).cpu().numpy()
#         y_pred = scaler_y.inverse_transform(y_pred)
#         y_true = scaler_y.inverse_transform(y_test.cpu().numpy())

#     mse = mean_squared_error(y_true, y_pred)
#     r2 = r2_score(y_true, y_pred)
#     print(f"✅ LSTM 完成 | MSE={mse:.6f}, R²={r2:.4f}")
#     return model, y_pred
# if __name__ == "__main__":
#     # 示例数据
#     df= pd.read_csv("./lang_rag/data/environment_data_export_2025-10-16_164127.csv")
#     print(train(df))
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=1, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0  # 仅多层时使用dropout
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])  # 取最后一个时间步输出


def create_sequences(X, y, seq_len=5):
    """将数据转换为序列格式 [samples, timesteps, features]"""
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_len):
        X_seq.append(X[i:i+seq_len])  # 连续seq_len个时间步作为输入
        y_seq.append(y[i+seq_len])    # 下一个时间步作为目标
    return np.array(X_seq), np.array(y_seq)


def train(df, seq_len=5, epochs=50, hidden_size=64):
    # 1️⃣ 数据预处理
    # 检查并处理非数值列（如日期时间）
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                # 尝试转换日期时间列为时间戳
                df[col] = pd.to_datetime(df[col]).astype('int64') // 10**9
            except:
                # 其他字符串列用均值填充（或根据实际情况处理）
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col].fillna(df[col].mean(), inplace=True)
    
    # 分离特征和目标
    X = df["air_temperature"].values.reshape(-1, 1)
    y = df["air_temperature"].values.reshape(-1, 1)

    # 处理缺失值
    X = np.nan_to_num(X, nan=np.nanmean(X))
    y = np.nan_to_num(y, nan=np.nanmean(y))

    # 归一化
    scaler_x, scaler_y = MinMaxScaler(), MinMaxScaler()
    X_scaled = scaler_x.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)

    # 创建序列数据（时间序列必须保持顺序，不打乱）
    X_seq, y_seq = create_sequences(X_scaled, y_scaled, seq_len)
    
    # 划分训练测试集（时间序列不打乱）
    split_idx = int(len(X_seq) * 0.8)
    X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
    y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]

    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device} | 序列长度: {seq_len} | 特征数: {X_train.shape[2]}")

    # 初始化模型、损失函数和优化器
    model = LSTMModel(
        input_size=X_train.shape[2],
        hidden_size=hidden_size,
        num_layers=2
    ).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)  # 添加L2正则化

    # 转换为张量
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

    # 训练过程
    print("🚀 LSTM 开始训练")
    train_losses = []
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        
        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.item())
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}")

    # 测试集评估
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test).cpu().numpy()
        # 反归一化
        y_pred = scaler_y.inverse_transform(y_pred)
        y_true = scaler_y.inverse_transform(y_test.cpu().numpy())

    # 计算指标
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"✅ LSTM 完成 | MSE={mse:.6f}, R²={r2:.4f}")

    # 可视化结果
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']  # 支持中文的字体列表
    plt.rcParams['axes.unicode_minus'] = False  
    plt.figure(figsize=(12, 6))
    plt.plot(y_true, label='真实值', color='blue')
    plt.plot(y_pred, label='预测值', color='red', linestyle='--')
    plt.title('LSTM预测结果对比')
    plt.xlabel('样本索引')
    plt.ylabel('目标值')
    plt.legend()
    plt.show()

    return  y_pred, y_true


if __name__ == "__main__":
    # 读取数据（替换为你的数据路径）
    df = pd.read_csv("./lang_rag/data/environment_data_export_2025-10-16_164127.csv")
    # 可调整参数：序列长度、训练轮数、隐藏层大小
    print(train(df))
