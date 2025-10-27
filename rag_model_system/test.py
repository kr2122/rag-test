# rag_model_system.py
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import faiss
from sentence_transformers import SentenceTransformer
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense
import torch
import torch.nn as nn 
from langchain_openai import ChatOpenAI
# ===========================
# 1. 模型训练函数
# ===========================
def train_knn(data, labels):
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
    model = KNeighborsRegressor(n_neighbors=3)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"KNN MSE: {mse:.4f}")
    return model

# def train_lstm(data, labels, epochs=5):
#     X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
#     X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
#     X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

#     model = Sequential()
#     model.add(LSTM(32, input_shape=(X_train.shape[1], 1)))
#     model.add(Dense(1))
#     model.compile(optimizer='adam', loss='mse')
#     model.fit(X_train, y_train, epochs=epochs, verbose=0)
#     y_pred = model.predict(X_test)
#     mse = mean_squared_error(y_test, y_pred)
#     print(f"LSTM MSE: {mse:.4f}")
#     return model

# def train_cnn(data, labels, epochs=5):
#     # 这里用简单的一维卷积示例
#     X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
#     X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
#     X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

#     model = Sequential()
#     model.add(tf.keras.layers.Conv1D(16, 3, activation='relu', input_shape=(X_train.shape[1], 1)))
#     model.add(tf.keras.layers.GlobalAveragePooling1D())
#     model.add(Dense(1))
#     model.compile(optimizer='adam', loss='mse')
#     model.fit(X_train, y_train, epochs=epochs, verbose=0)
#     y_pred = model.predict(X_test)
#     mse = mean_squared_error(y_test, y_pred)
#     print(f"CNN MSE: {mse:.4f}")
#     return model
# 封装一个简单的训练循环工具函数
def train_model(model, X_train, y_train, X_test, y_test, epochs=5, lr=0.001, verbose=0):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # 训练循环
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        
        if verbose and (epoch % (epochs//5) == 0 or epoch == epochs-1):
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
    
    # 评估
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test)
        mse = mean_squared_error(y_test.numpy(), y_pred.numpy())
    print(f"CNN MSE: {mse:.4f}")
    return model

# 主函数：保持与Keras版本相似的简洁性
def train_cnn(data, labels, epochs=5):
    # 数据分割与预处理
    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.2, random_state=42
    )
    # 调整形状并转为张量
    X_train = torch.FloatTensor(X_train.reshape(-1, X_train.shape[1], 1))
    X_test = torch.FloatTensor(X_test.reshape(-1, X_test.shape[1], 1))
    y_train = torch.FloatTensor(y_train).view(-1, 1)
    y_test = torch.FloatTensor(y_test).view(-1, 1)
    
    # 定义模型（使用Sequential风格的简洁写法）
    model = nn.Sequential(
        nn.Conv1d(1, 16, 3),  # 对应Conv1D(16, 3)
        nn.ReLU(),            # 激活函数单独作为一层
        nn.AdaptiveAvgPool1d(1),  # 对应GlobalAveragePooling1D
        nn.Flatten(),         # 展平维度
        nn.Linear(16, 1)      # 对应Dense(1)
    )
    
    # 调用封装的训练函数
    return train_model(model, X_train, y_train, X_test, y_test, epochs=epochs)
# ===========================
# 2. 模型库
# ===========================
model_library = [
    {
        "name": "KNN",
        "tags": ["小数据集", "回归", "预测"],
        "train_function": train_knn
    },
    # {
    #     "name": "LSTM",
    #     "tags": ["时间序列", "长序列预测", "预测"],
    #     "train_function": train_lstm
    # },
    {
        "name": "CNN",
        "tags": ["卷积", "特征提取", "预测"],
        "train_function": train_cnn
    }
]

# ===========================
# 3. 向量化 & FAISS
# ===========================
# embed_model = SentenceTransformer('all-MiniLM-L6-v2')
dim = 384
# 1. 初始化 LLM
embed_model = ChatOpenAI(
    model="gemini-embedding-001",
    temperature=0,
    api_key="sk-5hmjTgNvEwpzxliLy8Ub6SBkjdt5GkotJUcr9Y8HoW8CQ7bX",
    base_url="https://api2.aigcbest.top/v1"
)

# 2. 用 LLM 获取文本向量
def text_to_embedding(text):
    # 这里可以让 LLM 返回一个向量列表，比如 prompt: "请把这段文本编码成长度384的浮点向量，用逗号分隔"
    prompt = f"把下面文本转换成384维浮点向量，用逗号分隔:\n{text}"
    response = embed_model.invoke([prompt])  # 根据你的 SDK 替换调用方法
    vec = np.array([float(x) for x in response.content.split(",")], dtype='float32')
    return vec
faiss_index = faiss.IndexFlatL2(dim)
meta_data = []

# 构建向量库
for m in model_library:
    text = m['name'] + " " + " ".join(m['tags'])
    vec = text_to_embedding(text)
    faiss_index.add(np.array([vec], dtype='float32'))
    meta_data.append(m)

# ===========================
# 4. RAG 检索函数
# ===========================
def retrieve_best_model(task_description):
    query_vec = text_to_embedding(task_description)
    D, I = faiss_index.search(np.array([query_vec], dtype='float32'), k=1)
    return meta_data[I[0][0]]

# ===========================
# 5. 测试流程
# ===========================
if __name__ == "__main__":
    # 模拟数据
    np.random.seed(42)
    data = np.random.rand(100, 10)  # 100条样本, 10个特征
    labels = np.random.rand(100)

    # 用户输入任务描述
    task = "我想做时间序列预测未来数据趋势"
    best_model = retrieve_best_model(task)
    print("RAG 推荐模型:", best_model['name'])

    # 训练模型
    trained_model = best_model['train_function'](data, labels)
