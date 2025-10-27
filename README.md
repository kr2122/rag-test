# 基于RAG的机器学习模型推荐系统

本仓库实现了一个结合检索增强生成（RAG）技术的自动化机器学习模型推荐与训练平台。该系统能够自动分析数据集特征，基于知识库推荐最优模型，并完成端到端的模型训练与评估流程。


## 核心功能

- **智能数据集分析**：自动解析数据集结构（特征类型、分布规律）、识别任务类型（分类/回归/时序预测）
- **RAG驱动的模型推荐**：基于检索增强生成技术，结合内置知识库为数据集匹配最适合的机器学习模型
- **自动化模型训练**：支持多种经典机器学习与深度学习模型的一键训练
- **全面性能评估**：内置多维度评估指标，自动生成模型性能报告


## 支持的模型列表

| 模型类型         | 具体模型                 |
|------------------|--------------------------|
| 传统机器学习     | KNN、随机森林、XGBoost   |
| 深度学习         | CNN、RNN、LSTM           |


## 项目结构
```plaintext
rag-test/
├── lang_rag/               # 核心功能模块
│   ├── doc/                # RAG知识库文档（模型特性、适用场景等）
│   ├── data/               # 数据集存储目录
│   │   ├── log_ekf_gps_drift_0.csv           # 示例GPS漂移数据集
│   │   └── log_vehicle_angular_velocity_0.csv # 示例车辆角速度数据集
│   ├── model/              # 模型实现代码
│   │   ├── lstm_model.py        # LSTM模型定义
│   │   ├── random_forest_model.py # 随机森林模型定义
│   │   ├── rnn_model.py         # RNN模型定义
│   │   └── xgboost_model.py     # XGBoost模型定义
│   ├── data_analysis.py    # 数据集分析模块（特征提取、任务识别）
│   ├── model_py.py         # 模型训练与评估接口
│   └── rag_v1.py           # RAG核心逻辑（检索+生成推荐结果）
├── rag_model_system/       # 系统辅助模块
│   ├── download.py         # 模型/数据集下载工具
│   ├── hello.py            # 测试脚本
│   └── test.py             # 模型性能测试模块
├── vector_db/              # RAG向量数据库（存储知识库向量表示）
└── README.md               # 项目说明文档
```
```
## 快速开始

### 环境要求

- Python 3.8+
- 依赖库：pandas、numpy、scikit-learn、torch、langchain、faiss、sentence-transformers

### 安装步骤

```bash
# 克隆仓库
git clone https://github.com/你的用户名/rag-test.git
cd rag-test

# 安装依赖
pip install -r requirements.txt  # 若未创建requirements.txt，可手动安装上述依赖
```


### 使用流程

1. 将你的数据集放入 `lang_rag/data/` 目录（支持CSV格式）
2. 运行主程序：
   ```bash
   python lang_rag/rag_v1.py
   ```

3. 系统会自动执行：
   - 数据集特征分析（输出特征类型、分布统计）
   - 基于RAG的模型推荐（返回推荐模型及适配理由）
   - 模型自动训练与评估（输出准确率、MSE等关键指标）


## 许可证

本项目采用MIT许可证，详情参见LICENSE文件。


## 联系方式

如有问题或建议，欢迎提交Issue或联系：2624964839@qq.com

```

这个README基于你仓库的实际文件结构（包含`lang_rag/data`下的CSV文件、`model`目录下的模型文件等）进行了优化，补充了更具体的项目细节和使用流程，方便其他开发者快速理解和使用你的项目。
```



