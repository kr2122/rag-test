from operator import itemgetter
from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader,TextLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
# from langchain_community.embeddings import OllamaEmbeddings
from langchain_ollama.embeddings import OllamaEmbeddings
from data_analysis import analyze_excel
from model_py import train_model

# 1. 设置模型

# llm = ChatOpenAI(
#     model="deepseek-ai/DeepSeek-R1",
#     temperature=0,
#     api_key="sk-5hmjTgNvEwpzxliLy8Ub6SBkjdt5GkotJUcr9Y8HoW8CQ7bX",
#     base_url="https://api2.aigcbest.top/v1"
# )
llm = ChatOllama(
    model="qwen2.5:7b",
    # temperature=0,
    # api_key="sk-5hmjTgNvEwpzxliLy8Ub6SBkjdt5GkotJUcr9Y8HoW8CQ7bX",
    base_url="http://10.108.13.254:11434"
)

# nomic-embed-text:latest
# embed_model = HuggingFaceEmbeddings(
#     model_name="BAAI/bge-base-zh-v1.5"
# )
embed_model = OllamaEmbeddings(
    model="embeddinggemma:300m",
    # temperature=0,
    # api_key="sk-5hmjTgNvEwpzxliLy8Ub6SBkjdt5GkotJUcr9Y8HoW8CQ7bX",
    base_url="http://10.108.13.254:11434"
)



# 2. 数据处理
file_dir = Path('./lang_rag/doc')
# print(file_dir)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)
vector_store = Chroma(embedding_function=embed_model,persist_directory="./vector_db")
retriever = vector_store.as_retriever(search_kwargs={"k": 5})

prompt_template = PromptTemplate.from_template(
    """
    你是一个严谨的RAG助手
    基于以下上下文回答问题，结合提供的信息给出适合的模型，直接说出模型名称,不用输出多余内容。
    例如：
    q：根据以下数据特征，推荐最合适的机器学习模型：
    a：xgboost
    上下文: {context}
    问题: {question}
    回答:
    """
)



# 3. 编造链

# chain = {
#     RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
#     | prompt_template
#     | llm
#     | StrOutputParser()
# }
chain = (
    RunnableParallel({
        "context": retriever,
        "question": RunnablePassthrough()
    })
    | prompt_template  # 将并行结果传入提示词模板
    | llm  # 传入大模型
    | StrOutputParser()  # 解析输出为字符串
)



# 4.初始化知识库
if __name__ == "__main__":
    # docs = DirectoryLoader(str(file_dir),loader_cls=TextLoader,loader_kwargs={"encoding": "utf-8"}).load()
    # all_splits = text_splitter.split_documents(docs)
    # vector_store.add_documents(all_splits)
#更换嵌入模型要删除数据库，重新初始化
    data_path = "./lang_rag/data/environment_data_export_2025-10-16_164127.csv"  # 你自己的Excel文件路径
    data_summary = analyze_excel(data_path)
    print("📊 数据分析结果：")
    print(data_summary)
    print("——————————————————————————————————")

    # 3. 将数据描述作为问题输入RAG链
    question = f"根据以下数据特征，推荐最合适的机器学习模型：{data_summary}"
    result = chain.invoke(question)
    print("🤖 模型推荐结果：")
    print(result)

    df = pd.read_csv(data_path)
    
    try:
        model, y_pred = train_model(result, df)
        print("✅ 模型训练完成！")
    except Exception as e:
        print(f"❌ 模型训练失败: {e}")







    # # 简单处理：假设最后一列为目标列
    # X = df.iloc[:, :-1].values
    # y = df.iloc[:, -1].values

    # # 若是时间序列，可改为窗口化输入，暂时用普通拆分
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # # 如果模型是 LSTM，需要 3D 输入 [samples, timesteps, features]
    # if "lstm" in result.lower():
    #     X_train = np.expand_dims(X_train, axis=1)
    #     X_test = np.expand_dims(X_test, axis=1)

    # try:
    #     model, y_pred = train_model(result, X_train, y_train, X_test, y_test)
    #     print("✅ 模型训练完成！")
    # except Exception as e:
    #     print(f"❌ 模型训练失败: {e}")
    # # print(chain.invoke("我想做数据预测,帮我选择一个合适的模型"))









