from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
# from langchain_deepseek import ChatDeepSeek
from langchain_core.messages import HumanMessage, AIMessage
llm = ChatOpenAI(
    model="deepseek-ai/DeepSeek-R1",  # 聊天模型名称（正确）
    temperature=0,          # 生成随机性，0表示确定性最高
    api_key="sk-5hmjTgNvEwpzxliLy8Ub6SBkjdt5GkotJUcr9Y8HoW8CQ7bX",
    base_url="https://api2.aigcbest.top/v1"
)

# 构造用户消息（用 HumanMessage 包装）
message = HumanMessage(content="什么是大语言模型？")

# 调用模型（传入消息列表）
response = llm.invoke([message])

# 提取回答内容
print(response.content)