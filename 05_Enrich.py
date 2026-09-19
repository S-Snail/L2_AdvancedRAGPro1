"""
    完善问题
"""
import json
import os

from dotenv import load_dotenv
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from langchain_community.chat_message_histories import ChatMessageHistory

load_dotenv()

# 加载模型参数
model_name = "qwen-max"
apikey = os.getenv("DASHSCOPE_API_KEY")
api_base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"

llm = ChatOpenAI(
    api_key=apikey,
    model=model_name,
    base_url=api_base_url
)

embeddings_model = DashScopeEmbeddings(model="text_embeddings-v1")

# 示例代码模版
templates = {
    "定机票": ["起点", "终点", "时间", "座位等级", "座位偏好"],
    "定酒店": ["城市", "入住日期", "退房日期", "房型", "人数"]
}
print(str(list(templates.keys())))

# 意图识别提示模板
intent_prompt = PromptTemplate(
    input_variables=["user_input", "templates"],
    template="根据用户输入 '{user_input}'，选择最合适的业务模板。可用模板如下：{templates}。请返回模板名称。"
)

# 创建意图识别链
intent_chain = intent_prompt | llm

# 模拟用户输入
user_input = input("请输入问题：")

# 识别意图
intent = intent_chain.invoke({"user_input": user_input, "templates": str(list(templates.keys()))}).content

print("意图：", intent)
# 获取对应的模板
selected_template = templates.get(intent)
print("模板：", selected_template)

# 补充信息提示模板
info_prompt = f"""
    请根据用户原始问题和模板，判断原始问题是否完善。如果问题缺乏需要的信息，请生成一个友好的请求，明确指出需要补充的信息。若问题完善后，返回包含所有信息的完整问题。
    
    ### 原始问题
    {user_input}
    
    ### 模板
    {",".join(selected_template)}
    
    ### 输出示例
    {{
        "isComplete":true,
        "content":"`完整问题`"
    }}
    {{
        "isComplete":false,
        "content":"`友好的引导到需要补充信息`"
    }}
"""

# 历史记录
chat_history = ChatMessageHistory()

# 聊天模板
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个信息补充助手，任务是分析用户问题是否完整。"),
        ("placeholder", "{history}"),  # 历史记录的占位
        ("human", "{input}")
    ]
)

# 补充信息链
info_chain = prompt | llm

# 自动处理历史记录，将记录注入输入并在每次调用后更新它
with_history_message = RunnableWithMessageHistory(
    info_chain,
    lambda session_id: chat_history,
    input_messages_key="input",
    history_messages_key="history"
)

# 判断问题是否完整，如果不完整，则生成追问请求
info_request = with_history_message.invoke(
    input={"input": info_prompt},
    config={"configurable": {"session_id": "unused"}}).content

parser = JsonOutputParser()
json_data = parser.parse(info_request)

# 循环判断是否完整，并提交用户补充信息 --- LLM根据模版信息是否全部包含，以此来判断输入是否完整
while json_data.get("isComplete", False) is False:

    try:
        # 显示引导信息并等待用户输入
        user_answer = input(f"\033[1;33m{json_data['content']}\033[0m\n你的回复：")

        # 提交补充信息给AI处理
        info_request = with_history_message.invoke(
            input={"input": user_answer},
            config={"configurable": {"session_id": "unused"}}
        ).content

        # 解析AI响应
        json_data = parser.parse(info_request)

    except json.JSONDecoder:
        print("\033[1;31m[错误] AI返回了无效的JSON格式，请重试\033[0m")
        continue
    except KeyError:
        print("\033[1;31m[错误] 响应格式异常，正在终止流程\033[0m")
        break

# 输出最终结果
print(f"\033[1;32m[最终查询] {info_request}\033[0m")
