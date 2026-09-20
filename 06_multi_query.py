"""
    Multi-query：多路召回
"""
from operator import itemgetter

from langchain_chroma import Chroma
from langchain_classic.retrievers import MultiQueryRetriever
from langchain_community.document_loaders import TextLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnableMap
from langchain_text_splitters import RecursiveCharacterTextSplitter

from models import get_ali_clients

# 获得访问大模型和嵌入模型客户端
llm, embeddings_model = get_ali_clients()

# 加载文档
loader = TextLoader("./deepseek百度百科.txt", encoding="utf-8")
docs = loader.load()

# 创建文档分割器，并分割文档
text_spliter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
splits = text_spliter.split_documents(docs)

# 创建向量数据库
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings_model)

# 创建检索器
retriever = vectorstore.as_retriever()

# 检索测试
relevant_docs = retriever.invoke('deepseek的应用场景')
print(relevant_docs)
print("检索测试", f"检索到的文档个数: {len(relevant_docs)}")
# 检索测试 - 完成

# 创建prompt模板
template = """请根据下面给出的上下文来回答问题
${context}
问题：{question}
"""

# 由模板生成prompt
prompt = ChatPromptTemplate.from_template(template)
chain = RunnableMap({
    "context": lambda x: relevant_docs,
    "question": lambda x: x["question"]
}) | prompt | llm | StrOutputParser()

print("--------------优化前-------------------")
response = chain.invoke({"question": "deepseek的应用场景"})
print("大模型生出的不同视角的问题为：", response)

print("--------------开始优化-------------------")
# 方法一：使用langchain的MultiQueryRetriever
# 引入日志组件查看llm在原查询的基础上生成的多个查询
# import logging
# logging.basicConfig()
# logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)
prompt = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI language model assistant. Your task is
    to generate 4 different versions of the given user
    question to retrieve relevant documents from a vector  database.
    By generating multiple perspectives on the user question,
    your goal is to help the user overcome some of the limitations
    of distance-based similarity search. Provide these alternative
    questions separated by newlines. Original question: {question}""",
)
retriever_from_llm = MultiQueryRetriever.from_llm(
    retriever=retriever,
    llm=llm,
    prompt=prompt
)
unique_docs = retriever_from_llm.invoke({"question": "deepseek的应用场景"})
print(unique_docs)
print("优化后检索到的文档个数：", len(unique_docs))

print("--------------优化后-------------------")

template = """请根据下面给出的上下文来回答问题
{context}
问题：{question}
"""

prompt = ChatPromptTemplate.from_template(template)
final_rag_chain = (
        {"context": lambda x: retriever_from_llm.invoke(x["question"]),
         "question": itemgetter("question")}
        | prompt
        | llm
        | StrOutputParser()
)

question = "deepseek的应用场景"
response = final_rag_chain.invoke({"question": question})
print(response)
