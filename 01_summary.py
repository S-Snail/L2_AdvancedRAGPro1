"""
    1、摘要索引
"""
import uuid

from langchain_chroma import Chroma
from langchain_classic.retrievers import MultiVectorRetriever
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.stores import InMemoryStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sqlalchemy.testing.suite.test_reflection import metadata
from langchain_core.runnables import RunnableParallel  # 可运行的映射

from models import get_ali_clients

# 摘要索引示例代码

# 1. 提取，分割，块（比较大）
# 2. 块，生成摘要
# 3. 将摘要向量化，摘要和原始文档要建立联系
# 4. 摘要向量存储到向量数据库中，原始文档存储（不是存储到向量数据库）
# 5. 匹配摘要向量，返回原文档


# 注意事项：
# 1.摘要质量至关重要
#   “这是一个关于deepseek的介绍文章”：太过于简单，丢失了关键信息
# 2.存储开销比较大
#  不用摘要：只存储知识块对应的向量。摘要索引：既要存储向量，还要存储原始文档
# 3.一致性问题，维护一致性
#  通过uuid 将原始文档和摘要一一对应

# 获得访问大模型和嵌入模型客户端
client, embeddings_model = get_ali_clients()

# 初始化文档加载器
loader = TextLoader("./deepseek百度百科.txt", encoding="utf-8")

# 加载文档
docs = loader.load()
print("---------------------------------- 文档内容 ----------------------------------")
# print("文档内容：\n", docs[0].page_content)

print("---------------------------------- 分块 ----------------------------------")
# 初始化递归文档分割器（设置块大小和重叠）
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024,
    chunk_overlap=100
)
docs = text_splitter.split_documents(docs)
# print("块数量：", len(docs))

# 创建摘要生成链
chain = (
        {"doc": lambda x: x.page_content} |
        ChatPromptTemplate.from_template("总结下面的文档：\n\n{doc}") |
        client |
        StrOutputParser()
)
print("准备生成文档摘要，请耐心等待...")
# 批量生成文档摘要（最大并发数5）
summaries = chain.batch(docs, {"max_concurrency": 5})
# print("生成的摘要：\n", summaries)


# 初始化Chroma实例（用于存储摘要向量）
vectorstore = Chroma(
    collection_name="summaries",
    embedding_function=embeddings_model
)

# 初始化内存字节存储（用于存储原始文档）
store = InMemoryStore()

# 初始化多向量检索器（结合向量存储和文档存储）
id_key = "doc_id"
retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    byte_store=store,
    id_key=id_key
)

# 为每个文档生成唯一ID,该ID用于关联原始文档和摘要
doc_ids = [str(uuid.uuid4()) for _ in docs]

# 将文档摘要转换为LangChain中的Document
summary_docs = [
    Document(page_content=s, metadata={id_key: doc_ids[i]})
    for i, s in enumerate(summaries)
]
# print(summary_docs)

# 将摘要添加到向量数据库
print("准备将摘要添加到向量数据库...")
retriever.vectorstore.add_documents(summary_docs)

# 将原始文档存储到字节存储（使用ID关联）
print("准备将原始文档存储到字节存储...")
# mset：批量设置键值对
# list(zip(doc_ids,docs))：将ID和文档配对
retriever.docstore.mset(list(zip(doc_ids, docs)))

# # 手动测试代码 - 相似性搜索
# query = "deepseek企业事件"
# sub_docs = retriever.vectorstore.similarity_search(query)
# print("----------------- 匹配到摘要内容 ----------------")
# print(sub_docs[0])
#
# # 获取第一个匹配摘要的ID
# matched_id = sub_docs[0].metadata[id_key]
#
# print("----------------- 匹配到的原始文档 ----------------")
# # 通过id获取原始文档
# original_doc = retriever.docstore.mget(matched_id)
# print(original_doc)
# # 执行相似性搜索测试 --- 完成

prompt = ChatPromptTemplate.from_template("根据下面的文档回答问题:\n\n{doc}\n\n问题: {question}")
# 生成回答问题链
# retriever.invoke将上面对摘要进行检索，但是通过关联ID获得原始文档，最终返回原始文档的过程全都包含完成了
chain = RunnableParallel({
    "doc": lambda x: retriever.invoke(x["question"]),
    "question": lambda x: x["question"]
}) | prompt | client | StrOutputParser()

# 生成问题回答
query = "deepseek的企业事件"
answer = chain.invoke({"question": query})
print("----------------- 问题回答 ----------------")
print(answer)

# retriever.invoke(query)  # 1.向量数据库中检索摘要向量   2.匹配对应的原始文档并返回
# retrieved_docs = retriever.invoke(query)
# print("-------------检索到的文档--------------")
# print(retrieved_docs)