"""
    2、父子索引

    父子索引-痛点分析：
        我们在利用大模型进行文档检索的时候，常常会有相互矛盾的需求，比如：
        1、你可能希望得到较小的文档块，以便它们Embedding以后能够最准确地反映出文档的含义，如果文档块太大（会被大量无关内容稀释），
        Embedding就失去了意义。
        2、你可能希望得到较大的文档块以保留较多的内容，然后将它们发送给LLM以便得到全面且正确的答案。

        检索粒度和理解粒度的矛盾：
        1、检索需要细粒度：越细越容易匹配用户问题中的关键词（精度）。
        2、生成需要粗粒度：LLM推理需要完整上下文才能正确理解（完整性）。传统RAG无法兼顾二者，陷入精度 vs 完整性的两难。

    父子索引-基本思路：
        1、文档被分割成一个层级化的块结构，随后用最小的叶子块进行索引
        2、在检索过程中检索出top-k个叶子块
        3、如果存在n个叶子块都指向同一个更大的父块，那么我们就用这个父块来替换这些子块，并将父块送入大模型用于生成答案。

    示例代码
"""
from langchain_chroma import Chroma
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_community.document_loaders import TextLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel
from langchain_core.stores import InMemoryStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from models import get_ali_clients

# 获取访问大模型和嵌入模型客户端
client, embeddings_model = get_ali_clients()

# 加载数据
loader = TextLoader("./deepseek百度百科.txt", encoding="utf-8")
docs = loader.load()

# 查看长度
print(f"文章的长度：{len(docs[0].page_content)}")

# 子块是父块内容的子集
# 创建主文档分割器
parent_spliter = RecursiveCharacterTextSplitter(
    chunk_size=1024, chunk_overlap=100
)

# 创建子文档分割器
child_spliter = RecursiveCharacterTextSplitter(
    chunk_size=256, chunk_overlap=30
)

# 创建向量数据库对象
vectorstore = Chroma(
    collection_name="splits_parents",
    embedding_function=embeddings_model
)

# 创建内存存储对象
store = InMemoryStore()

# 创建父子文档检索器，帮我们通过检索子块，返回父块文档
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,  # 文档存储对象
    child_splitter=child_spliter,  # 子文档分割器，子文档存储到向量数据库
    parent_splitter=parent_spliter,  # 主文档分割器，主文档存储到内存中
    search_kwargs={"k": 5}
)

# 添加文档集
retriever.add_documents(docs)

print(f"主文档块数量：{len(list(store.yield_keys()))}")

# 测试 - 相似性搜索
"""
这里我们通过向量数据库的similarity_search方法搜索出来的是与用户问题相关的子文档块的内容，
下面我们使用检索器的get_relevant_documents的方法来对这个问题进行检索，
它会返回该子文档块所属的主文档块的全部内容： 
"""
# print("------------similarity_search------------------------")
# sub_docs = vectorstore.similarity_search("deepseek的应用场景", k=5)
# print(f"子文档块数量：{len(sub_docs)}")
# # print(sub_docs[0].page_content)
# print([doc.page_content for doc in sub_docs])
#
# print("------------get_relevant_documents-----------通过子找父-------------")
# retrieved_docs = retriever.invoke("deepseek的应用场景")
# # print(retrieved_docs[0].page_content)
# print([doc.page_content for doc in retrieved_docs])
# # 测试 - 相似性搜索 - 完成
# exit()

# 创建prompt模板
template = """
请根据下面给出的上下文来回答题：
{context}
问题：{question}
"""

# 由模板生成prompt
prompt = ChatPromptTemplate.from_template(template)

# 创建chain
chain = RunnableParallel({
    "context": lambda x: retriever.invoke(x["question"]),
    "question": lambda x: x["question"]
}) | prompt | client | StrOutputParser()

print("------------模型回复------------------------")
response = chain.invoke({"question": "deepseek的应用场景"})
print(response)
