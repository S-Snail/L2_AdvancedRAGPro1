"""
   # Post-Retrieval后检索-上下文压缩：目的是在检索文档后，对检索到的文档进行压缩，以减少无关信息，提高查询结果的相关性和质量
"""
from langchain_chroma import Chroma
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor, LLMChainFilter, EmbeddingsFilter, \
    DocumentCompressorPipeline
from langchain_community.document_loaders import TextLoader
from langchain_community.document_transformers import EmbeddingsRedundantFilter
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter

from models import get_ali_clients, get_ali_rerank

"""
    1. 加载文档
    2. 存储 --- 向量化存储
    3. 检索 --- 根据用户的问题进行相似度检索
    4. 把文档传递给大模型 进行回答
"""

# 获得访问大模型和嵌入模型客户端
llm, embeddings_model = get_ali_clients()


# 格式化输出
def pretty_print_docs(docs):
    print(f"\n{'-'}*100".join(
        [f"Document {i + 1}:\n\n" + d.page_content for i, d in enumerate(docs)]
    ))


documents = TextLoader("./deepseek百度百科.txt", encoding="utf-8").load()
text_spliter = RecursiveCharacterTextSplitter(
    chunk_overlap=100,
    chunk_size=1024
)
texts = text_spliter.split_documents(documents)

# 使用基础检索器
retriever = Chroma.from_documents(texts, embeddings_model).as_retriever()

# docs = retriever.invoke("deepseek的发展历程")
# print("------------------------------ 压缩前 ---------------------------")
# pretty_print_docs(docs)

"""
    LLMChainExtractor压缩
    利用大语言模型（LLM）从检索到的文档中提取与查询相关的信息
    它会将文档内容输入到LLM中，让LLM分析并提取出最相关的部分，从而实现文档的压缩
    需要调用大语言模型，计算成本较高，并且处理速度相对较慢
"""
print("-------------------------- 第一种：LLMChainExtractor压缩 -------------------------------")
# 使用上下文压缩检索器
compressor = LLMChainExtractor.from_llm(llm)
# ContextualCompressionRetriever -- 创建文档压缩器
compression_retriever = ContextualCompressionRetriever(
    base_retriever=retriever,
    base_compressor=compressor
)
compressed_docs = compression_retriever.invoke("deepseek的发展历程")
print("----------------------- 第一种：压缩后 ------------------------------")
pretty_print_docs(compressed_docs)

"""
    LLMChainFilter 同样基于大语言模型
    工作方式：
    对检索到的文档进行过滤，只保留与查询相关的文档。
    它会让LLM判断每个文档是否与查询相关，如果相关则保留，否则过滤掉。
    相对 LLMChainExtractor来说，计算成本可能稍微低一些，因为它只进行简单的过滤操作。同时，也能有效地筛选出相关文档。
    仍然依赖于大语言模型调用，计算成本和处理速度仍然是需要考虑的因素。
"""
print("----------------------------- 第二种：LLMChainFilter压缩后 ------------------------------------")
# LLMChainFilter 是稍微简单但更强大的压缩器
_filter = LLMChainFilter.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_retriever=retriever,
    base_compressor=_filter
)
compressed_docs = compression_retriever.invoke("deepseek的发展历程")
pretty_print_docs(compressed_docs)

print("----------------------------- 第三种：EmbeddingsFilter压缩后 --------------------------")
# 对每个检索到的文档进行额外的LLM调用既昂贵又缓慢
# EmbeddingsFilter 通过嵌入文档和查询并仅返回那些与查询足够相似嵌入的文档来提供更便宜且更快的选项
"""
    EmbeddingsFilter
    通过计算文档和查询的嵌入向量之间的相似度，返回与查询相似度超过设定阈值的文档
    本质上就是利用嵌入模型将文档和查询转换为向量表示，然后使用余弦相似度方法来衡量它们之间的相似性。
    计算成本低了，处理速度较快，因为它主要是基于向量计算，而不需要调用大语言模型。
    相似度的判断可能不够准确，因为它只基于嵌入向量的相似度，而没有考虑语义的深层次理解。
"""
embeddings_filter = EmbeddingsFilter(
    embeddings=embeddings_model, similarity_threshold=0.6
)
compression_retriever = ContextualCompressionRetriever(
    base_retriever=retriever,
    base_compressor=embeddings_filter
)
compressed_docs = compression_retriever.invoke(
    "deepseek的发展历程"
)
pretty_print_docs(compressed_docs)

print("------------------------------ 第四种：组合压缩后 -----------------------------------")
# DocumentCompressorPipeline轻松的按顺序组合多个压缩器
"""
    首先TextSpliter可以用作文档转换器，将文档分割成更小的块，
    然后EmbeddingsRedundantFilter 根据文档之间嵌入的相似性来过滤掉冗余文档，
    该过滤操作以文本的嵌入向量为依据，也就是借助余弦相似度来衡量文本之间的相似程度，
    进而判定是否存在冗余，它会把文本列表转化成对应的嵌入向量，然后计算每对文本之间的余弦相似度。
    一旦相似度超出设定的阈值，就会将其中一个文本判定为冗余并过滤掉。
    最后，EmbeddingFilter 根据与查询的相关性进行过滤
"""
# 创建字符文本分割器，设置每个块的大小为300个字符，块之间无重叠，分隔符为句号和空格
spliter = CharacterTextSplitter(chunk_size=300, chunk_overlap=0, separator=". ")
# 创建基于嵌入的冗余过滤器，使用之前获取的嵌入模型
redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings_model)  # 去重，冗余
# 创建基于嵌入的相关性过滤器，使用之前获取的嵌入模型，设置相似度阈值为0.6
relevant_filter = EmbeddingsFilter(embeddings=embeddings_model, similarity_threshold=0.6)
# 创建文档压缩管道：先分割文档，然后过滤冗余文档，最后根据查询相关性过滤文档。
pipeline_compressor = DocumentCompressorPipeline(
    transformers=[spliter, redundant_filter, relevant_filter]
)
# 创建上下文压缩检索器
reranker = get_ali_rerank(top_n=5)
compression_retriever = ContextualCompressionRetriever(
    base_retriever=retriever,
    base_compressor=pipeline_compressor
)
# 执行查询并获取压缩后的文档
compressed_docs = compression_retriever.invoke("deepseek的发展历程")
pretty_print_docs(compressed_docs)
