"""
    Post-Retrieval后检索优化
    2、重排序 --- RAG-Fusion - 使用算法进行重排序
    RAG-Fusion是一种搜索方法，通过使用多重查询生成和倒数排名融合（Reciprocal Rank Fusion）对搜索结果进行重新排序。
    在Multi Query的基础上，对其检索结果进行重新排序(即reranking)后 输出Top-K个最相关文档，最后将这top-k个文档喂给LLM并生成最终的答案。
"""
from langchain_chroma import Chroma
from langchain_core.load import dumps, loads
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from models import get_ali_clients

# 获得访问大模型和嵌入模型客户端
llm, embedding_model = get_ali_clients()

texts = [
    "人工智能在医疗诊断中的应用。",
    "人工智能如何提升供应链效率。",
    "NBA季后赛最新赛况分析。",
    "传统法式烘焙的五大技巧。",
    "红楼梦人物关系图谱分析。",
    "人工智能在金融风险管理中的应用。",
    "人工智能如何影响未来就业市场。",
    "人工智能在制造业的应用。",
    "今天天气怎么样",
    "人工智能伦理：公平性与透明度。"
]

# 创建向量数据库对象
vectorstore = Chroma.from_texts(
    texts=texts, embedding=embedding_model
)
retriever = vectorstore.as_retriever()

# 定义prompt如下：
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that generates multiple search queries based on a single input query."),
    ("user", "Generate multiple search queries related to: {original_query}"),
    ("user", "OUTPUT (4 queries):")
])

# 创建多重查询chain
generate_queries = (
        prompt | llm | StrOutputParser() | (lambda x: x.split("\n"))
)

original_query = "人工智能应用"
queries = generate_queries.invoke({"original_query": original_query})
print(f"原始查询：\n\t{original_query}\n生成的查询：\n\t{queries}")


def reciprocal_rank_fusion(results: list[list], k=60):
    """互逆排序融合算法，用于合并多个排序文档列表
    Args:
        results：包含多个排序文档列表的二位列表
        k：融合公式中的平滑参数（默认60），值越小排名影响越大
    RRF 分数计算
    假设我们有两个查询的检索结果
        results = [
        [doc1,doc2,doc3],# 第一个查询的结果
        [doc2,doc4,doc1] # 第二个查询的结果
        ]
        k = 60
        在遍历过程中，doc1 在第一个查询中排名第0，在第二个查询中排名第 2 ，它的RRF 分数计算如下：
        第一个查询：1/(0 + 60) = 1/60
        第二个查询：1/(2 + 60) = 1/62
        累计分数：1/60 + 1/62
    通过这种计算方式，每个文档都会根据其在各个查询结果中的排名获得一个综合的分数，最终根据这个分数进行降序排序，得到融合后的结果。
    Returns:
        按融合分数降序排列的文档列表，每个元素为（文档对象，分数）元组
    """
    # 初始化融合分数字典（key = 序列化文档，value = 累计分数）
    fused_scores = {}

    # 遍历每个检索结果列表（每个查询对应的结果）
    for docs in results:
        for rank, doc in enumerate(docs):
            # 序列化文档对象为字符串（用于唯一标识）
            doc_str = dumps(doc)
            # 初始化文档得分（如果是首次出现）
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            # 计算并累加RRF：1 / (当前排名 + k)
            # 排名越靠前（rank值小）的文档获得的分数越高

            fused_scores[doc_str] += 1 / (rank + k)

    # 按融合分数降序排序（分数越高排名越前）生成按相关度从高到低排列的文档列表
    reranked_results = [
        (loads(doc), score)  # 反序列化还原文档对象
        # sorted 对 fused_scores 字典进行排序
        # fused_scores.items() 返回键值对列表 [(doc_str1, score1), (doc_str2, score2), ...]
        # key=lambda x: x[1] 指定按分数（元组的第二个元素）排序
        # reverse=True 降序排序 （分数高的在前）
        for doc, score in sorted(fused_scores.items(),
                                 key=lambda x: x[1],  # 取分数
                                 reverse=True)
    ]
    print(reranked_results)
    return reranked_results


original_query = "人工智能的应用"
'''
generate_queries会生成4个多角度的query,
retriever.map()的作用是根据generate_queries的结果映射出4个retriever(可以理解为同时复制出4个retriever)
与generate_queries生成的4个query对应，
并为每个query检索出来的一组相关文档集(默认为4个相关文档)，
那么4个query总共可以生成16个相关文档。
最后会经过RRF算法重新排序后输出最相关的文档

'''
chain = generate_queries | retriever.map() | reciprocal_rank_fusion
# 输入结果列表
result_list = chain.invoke({"original_query": original_query})
print("输出重排序结果列表", result_list)
# 提取文档内容和对应分数
contents = [doc[0].page_content for doc in result_list]
scores = [doc[1] for doc in result_list]

combined_tuples = list(zip(contents, scores))
print("--" * 15, "最相关的文档及其得分：")
for item in combined_tuples:
    print(item)

print("--" * 15, "分析一下这些分数是如何统计出来的：")
# 分析一下这些分数是如何统计出来的
chain1 = generate_queries | retriever.map()
chain1_result = chain1.invoke({"original_query": original_query})

# 原始输出是一个二位列表，每个元素是由4个query生成的Document列表
print(chain1_result)

# 处理输出格式
contents = [doc.page_content
            for group in chain1_result  # 遍历外层列表
            for doc in group  # 遍历内层文档列表
            ]
print(contents)
