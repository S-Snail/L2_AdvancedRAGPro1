"""
    4、元数据索引
"""
from langchain_chroma import Chroma
from langchain_classic.chains.query_constructor.base import get_query_constructor_prompt, StructuredQueryOutputParser
from langchain_classic.chains.query_constructor.schema import AttributeInfo
from langchain_classic.retrievers import SelfQueryRetriever
from langchain_core.documents import Document
from six import integer_types

from models import get_ali_clients

# 获得访问大模型和嵌入模型的客户端
llm, embeddings_model = get_ali_clients()

# 加载文档
docs = [
    Document(
        page_content="作者A团队开发出基于人工智能的自动驾驶决策系统，在复杂路况下的响应速度提升300%",
        metadata={"year": 2024, "rating": 9.2, "genre": "AI", "author": "A"},
    ),
    Document(
        page_content="区块链技术成功应用于跨境贸易结算，作者B主导的项目实现交易确认时间从3天缩短至30分钟",
        metadata={"year": 2023, "rating": 9.8, "genre": "区块链", "author": "B"},
    ),
    Document(
        page_content="云计算平台实现量子计算模拟突破，作者C构建的新型混合云架构支持百万级并发计算",
        metadata={"year": 2022, "rating": 8.6, "genre": "云", "author": "C"},
    ),
    Document(
        page_content="大数据分析预测2024年全球经济趋势，作者A团队构建的模型准确率超92%",
        metadata={"year": 2023, "rating": 8.9, "genre": "大数据", "author": "A"},
    ),
    Document(
        page_content="人工智能病理诊断系统在胃癌筛查中达到三甲医院专家水平，作者B获医疗科技创新奖",
        metadata={"year": 2024, "rating": 7.1, "genre": "AI", "author": "B"},
    ),
    Document(
        page_content="基于区块链的数字身份认证系统落地20省市，作者C设计的新型加密协议通过国家级安全认证",
        metadata={"year": 2022, "rating": 8.7, "genre": "区块链", "author": "C"},
    ),
    Document(
        page_content="云计算资源调度算法重大突破，作者A研发的智能调度器使数据中心能效提升40%",
        metadata={"year": 2023, "rating": 8.5, "genre": "云", "author": "A"},
    ),
    Document(
        page_content="大数据驱动城市交通优化系统上线，作者B团队实现早晚高峰通行效率提升25%",
        metadata={"year": 2024, "rating": 7.4, "genre": "大数据", "author": "B"},
    )
]

vectorstore = Chroma.from_documents(docs, embedding=embeddings_model)

# 元数据字段定义（指导LLM如何解析查询条件）
metadata_field_info = [
    AttributeInfo(
        name="genre",
        description="文章的技术领域，选项:['AI '，'区块链'，'云'，'大数据']",
        type="string"
    ),
    AttributeInfo(
        name="year",
        description="文章的出版年份",
        type="integer"
    ),
    AttributeInfo(
        name="author",
        description="署名文章的作者姓名",
        type="string"
    ),
    AttributeInfo(
        name="rating",
        description="技术价值评估得分（1-10分）",
        type="float"
    )
]

# 文档内容描述(指导LLM理解文档内容)
document_content_description = "技术文章简述"

# 创建自查询检索器（核心组件）
"""
SelfQueryRetriever是Langchain库中的一个工具，其主要功能是把自然语言查询转变为结构化查询，以提升检索的准确度。
它整合了检索器和语言模型，能依据查询内容自动推断出筛选条件，还能识别出相关的元数据字段。
"""
retriever = SelfQueryRetriever.from_llm(
    llm=llm,
    vectorstore=vectorstore,
    document_contents=document_content_description,
    metadata_field_info=metadata_field_info,
    # enable_limit=True  # 限定返回结果，和query搭配使用
)

# 检索
print("-------------------------------- 评分在9分以上的文章 ----------------------------------------------")
# 查询条件：查询只约束分数rating > 9
print(retriever.invoke("我想了解评分在9分以上的文章"))
# print(retriever.invoke("我想了解评分在9分以上的文章，返回1篇文章"))  # 有限定：返回1篇文章

print("-------------------------------- 作者B在2023年发布的文章 ----------------------------------------------")
# 第二个查询只约束作者和年份
print(retriever.invoke("作者B在2023年发布的文章"))
exit()

# 了解
# # 原理：构建查询解析器（分析内部工作机制用）
# '''构建查询提示模板get_query_constructor_prompt：
#     document_content_description：对文档内容的概括性描述，例如 "有关各种主题的文章"。
#     metadata_field_info：元数据字段的详细描述，涵盖字段名称、类型以及描述。
# 此函数会生成一个提示模板，其用途是指导语言模型如何将自然语言查询转换为结构化查询。'''
# prompt = get_query_constructor_prompt(
#     document_content_description,
#     metadata_field_info
# )
# '''StructuredQueryOutputParser解析器的作用是：把语言模型的输出转换为 StructuredQuery 对象，
#     这个对象包含了 query（文本查询）和 filter（元数据筛选条件）
# '''
# output_parser = StructuredQueryOutputParser.from_components()
# # 链式操作，先将用户查询填入提示模板，接着由语言模型生成结构化输出，最后通过解析器得到结构化查询。
# query_constructor = prompt | llm | output_parser
# #
# # # 打印查询构造提示
# print("提示词：", prompt.format(query="我想了解评分在9分以上的文章"))
# print("提示词显示结束--------------------------提示词显示结束----------------------")
# #
# # 打印结构化查询的结果
# print("结构化查询结果：",query_constructor.invoke({"query": "作者B在2023年发布的文章"}))
# # 看工作原理-结束
