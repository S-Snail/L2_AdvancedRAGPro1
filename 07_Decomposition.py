"""
    查询优化 --- 问题分解
"""
from typing import List

from langchain_chroma import Chroma
from langchain_classic.retrievers.multi_query import LineListOutputParser
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import PromptTemplate, BasePromptTemplate, ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable, RunnableLambda

from models import get_ali_clients

# 获得访问大模型和嵌入模型客户端
llm, embedding_model = get_ali_clients()


# 格式化输出
def pretty_print_docs(docs):
    """
    将文档列表格式化为可读性强的字符串
    :param docs:
    :return:
    """
    print(
        f"\n{'-' * 100}".join(
            [f"Document {i + 1}:\n\n" + d.page_content for i, d in enumerate(docs)]
        )
    )


# 创建示例文档数据集 - 番茄炒蛋制作指南
documents = [
    Document(
        page_content="番茄炒蛋的食材：\n\n- 新鲜鸡蛋：3-4个（根据人数调整）\n- 番茄：2-3个中等大小\n- 盐：适量\n- 白糖：一小勺（可选，用于提鲜）\n- 食用油：适量\n- 葱花：少许（可选，用于增香）\n\n这些是最基本的材料，当然也可以根据个人口味添加其他调料或配料。"),
    Document(
        page_content="番茄炒蛋的步骤：鸡蛋打入碗中，加入少许盐，用筷子或打蛋器充分搅拌均匀；\n   - 番茄洗净后切成小块备用。\n\n3. **炒鸡蛋**：锅内倒入适量食用油加热至温热状态，然后将搅拌好的鸡蛋液缓缓倒入锅中。待鸡蛋凝固时轻轻翻动几下，让其受热均匀直至完全熟透，随后盛出备用。\n\n4. **炒番茄**：在同一锅里留下的底油中放入切好的番茄块，中小火慢慢翻炒至出汁，可根据个人口味加一点点白糖提鲜。\n\n5. **合炒**：当番茄炒至软烂并开始释放大量汤汁时，再把之前炒好的鸡蛋倒回锅里，快速与番茄混合均匀，同时加入适量的盐调味。如果喜欢的话还可以撒上一些葱花增加香气。\n\n6. **完成**：最后检查一下味道是否合适，确认无误后即可关火装盘享用美味的番茄炒蛋啦！"),
    Document(
        page_content="技巧与注意事项：1. **选材**：选择新鲜的鸡蛋和成熟的番茄。新鲜的食材是做好这道菜的基础。\n2. **打蛋液**：将鸡蛋打入碗中后加入少许盐（根据个人口味调整），然后充分搅拌均匀。这样做可以让蛋更加松软且味道更佳。\n3. **处理番茄**：番茄最好先用开水稍微焯一下皮，然后去皮切块。这样可以去除表皮的硬质部分，让番茄更容易入味，并且口感更好。\n4. **热锅冷油**：先用中小火把锅烧热，再倒入适量食用油，待油温五成热时下蛋液。这样的做法可以使蛋快速凝固形成漂亮的形状而不易粘锅。\n5. **分步烹饪**：通常建议先炒鸡蛋至半熟状态取出备用；接着利用剩下的底油继续翻炒番茄至出汁，最后再将之前炒好的鸡蛋倒回锅里与番茄混合均匀加热即可。\n6. **调味品**：除了基本的盐之外，还可以根据喜好添加少量糖来提鲜或者一点酱油增色添香。注意调味料不宜过多以免掩盖了食材本身的味道。\n7. **出锅前加葱花**：如果喜欢的话，在即将完成时撒上一些葱花不仅能增加菜品色泽还能增添香气。")
]
# 创建Chroma向量存储并添加文档
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embedding_model,
    collection_name="decomposition"
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

print("-------------检索到的文档（拆解前）--------------")
# 从实际来说，该问题的答案应该包括原材料、步骤，以及注意事项才算完整
pretty_print_docs(retriever.invoke("新手如何制作番茄炒蛋？"))

print("开始问题拆解的处理=======================>")
# 子问题生成提示模板
template = """你是一名AI语言模型助理。你的任务是将输入的问题分解成3个子问题，通过一个个解决这些子问题从而解决完整的问题。
            子问题需要在矢量数据库中检索相关文档。通过分解用户问题生成子问题，你的目标是帮助用户克服基于距离的相似性搜索的一些局限性。
            请提供这些用换行符分隔子问题本身，不需要额外内容。
            原始问题：{question}"""
DEFAULT_QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template=template
)

# 子问题回答提示模板
DEFAULT_SUB_QUESTION_PROMPT = PromptTemplate(
    input_variables=["question", "sub_question", "documents"],
    template="""要解决主要问题{question}，需要先解决子问题{sub_question}。
    以下是为支持您的推理而提供的参考文档：{documents}。请直接给出当前子问题的答案。不需要额外内容"""
)


# 自定义一个检索器，将对子问题的生成、获得子问题的答案组合起来，通过组合简化使用过程
# 继承自BaseRetriever
class DecompositionQueryRetriever(BaseRetriever):
    # 定义3个类属性
    # 向量数据库检索器
    retriever: BaseRetriever  # 基础检索器，用于检索与子问题相关的文档
    # 生成子问题链
    make_sub_chain: Runnable  # 一个Runnable，用于生成子问题
    # 解决子问题链 一个Runnable，用于解决子问题（即根据检索到的文档生成答案）
    resolve_sub_chain: Runnable

    # 构造函数
    @classmethod  # 类方法装饰器
    # self 表示示例，cls 表示类
    def from_llm(
            cls,
            retriever: BaseRetriever,  # 基础检索器
            llm: BaseLanguageModel,  # 语言模型
            prompt: BasePromptTemplate = DEFAULT_QUERY_PROMPT,  # 子问题生成提示
            sub_prompt: BasePromptTemplate = DEFAULT_SUB_QUESTION_PROMPT  # 子问题回答提示
    ) -> "DecompositionQueryRetriever":
        """类方法：通过LLM创建分解检索器实例"""
        output_parser = LineListOutputParser()  # 多行输出解析器
        return cls(
            retriever=retriever,
            make_sub_chain=prompt | llm | output_parser,  # 构建子问题生成链：提示模板 -> 语言模型 -> 输出解析器
            resolve_sub_chain=sub_prompt | llm  # 构架子问题解答链：提示模板 -> 语言模型
        )

    # 生成子问题，使用LLM将用户问题分解为子问题
    def generate_queries(self, question: str) -> List[str]:
        """
        输入一个用户问题，使用make_sub_chain生成子问题列表。
        它调用链并返回一个字符串列表，每个字符串是一个子问题。
        """
        response = self.make_sub_chain.invoke({"question": question})
        lines = response  # 获取解析后的子问题列表
        print(f"生成子问题：{lines}")
        return lines

    # 获得子问题答案 并行处理子问题，检索+生成答案
    def retrieve_documents(self, query: str, sub_queries: List[str]) -> List[Document]:
        """
        输入原始查询(query)和子问题列表(sub_queries: List[str])，对每个子问题并行处理：
        使用retriever检索与子问题相关的文档。
        使用resolve_sub_chain根据检索到的文档生成该子问题的答案。
        然后，将每个子问题和其答案组合成一个Document对象（页面内容为子问题加上答案）。
        最后返回这些文档列表。
        """

        def process_sub_query(sub_query):
            # 检索与子问题相关的文档
            docs = self.retriever.invoke(sub_query)
            # 构建子问题解答提示并获取答案
            return self.resolve_sub_chain.invoke({
                "question": query,  # 原始问题
                "sub_question": sub_query,  # 当前子问题
                "documents": [doc.page_content for doc in docs]  # 检索到的文档内容
            })

        # 创建可行的处理链
        sub_llm_chain = RunnableLambda(process_sub_query)
        # 批量执行所有的子问题
        responses = sub_llm_chain.batch(sub_queries)
        # 将子问题和答案合并作为解决主问题的文档(结果文档)
        documents = [
            Document(page_content=sub_query + "\n" + response.content)
            for sub_query, response in zip(sub_queries, responses)
        ]
        return documents

    # 主入口，协调整个分解流程
    def _get_relevant_documents(
            self,
            query: str,
            *,
            run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        # 生成子问题列表
        sub_queries = self.generate_queries(query)
        # 将子问题列表传递给retrieve_documents 解决子问题并构建结果文档
        documents = self.retrieve_documents(query, sub_queries)
        return documents


# ====================== 主执行流程 ====================== #
print("使用DecompositionQueryRetriever来分解问题------------>")
# 创建问题分解器实例
decompositionQueryRetriever = DecompositionQueryRetriever.from_llm(llm=llm, retriever=retriever)
# 执行问题分解解锁
decomposition_docs = decompositionQueryRetriever.invoke("番茄炒蛋怎么制作？")
print("-------------检索到的文档（拆解后）--------------")
pretty_print_docs(decomposition_docs)

# 创建prompt模板
template = """请根据以下文档回答问题：
### 文档：
{context}
### 问题：
{question}
"""

# 由模板生成prompt
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | llm
print("-------------回答--------------")
question = "新手如何制作番茄炒蛋？"
response = chain.invoke({
    "context": [doc.page_content for doc in decomposition_docs],
    "question": question
})
print(response.content)
