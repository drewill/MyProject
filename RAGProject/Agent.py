from pathlib import Path

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
import os
from rich import print as rprint
from langchain_core.tools import tool

from RAGSystem import build_RAG_graph,retrieval_docs

load_dotenv(override=True)
key = os.getenv("DEEPSEEK_FLASH_KEY")
url = os.getenv("DEEPSEEK_FLASH_URL")

vectorstore = build_RAG_graph()


@tool
def get_weather(city: str):
    """
    这是一个查询天气的工具

    args:
        city: 城市的名称
    """
    return f"{city}天气很差，雾霾很重！"

@tool(parse_docstring=True)
def local_RAG_knowledge(query: str)->str :
    """
    搜索本地知识库中的 PDF、论文和技术文档。
    当用户的问题需要依据知识库资料回答时使用该工具。

    Args:
        query: 这是用户提出的问题

    """
    out = retrieval_docs(vectorstore,query)
    if not out:
        print("知识库中没有检索到相关内容。")

    results = []
    for doc in out:
        source = Path(doc.metadata.get('source','unkown')).name
        page = doc.metadata.get('page',0)

        results.append(
            f"资料{page}页，来源:{source},内容：\n{doc.page_content} \n"
        )

    return results


prompt = """
        你是一个知识库智能助手。
        
        当用户的问题涉及论文、PDF、技术资料或项目文档时，
        优先调用 search_knowledge_base 搜索知识库。
        
        回答必须优先依据检索结果。
        如果知识库没有相关内容，应明确说明。
        回答时尽量注明来源文件和页码。
        
        对于普通闲聊或常识等和内部数据库无关的问题，可以拒绝直接回答，
        
        """
prompt1 = """
You are a knowledge base assistant.

When answering questions based on the knowledge base,
always answer in English.

Use the retrieved documents as the factual basis for your answer.
Do not translate the answer into Chinese.
"""

model = init_chat_model(
    model = "deepseek-flash",
    model_provider="openai",
    api_key = key,
    base_url = url,
)

agent = create_agent(
    model=model,
    tools=[get_weather, local_RAG_knowledge],
    system_prompt=prompt
)
# response = agent.invoke({
#     "messages":[
#         {"role":"system","content":"你是一个天气查询助手"},
#         {"role":"user","content":"北京的天气今天怎么样？"}
#     ]
# })
response = agent.invoke({
    "messages":[
        {"role":"user","content":"今天纽约的天气怎么样"}
    ]
})
rprint(
    response
)