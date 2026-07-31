from langchain_deepseek import ChatDeepSeek
from langchain_community.tools.plugin import SerpAPIWrapper
from langchain_core.tools import Tool
from langgraph.prebuilt import create_react_agent

# 1. 初始化 DeepSeek 模型
llm = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0,
    max_tokens=2048,
    api_key="your-deepseek-api-key"  # 也可以通过环境变量 DEEPSEEK_API_KEY 配置
)

# 2. 加载 SerpAPI 搜索工具
# 需要提前安装：pip install google-search-results
# 并且设置环境变量：export SERPAPI_API_KEY="your-serpapi-key"
search = SerpAPIWrapper()
tools = [
    Tool(
        name="Search",
        func=search.run,
        description="用于在互联网上搜索最新信息、历史事件或实时数据的工具。"
    )
]

# 如果需要数学计算功能，可以直接自定义 Python 工具或使用内置 LangChain 工具：
# from langchain_community.tools import PythonREPLTool
# tools.append(PythonREPLTool())

# 3. 使用现代 LangGraph prebuilt 创建 ReAct Agent
agent_executor = create_react_agent(llm, tools)

# 4. 运行 agent
query = "What's the date today? What great events have taken place today in history?"
response = agent_executor.invoke({"messages": [("user", query)]})

# 5. 打印最终回答
print(response["messages"][-1].content)