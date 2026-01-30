import os
import requests
from typing import TypedDict , List , Union , Annotated 
from operator import add
from langchain_openai import ChatOpenAI
from langchain_core.messages import (SystemMessage , HumanMessage , AIMessage , ToolMessage , AnyMessage)
from langgraph.graph import StateGraph , START , END
from langgraph.prebuilt import ToolNode , tools_condition
from langchain_core.tools import tool
import json
from datetime import datetime
from tavily import TavilyClient




OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY is not set. Set in the OS PATH"
    )


TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not TAVILY_API_KEY:
    raise RuntimeError("TAVILY_API_KEY is not set in the OS PATH")

tavily = TavilyClient(api_key=TAVILY_API_KEY)

system_message = """ You are a helpful corporate customer service agent.\n
Answer queries about jobs, policies, news, and log grievances when needed\n. 
Use the available tools to help users\n.

IMPORTANT :

-First ask the user it's name and then continue the conversation , Something like "Hi,Looking forward to our conversation,May I please know your name before we begin\n"
-NOTE: Not compulsory , you say the exact same sentence , reason it however you prefer.\n
-NOTE: Ask for name only once in the thread , ask name again only if you are in a different thread.\n
-Analyze the queries and the files you have access to properly , You should be able to interpret even short forms\n
-Also give extra information if needed for the employees as it goes as the info in JSON is quite limited.\n
-Refuse to answer for anything outside the context.
"""


class EmailAgent(TypedDict):
    messages : Annotated[list[Union[AnyMessage]] , add]


llm = ChatOpenAI(model = "gpt-4o-mini" , temperature=0 , api_key=OPENAI_API_KEY)


@tool
def web_search(query: str, max_results: int = 5) -> str:
    """
    Search the web using Tavily for job vacancies or oppurtunities outside the company.
    Use for current industry trends in the IT or CSE Sector.
    """

    print(f"[TOOL:WEB_SEARCH] Searching web for: {query}")

    response = tavily.search(
        query=query,
        max_results=max_results,
        include_answer=True,
        include_sources=True
    )

    return json.dumps(response, indent=2)


@tool
def job_oppurtunity(title : str = " ") -> str:

    """Search for Job Oppurtunities for the user within the company"""

    print(f"[TOOL:JOB_OPPURTUNITY] , Looking for roles in : {title} ")

    with open("job_vacancy.json","r") as f:
        jobs = json.load(f)

    results = [job for job in jobs if title.lower() in job['title'].lower()]

    if results:
        return f"Found a job of your interest i.e {results}"
    
    else:
        return f"No job found of your interest"


@tool
def company_policy(query : str) -> str:
    """Provide Company Policy to people seeking out oppurtunities"""

    print(f"[TOOL:COMPANY_POLICY] , Looking for {query} to be followed in the company: ")

    with open("company_policy.json", "r") as f:
        policies = json.load(f)

    return json.dumps(policies , indent = 2)
    

@tool
def company_news(query : str) -> str:
    """Provide latest news about the company's activities"""

    print(f"[TOOLS:COMPANY NEWS] , Looking for the latest {query} trends happening in the company")

    with open("company_news.json") as f:
        news = json.load(f)

    return json.dumps(news , indent = 2)


@tool
def log_grievances(complaint_by : str ,complaint_against: str , issue : str) -> str:
    """Log a complaint about a fellow colleague when prompted by the user"""

    try:
        with open("log_grievances.json", "r") as f:
            complaints = json.load(f)
    except FileNotFoundError:
        complaints = []

    new_complaint = {
        "id" : len(complaints) + 1,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Complaint_By": complaint_by,
        "Complaint_Against": complaint_against,
        "issue":issue,
        "status":"open"
    }

    complaints.append(new_complaint)

    with open("log_grievances.json","w") as w:
        json.dump(complaints , w , indent=2)

    return f"Complaint logged successfully"


tools = [job_oppurtunity , company_policy , company_news , log_grievances , web_search]


def input_node(state: EmailAgent):
    return {
        "messages" : state["messages"]
    }


def agent_node(state : EmailAgent):

    messages = state["messages"]

    messages_system = [SystemMessage(content = system_message )] + messages

    llm_with_tools = llm.bind_tools(tools)

    response = llm_with_tools.invoke(messages_system)

    return {"messages": [response]}


def should_continue(state : EmailAgent):

    """Determine weather the agent should go to Tools or Not"""

    last_message = state["messages"][-1]

    if hasattr(last_message,"tool_calls") and last_message.tool_calls:
        return "tools"
    else:
        return END


Graph = StateGraph(EmailAgent)

# Graph.add_node("input",input_node)
Graph.add_node("Agent",agent_node)
Graph.add_node("tools",ToolNode(tools))

Graph.add_edge(START , "Agent")
# Graph.add_edge("input","Agent")
Graph.add_conditional_edges(
    "Agent",
    should_continue,
    {
        "tools":"tools",
        END:END
    }
)

Graph.add_edge("tools","Agent")

app = Graph.compile()


def main():

    state : EmailAgent = {
        "messages":[]
    }

    name = input("Please enter your name: ")

    while True:
        user_input = input("Enter your queries: ")
        if user_input == "exit":
            break

        state["messages"].append(HumanMessage(content = user_input))
        state = app.invoke(state)

        last_message = state["messages"][-1]
        if isinstance(last_message,AIMessage):
            print("\nAI:",last_message.content)
        else:
            print("\nAI","Unable to generate a response")


if __name__ == "__main__":
    main()
