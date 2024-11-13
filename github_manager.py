import requests
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnableConditional, RunnableLambda
from langchain.tools import ShellRun
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import GitHubRepoLoader
from langchain_community.chat_models import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser

# LM-Studio Local Integration
class LMStudioAPI:
    def __init__(self, base_url="http://localhost:5000"):
        self.base_url = base_url

    def query(self, prompt):
        response = requests.post(f"{self.base_url}/generate", json={"prompt": prompt})
        return response.json().get("output", "")

lm_studio = LMStudioAPI()

# Initialize Memory
memory = ConversationBufferMemory(memory_key="chat_history")
summary_memory = ConversationSummaryMemory()

# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)

# Load GitHub Repository and Create Retriever
repo_url = "https://github.com/your-repo"
loader = GitHubRepoLoader(repo_url)
documents = loader.load_and_split()
vectorstore = FAISS.from_documents(documents, OpenAIEmbeddings())
retriever = vectorstore.as_retriever()

# Tools and Agent Initialization
tools = [ShellRun()]
agent = create_openai_functions_agent(llm=llm, tools=tools, prompt="Enhanced GitHub Manager")
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Task Agents
def handle_pull_requests():
    return executor.invoke({"task": "manage PRs"})

def handle_ci_cd():
    return executor.invoke({"task": "trigger CI/CD pipelines"})

def handle_issues():
    return executor.invoke({"task": "manage issues"})

collaboration_chain = RunnableParallel(
    pr_agent=RunnableLambda(handle_pull_requests),
    ci_agent=RunnableLambda(handle_ci_cd),
    issue_agent=RunnableLambda(handle_issues),
)

# RAG Chain
prompt_template = ChatPromptTemplate.from_template("""
Use the following context to respond:
{context}

Question: {question}
""")
rag_chain = retriever | prompt_template | llm

# Task Prioritization
priority_logic = RunnableConditional(
    condition=lambda x: "urgent" in x["task_metadata"],
    if_true=lambda x: executor.invoke({"task": x["task_name"]}),
    if_false=lambda x: print("Task deferred.")
)

# CI/CD Trigger
def trigger_pipeline(pipeline_name):
    shell_tool = ShellRun()
    command = f"bash {pipeline_name}"
    return shell_tool.invoke({"command": command})

# Safe Execution with Fallbacks
def safe_execution(task):
    try:
        return executor.invoke(task)
    except Exception as e:
        print(f"Error encountered: {str(e)}")
        return {"error": str(e)}

safe_chain = RunnableLambda(safe_execution)

# JSON Output Parser
parser = JsonOutputParser(schema={
    "pull_requests": {
        "type": "array",
        "items": {"type": "object", "properties": {"id": {"type": "string"}, "status": {"type": "string"}}}
    }
})
chain_with_parser = executor | parser

# Main Function
def main():
    print("Starting GitHub Management System...")

    # Task Prioritization Example
    task = {"task_name": "merge PR", "task_metadata": "urgent"}
    print("Task Prioritization:")
    priority_logic.invoke(task)

    # RAG Example
    print("\nRAG Chain Example:")
    rag_response = rag_chain.invoke({"question": "What is the CI/CD status?"})
    print("RAG Response:", rag_response)

    # Multi-Agent Collaboration Example
    print("\nMulti-Agent Collaboration:")
    collaboration_result = collaboration_chain.invoke({"repo": "your-repo"})
    print("Collaboration Result:", collaboration_result)

    # CI/CD Pipeline Example
    print("\nTrigger CI/CD Pipeline:")
    pipeline_result = trigger_pipeline("deploy.sh")
    print("Pipeline Trigger Result:", pipeline_result)

    # Safe Execution Example
    print("\nSafe Execution Example:")
    safe_execution_result = safe_chain.invoke({"task": "invalid task"})
    print("Safe Execution Result:", safe_execution_result)

if __name__ == "__main__":
    main()
