import asyncio
import os
import sys
import warnings
from typing import TypedDict
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.syntax import Syntax
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END
from langchain_mcp_adapters.client import MultiServerMCPClient
import phoenix as px
from openinference.instrumentation.langchain import LangChainInstrumentor
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

load_dotenv()
console = Console()
warnings.filterwarnings("ignore")

REQUIRED = ["SQLCL_BIN", "TAVILY_API_KEY", "DB_APP_USER", "DB_APP_PASSWORD", "DB_DSN"]
if missing := [v for v in REQUIRED if not os.getenv(v)]:
    console.print(f"Missing .env vars: {', '.join(missing)}")
    sys.exit(1)

SQLCL_BIN = os.getenv("SQLCL_BIN")

if "PHOENIX_COLLECTOR_ENDPOINT" in os.environ:
    del os.environ["PHOENIX_COLLECTOR_ENDPOINT"]

try:
    session = px.launch_app()
    tracer_provider = TracerProvider()
    span_exporter = OTLPSpanExporter(endpoint="http://127.0.0.1:6006/v1/traces")
    tracer_provider.add_span_processor(BatchSpanProcessor(span_exporter))
    trace.set_tracer_provider(tracer_provider)
    LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
except Exception as e:
    console.print(f"[yellow]⚠️ Observability skipped: {e}[/yellow]")

llm = ChatOllama(model=os.getenv("OLLAMA_MODEL", "gemma2:9b"), temperature=0)

mcp_client = MultiServerMCPClient(
    {
        "oracle": {
            "transport": "stdio",
            "command": SQLCL_BIN,
            "args": ["-nolog", "-mcp"],
            "env": os.environ.copy()
        }
    }
)

class AgentState(TypedDict):
    request: str
    research_data: str
    db_data: str
    final_report: str

async def web_search_node(state: AgentState):
    console.print(f"\nWeb Agent: Researching via Tavily...")
    
    query_prompt = f"Provide 2-3 actors suitable for: {state['request']}. Return names only."
    search_query = await (llm | StrOutputParser()).ainvoke(query_prompt)
    
    try:
        tavily = TavilySearch(max_results=2, topic="general")
        result = await asyncio.to_thread(tavily.invoke, {"query": search_query})
        
        if isinstance(result, dict) and 'results' in result:
             context = "\n".join([f"- {r.get('content', '')}" for r in result['results']])
        else:
             context = str(result)
    except Exception as e:
        context = f"Search Error: {e}"
    
    return {"research_data": context}

async def db_mcp_node(state: AgentState):
    console.print(f"\nDB Agent: Connecting via Adapter...")
    
    # Generate SQL
    sql_prompt = f"""
    You are an Oracle SQL generator. 
    Table: talent_roster (actor_name, availability_status, min_fee_usd)
    Search Context: {state['research_data']}
    
    Task: Write a single SQL query to select actor_name and availability_status from talent_roster 
    where availability_status = 'AVAILABLE'.
    
    Output ONLY the raw SQL string. No formatting.
    """
    sql_raw = await (llm | StrOutputParser()).ainvoke(sql_prompt)
    clean_sql = sql_raw.strip().replace("```sql", "").replace("```", "").rstrip(';')
    
    console.print(Panel(Syntax(clean_sql, "sql", theme="monokai"), title="⚡ MCP: Sending to Oracle", border_style="yellow"))

    try:
        async with mcp_client.session("oracle") as session:
            await session.initialize()
            
            tools = await session.list_tools()
            tool_names = [t.name for t in tools.tools]
            
            if "run-sqlcl" in tool_names:
                use_tool = "run-sqlcl"
                arg_key = "sqlcl" 
            elif "run-sql" in tool_names:
                use_tool = "run-sql"
                arg_key = "sql"
            else:
                use_tool = "sql"
                arg_key = "sql"

            db_user = os.getenv("DB_APP_USER")
            db_pass = os.getenv("DB_APP_PASSWORD")
            db_dsn = os.getenv("DB_DSN")

            full_script = f"""
            connect {db_user}/{db_pass}@{db_dsn};
            {clean_sql};
            """
            
            console.print(f"[dim]→ Executing via {use_tool} (arg: {arg_key})...[/dim]")
            
            result = await session.call_tool(
                use_tool,
                arguments={arg_key: full_script}
            )
            
            output = ""
            if hasattr(result, 'content'):
                for item in result.content:
                    if hasattr(item, 'text'):
                        output += item.text + "\n"
            
            final_output = output.strip() if output.strip() else "Success: Query executed (Silent output)."
            
            if "ORA-" in final_output or "SP2-" in final_output:
                console.print(Panel(final_output, title="⚠️  Query Error", border_style="red"))
            else:
                console.print(Panel(final_output, title="💾 MCP Result", border_style="green"))
            
            return {"db_data": final_output}

    except Exception as e:
        console.print(f"[bold red]💥 MCP Error:[/bold red] {e}")
        return {"db_data": f"Error: {e}"}

async def orchestrator_node(state: AgentState):
    console.print(f"\nOrchestrator: Finalizing Report...")
    
    report_prompt = f"""
    Context: {state['research_data']}
    Database Results: {state['db_data']}
    
    Task: Create a Casting Recommendation for: '{state['request']}'.
    Logic: Recommend an AVAILABLE actor from the database results that fits the context.
    """
    report = await (llm | StrOutputParser()).ainvoke(report_prompt)
    return {"final_report": report}

workflow = StateGraph(AgentState)
workflow.add_node("web_search", web_search_node)
workflow.add_node("db_mcp", db_mcp_node)
workflow.add_node("orchestrator", orchestrator_node)

workflow.set_entry_point("web_search")
workflow.add_edge("web_search", "db_mcp")
workflow.add_edge("db_mcp", "orchestrator")
workflow.add_edge("orchestrator", END)

app = workflow.compile()

async def main():
    console.clear()
    console.print(Panel("🎬 TALENT SCOUT AI", style="bold blue"))
    
    user_request = "We need a dramatic actor for a Cyberpunk thriller."
    
    try:
        final_state = await app.ainvoke({"request": user_request})
        console.print("\n")
        console.print(Panel(Markdown(final_state['final_report']), title="✅ Recommendation", border_style="green"))
    except Exception as e:
        console.print(f"Critical Failure: {e}")

    try:
        tracer_provider.force_flush()
    except:
        pass
    input("Press Enter to exit...")

if __name__ == "__main__":
    asyncio.run(main())