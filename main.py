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
from langchain_ollama import ChatOllama, OllamaEmbeddings
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
DB_USER = os.getenv("DB_APP_USER")
DB_PASS = os.getenv("DB_APP_PASSWORD")
DB_DSN = os.getenv("DB_DSN")

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
    console.print(f"Observability skipped: {e}")

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
    
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    
    # 1. Embed the User Request (Semantic Search)
    console.print(f"[dim]   -> Embedding query with nomic-embed-text...[/dim]")
    query_vector = await asyncio.to_thread(embeddings.embed_query, state['request'])
    vector_str = str(query_vector)
    
    # 2. Vector Search SQL (Chunks for ORA-01704 fix)
    # We split the large vector string into 2000-char chunks to avoid string literal limits
    chunk_size = 2000
    chunks = [vector_str[i:i+chunk_size] for i in range(0, len(vector_str), chunk_size)]
    
    plsql_build_clob = ""
    for chunk in chunks:
        plsql_build_clob += f"    v_clob := v_clob || '{chunk}';\n"

    clean_sql = f"""
    DECLARE
        v_clob CLOB;
        v_vec VECTOR;
        v_res SYS_REFCURSOR;
    BEGIN
        v_clob := TO_CLOB('');
{plsql_build_clob}
        -- Cast to FLOAT64 to match standard oracledb insertions (array('d'))
        v_vec := VECTOR(v_clob, *, FLOAT64);
        
        OPEN v_res FOR
            SELECT actor_name, availability_status, bio
            FROM talent_roster
            WHERE availability_status = 'AVAILABLE'
            ORDER BY VECTOR_DISTANCE(param_vector, v_vec, COSINE)
            FETCH FIRST 3 ROWS ONLY;
            
        DBMS_SQL.RETURN_RESULT(v_res);
    END;
    /
    """
    
    console.print(Panel(Syntax(clean_sql[:300] + "... (truncated PL/SQL)", "sql", theme="monokai"), title="⚡ MCP: Sending Vector Query (PL/SQL)", border_style="yellow"))

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

            # Note: Credentials now handled in MultiServerMCPClient init
            # Note: For PL/SQL via SQLcl, ensures "/" is on a new line
            full_script = f"""
            connect {DB_USER}/{DB_PASS}@{DB_DSN};
            SET SERVEROUTPUT ON;
            {clean_sql}
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
    
    # user_request = "We need a dramatic actor for a Cyberpunk thriller."
    try:
        user_request = console.input("[bold yellow]Enter your casting request:[/bold yellow] ")
    except KeyboardInterrupt:
        return

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