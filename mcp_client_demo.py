#!/usr/bin/env python3
"""
Crawl4ai Reference Application - MCP Client Demo

This script demonstrates how to use the MCP client directly with Crawl4ai.
"""

import asyncio
import json
import os
import sys
from typing import Dict, List, Optional

import click
from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax

# Load environment variables from .env file
load_dotenv()

# Initialize rich console for pretty output
console = Console()

# Default MCP server configuration
DEFAULT_MCP_PORT = int(os.getenv("CRAWL4AI_MCP_PORT", "11235"))
DEFAULT_MCP_HOST = os.getenv("CRAWL4AI_MCP_HOST", "localhost")


async def call_mcp_tool(host: str, port: int, tool_name: str, arguments: Dict) -> Dict:
    """
    Call an MCP tool directly using httpx.
    
    Args:
        host: The MCP server host
        port: The MCP server port
        tool_name: The name of the tool to call
        arguments: The arguments to pass to the tool
        
    Returns:
        Dict: The result of the tool call
    """
    import httpx
    
    # Construct the URL
    url = f"http://{host}:{port}/mcp/tools/call"
    
    # Construct the payload
    payload = {
        "name": tool_name,
        "arguments": json.dumps(arguments)
    }
    
    # Make the request
    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=payload)
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"MCP tool call failed with status code: {response.status_code}\n{response.text}")


async def list_mcp_tools(host: str, port: int) -> List[Dict]:
    """
    List all available MCP tools.
    
    Args:
        host: The MCP server host
        port: The MCP server port
        
    Returns:
        List[Dict]: A list of available tools
    """
    import httpx
    
    # Construct the URL
    url = f"http://{host}:{port}/mcp/tools/list"
    
    # Make the request
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Failed to list MCP tools with status code: {response.status_code}\n{response.text}")


async def check_mcp_server(host: str, port: int) -> bool:
    """
    Check if the MCP server is running.
    
    Args:
        host: The MCP server host
        port: The MCP server port
        
    Returns:
        bool: True if the server is running, False otherwise
    """
    import httpx
    
    # Construct the URL
    url = f"http://{host}:{port}/health"
    
    # Make the request
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=2.0)
            return response.status_code == 200
    except:
        return False


@click.group()
def cli():
    """Crawl4ai Reference Application - MCP Client Demo"""
    pass


@cli.command("list")
@click.option("--host", default=DEFAULT_MCP_HOST, help="MCP server host")
@click.option("--port", "-p", default=DEFAULT_MCP_PORT, help="MCP server port")
def list_cmd(host: str, port: int):
    """List all available MCP tools"""
    console.print(f"[bold blue]Listing MCP tools from server at {host}:{port}[/]")
    
    # Check if the server is running
    if not asyncio.run(check_mcp_server(host, port)):
        console.print(f"[bold red]MCP server at {host}:{port} is not running[/]")
        console.print(f"[bold yellow]Start the server with:[/] python mcp_demo.py start --port {port}")
        return
    
    # List the tools
    try:
        tools = asyncio.run(list_mcp_tools(host, port))
        
        if not tools:
            console.print("[bold yellow]No MCP tools available[/]")
            return
        
        console.print(f"[bold green]Found {len(tools)} MCP tools:[/]")
        
        # Display each tool
        for i, tool in enumerate(tools):
            console.print(f"\n[bold]{i+1}. {tool.get('name', 'Unknown')}[/]")
            console.print(f"Description: {tool.get('description', 'No description')}")
            
            # Display input schema if available
            if "inputSchema" in tool:
                console.print("\nInput Schema:")
                schema_str = json.dumps(tool["inputSchema"], indent=2)
                console.print(Syntax(schema_str, "json", theme="monokai", line_numbers=True))
    
    except Exception as e:
        console.print(f"[bold red]Error listing MCP tools:[/] {str(e)}")


@cli.command("crawl")
@click.argument("url", required=True)
@click.option("--host", default=DEFAULT_MCP_HOST, help="MCP server host")
@click.option("--port", "-p", default=DEFAULT_MCP_PORT, help="MCP server port")
@click.option("--output", "-o", type=click.Choice(["markdown", "json"]), default="markdown", 
              help="Output format (markdown or json)")
def crawl_cmd(url: str, host: str, port: int, output: str):
    """Crawl a website using the MCP server"""
    console.print(f"[bold blue]Crawling {url} using MCP server at {host}:{port}[/]")
    
    # Check if the server is running
    if not asyncio.run(check_mcp_server(host, port)):
        console.print(f"[bold red]MCP server at {host}:{port} is not running[/]")
        console.print(f"[bold yellow]Start the server with:[/] python mcp_demo.py start --port {port}")
        return
    
    # Prepare arguments
    arguments = {
        "url": url,
        "output_format": output,
        "scan_full_page": True,
        "headless": True
    }
    
    # Call the tool
    try:
        result = asyncio.run(call_mcp_tool(host, port, "crawl_url", arguments))
        
        console.print("[bold green]Crawl successful![/]")
        
        # Display the result
        if "content" in result and result["content"]:
            content = result["content"][0].get("text", "")
            
            if output == "markdown":
                console.print(Panel(Markdown(content), title=f"Content from {url}", expand=False))
            else:
                try:
                    # Try to parse as JSON
                    json_content = json.loads(content)
                    console.print(Panel(json.dumps(json_content, indent=2), 
                                       title=f"JSON from {url}", expand=False))
                except:
                    # If not valid JSON, just print as text
                    console.print(Panel(content, title=f"Content from {url}", expand=False))
        else:
            console.print("[yellow]No content returned[/]")
    
    except Exception as e:
        console.print(f"[bold red]Error calling MCP tool:[/] {str(e)}")


@cli.command("extract")
@click.argument("url", required=True)
@click.argument("instruction", required=False, default="Extract main content and metadata as structured JSON")
@click.option("--host", default=DEFAULT_MCP_HOST, help="MCP server host")
@click.option("--port", "-p", default=DEFAULT_MCP_PORT, help="MCP server port")
def extract_cmd(url: str, instruction: str, host: str, port: int):
    """Extract structured data from a website using the MCP server"""
    console.print(f"[bold blue]Extracting data from {url} using MCP server at {host}:{port}[/]")
    console.print(f"[bold blue]Instruction:[/] {instruction}")
    
    # Check if the server is running
    if not asyncio.run(check_mcp_server(host, port)):
        console.print(f"[bold red]MCP server at {host}:{port} is not running[/]")
        console.print(f"[bold yellow]Start the server with:[/] python mcp_demo.py start --port {port}")
        return
    
    # Prepare arguments
    arguments = {
        "url": url,
        "instruction": instruction,
        "output_format": "json",
        "scan_full_page": True,
        "headless": True
    }
    
    # Call the tool
    try:
        result = asyncio.run(call_mcp_tool(host, port, "extract_data", arguments))
        
        console.print("[bold green]Extraction successful![/]")
        
        # Display the result
        if "content" in result and result["content"]:
            content = result["content"][0].get("text", "")
            
            try:
                # Try to parse as JSON
                json_content = json.loads(content)
                console.print(Panel(json.dumps(json_content, indent=2), 
                                   title="Extracted Data", expand=False))
            except:
                # If not valid JSON, just print as text
                console.print(Panel(content, title="Extracted Data", expand=False))
        else:
            console.print("[yellow]No content returned[/]")
    
    except Exception as e:
        console.print(f"[bold red]Error calling MCP tool:[/] {str(e)}")


if __name__ == "__main__":
    try:
        cli()
    except Exception as e:
        console.print(f"[bold red]Error:[/] {str(e)}")
        sys.exit(1) 