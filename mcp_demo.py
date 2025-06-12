#!/usr/bin/env python3
"""
Crawl4ai Reference Application - MCP Server Demo

This script demonstrates how to use Crawl4ai with the Model Context Protocol (MCP).
MCP allows AI models to call external tools, including web crawling and data extraction.
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

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CrawlResult

# Load environment variables from .env file
load_dotenv()

# Initialize rich console for pretty output
console = Console()

# Default configuration
DEFAULT_BROWSER_CONFIG = {
    "headless": os.getenv("CRAWL4AI_HEADLESS", "true").lower() == "true",
    "viewport_width": 1280,
    "viewport_height": 800,
    "user_agent_mode": "random",
    "ignore_https_errors": True,
}

DEFAULT_CRAWLER_CONFIG = {
    "verbose": os.getenv("CRAWL4AI_VERBOSE", "false").lower() == "true",
    "scan_full_page": True,
    "delay_before_return_html": 2,
}

# MCP server configuration
DEFAULT_MCP_PORT = 11235


class MCPServer:
    """Simple wrapper for the Crawl4ai MCP server functionality"""
    
    def __init__(self, port: int = DEFAULT_MCP_PORT):
        self.port = port
        self.process = None
    
    async def start(self):
        """Start the MCP server"""
        import subprocess
        
        # Check if the port is already in use
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(("localhost", self.port))
            port_available = True
        except:
            port_available = False
        finally:
            sock.close()
        
        if not port_available:
            console.print(f"[bold yellow]Warning:[/] Port {self.port} is already in use. MCP server may already be running.")
            return
        
        # Start the Crawl4ai MCP server
        console.print(f"[bold blue]Starting MCP server on port {self.port}[/]")
        try:
            # Use the Crawl4ai CLI to start the server
            self.process = subprocess.Popen(
                ["crwl", "server", "--port", str(self.port)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            
            # Wait a moment for the server to start
            await asyncio.sleep(2)
            
            if self.process.poll() is not None:
                # Process has terminated
                stdout, stderr = self.process.communicate()
                console.print(f"[bold red]Error starting MCP server:[/] {stderr}")
                return False
            
            console.print("[bold green]MCP server started successfully[/]")
            return True
        except Exception as e:
            console.print(f"[bold red]Failed to start MCP server:[/] {str(e)}")
            return False
    
    def stop(self):
        """Stop the MCP server"""
        if self.process:
            console.print("[bold blue]Stopping MCP server[/]")
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except:
                self.process.kill()
            console.print("[bold green]MCP server stopped[/]")


async def perform_crawl(url: str, browser_config: Dict = None, crawler_config: Dict = None) -> CrawlResult:
    """
    Perform a basic web crawl using Crawl4ai.
    
    Args:
        url: The URL to crawl
        browser_config: Optional browser configuration
        crawler_config: Optional crawler configuration
        
    Returns:
        CrawlResult: The result of the crawl operation
    """
    # Use default configs if not provided
    browser_config = browser_config or DEFAULT_BROWSER_CONFIG
    crawler_config = crawler_config or DEFAULT_CRAWLER_CONFIG
    
    browser_cfg = BrowserConfig(**browser_config)
    crawler_cfg = CrawlerRunConfig(**crawler_config)
    
    # Initialize the crawler
    crawler = AsyncWebCrawler(config=browser_cfg)
    await crawler.start()
    
    try:
        # Execute the crawl
        result = await crawler.arun(url=url, config=crawler_cfg)
        return result
    finally:
        # Always close the crawler to clean up resources
        await crawler.close()


@click.group()
def cli():
    """Crawl4ai Reference Application - MCP Server Demo"""
    pass


@cli.command("start")
@click.option("--port", "-p", default=DEFAULT_MCP_PORT, help="Port to run the MCP server on")
def start_cmd(port: int):
    """Start the Crawl4ai MCP server"""
    server = MCPServer(port)
    asyncio.run(server.start())
    
    console.print("\n[bold green]MCP Server is running![/]")
    console.print(f"Server URL: [bold]http://localhost:{port}[/]")
    console.print(f"SSE Endpoint: [bold]http://localhost:{port}/mcp/sse[/]")
    
    # Show instructions for connecting to the MCP server
    console.print("\n[bold]To connect to this MCP server from Claude:[/]")
    console.print(Syntax(f"claude mcp add --transport sse c4ai-sse http://localhost:{port}/mcp/sse", "bash"))
    
    console.print("\n[bold]To connect to this MCP server from LiteLLM:[/]")
    console.print(Syntax(f"""
from litellm import completion

# Add MCP server to LiteLLM
litellm.add_mcp_server(
    server_id="crawl4ai",
    name="crawl4ai_server",
    url="http://localhost:{port}/mcp/sse",
    transport="sse",
    auth_type="none",
    spec_version="2025-03-26"
)

# Use the MCP server in a completion
response = completion(
    model="gpt-4",
    messages=[
        {{"role": "user", "content": "Crawl example.com and summarize the content"}}
    ],
    tools=[{{"type": "mcp_server", "server_id": "crawl4ai"}}]
)
    """, "python"))
    
    console.print("\n[bold yellow]Press Ctrl+C to stop the server[/]")
    
    try:
        # Keep the server running until interrupted
        while True:
            asyncio.run(asyncio.sleep(1))
    except KeyboardInterrupt:
        console.print("\n[bold blue]Stopping MCP server...[/]")
        server.stop()
        console.print("[bold green]Server stopped[/]")


@cli.command("test")
@click.argument("url", required=True)
@click.option("--port", "-p", default=DEFAULT_MCP_PORT, help="Port of the MCP server")
def test_cmd(url: str, port: int):
    """Test crawling a URL using the MCP server"""
    import httpx
    
    console.print(f"[bold blue]Testing MCP server with URL:[/] {url}")
    
    # Check if the MCP server is running
    try:
        response = httpx.get(f"http://localhost:{port}/health")
        if response.status_code != 200:
            console.print(f"[bold red]MCP server is not running or not healthy. Status code: {response.status_code}[/]")
            return
    except Exception as e:
        console.print(f"[bold red]Failed to connect to MCP server:[/] {str(e)}")
        console.print(f"[bold yellow]Make sure the server is running with:[/] python mcp_demo.py start --port {port}")
        return
    
    # Construct the MCP tool call payload
    payload = {
        "name": "crawl_url",
        "arguments": json.dumps({
            "url": url,
            "output_format": "markdown"
        })
    }
    
    # Call the MCP tool
    try:
        response = httpx.post(
            f"http://localhost:{port}/mcp/tools/call",
            json=payload
        )
        
        if response.status_code == 200:
            result = response.json()
            console.print("[bold green]MCP tool call successful![/]")
            
            # Display the result
            if "content" in result and result["content"]:
                content = result["content"][0].get("text", "")
                console.print(Panel(Markdown(content), title=f"Content from {url}", expand=False))
            else:
                console.print("[yellow]No content returned[/]")
        else:
            console.print(f"[bold red]MCP tool call failed with status code: {response.status_code}[/]")
            console.print(response.text)
    except Exception as e:
        console.print(f"[bold red]Error calling MCP tool:[/] {str(e)}")


if __name__ == "__main__":
    try:
        cli()
    except Exception as e:
        console.print(f"[bold red]Error:[/] {str(e)}")
        sys.exit(1) 