#!/usr/bin/env python3
"""
B4UCrawl - Crawl4ai Reference Application

This script provides a unified interface to all the Crawl4ai demos.
"""

import os
import sys
import click
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

# Initialize rich console for pretty output
console = Console()


@click.group()
def cli():
    """B4UCrawl - Crawl4ai Reference Application

    This application demonstrates the core functionality of Crawl4ai,
    an open-source LLM-friendly web crawler and scraper.
    """
    pass


@cli.command("crawl")
@click.argument("url", required=True)
@click.option("--output", "-o", type=click.Choice(["json", "markdown"]), default="markdown", 
              help="Output format (json or markdown)")
@click.option("--headless/--no-headless", default=True, help="Run in headless mode")
@click.option("--verbose/--no-verbose", default=False, help="Enable verbose output")
def crawl_cmd(url, output, headless, verbose):
    """Crawl a website and display its content"""
    # Import and run the crawl_demo.py script
    from crawl_demo import crawl_cmd as run_crawl
    run_crawl(url, output, headless, verbose)


@cli.command("extract")
@click.argument("url", required=True)
@click.argument("instruction", required=False, default="Extract main content and metadata as structured JSON")
@click.option("--provider", "-p", help="LLM provider (e.g., openai/gpt-4)")
@click.option("--token", "-t", help="API token for the LLM provider")
def extract_cmd(url, instruction, provider, token):
    """Extract structured data from a website using LLM"""
    # Import and run the crawl_demo.py script's extract command
    from crawl_demo import extract_cmd as run_extract
    run_extract(url, instruction, provider, token)


@cli.command("tables")
@click.argument("url", required=True)
@click.option("--export", "-e", is_flag=True, help="Export tables to CSV files")
@click.option("--output-dir", "-o", default=".", help="Directory to save exported tables")
@click.option("--threshold", "-t", default=8, type=int, help="Table score threshold (1-10)")
@click.option("--headless/--no-headless", default=True, help="Run in headless mode")
@click.option("--verbose/--no-verbose", default=False, help="Enable verbose output")
def tables_cmd(url, export, output_dir, threshold, headless, verbose):
    """Extract tables from a website"""
    # Import and run the table_extraction_demo.py script
    from table_extraction_demo import extract_cmd as run_tables
    run_tables(url, export, output_dir, threshold, headless, verbose)


@cli.command("pandas")
@click.argument("url", required=True)
@click.option("--threshold", "-t", default=8, type=int, help="Table score threshold (1-10)")
def pandas_cmd(url, threshold):
    """Extract tables and convert to pandas DataFrame"""
    # Import and run the table_extraction_demo.py script's pandas command
    from table_extraction_demo import pandas_cmd as run_pandas
    run_pandas(url, threshold)


@cli.command("mcp-start")
@click.option("--port", "-p", default=11235, help="Port to run the MCP server on")
def mcp_start_cmd(port):
    """Start the Crawl4ai MCP server"""
    # Import and run the mcp_demo.py script's start command
    from mcp_demo import start_cmd as run_mcp_start
    run_mcp_start(port)


@cli.command("mcp-test")
@click.argument("url", required=True)
@click.option("--port", "-p", default=11235, help="Port of the MCP server")
def mcp_test_cmd(url, port):
    """Test crawling a URL using the MCP server"""
    # Import and run the mcp_demo.py script's test command
    from mcp_demo import test_cmd as run_mcp_test
    run_mcp_test(url, port)


@cli.command("mcp-tools")
@click.option("--host", default="localhost", help="MCP server host")
@click.option("--port", "-p", default=11235, help="MCP server port")
def mcp_tools_cmd(host, port):
    """List all available MCP tools"""
    # Import and run the mcp_client_demo.py script's list command
    from mcp_client_demo import list_cmd as run_mcp_tools
    run_mcp_tools(host, port)


@cli.command("mcp-crawl")
@click.argument("url", required=True)
@click.option("--host", default="localhost", help="MCP server host")
@click.option("--port", "-p", default=11235, help="MCP server port")
@click.option("--output", "-o", type=click.Choice(["markdown", "json"]), default="markdown", 
              help="Output format (markdown or json)")
def mcp_crawl_cmd(url, host, port, output):
    """Crawl a website using the MCP server"""
    # Import and run the mcp_client_demo.py script's crawl command
    from mcp_client_demo import crawl_cmd as run_mcp_crawl
    run_mcp_crawl(url, host, port, output)


@cli.command("about")
def about_cmd():
    """Show information about Crawl4ai"""
    about_text = """
# Crawl4ai Reference Application

This application demonstrates the core functionality of [Crawl4ai](https://github.com/unclecode/crawl4ai), 
an open-source LLM-friendly web crawler and scraper.

## Features

- **Basic Crawling**: Extract content from websites in markdown or JSON format
- **Structured Data Extraction**: Use LLMs to extract specific data from websites
- **Table Extraction**: Extract tables from websites to CSV or pandas DataFrames
- **MCP Integration**: Use the Model Context Protocol to connect Crawl4ai to AI models

## Commands

- `crawl`: Basic web crawling and content extraction
- `extract`: Extract structured data using LLMs
- `tables`: Extract tables from websites
- `pandas`: Convert extracted tables to pandas DataFrames
- `mcp-start`: Start the Crawl4ai MCP server
- `mcp-test`: Test crawling a URL using the MCP server
- `mcp-tools`: List all available MCP tools
- `mcp-crawl`: Crawl a website using the MCP server

## Learn More

Visit the [Crawl4ai GitHub repository](https://github.com/unclecode/crawl4ai) for more information.
    """
    
    console.print(Panel(Markdown(about_text), title="About B4UCrawl", expand=False))


if __name__ == "__main__":
    try:
        if len(sys.argv) == 1:
            # If no arguments are provided, show the about page
            about_cmd()
            console.print("\nRun with --help to see available commands.")
        else:
            cli()
    except Exception as e:
        console.print(f"[bold red]Error:[/] {str(e)}")
        sys.exit(1) 