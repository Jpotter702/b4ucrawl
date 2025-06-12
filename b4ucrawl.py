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
    from crawl_demo import run_crawl
    run_crawl(url, output, headless, verbose)


@cli.command("extract")
@click.argument("url", required=True)
@click.argument("instruction", required=False, default="Extract main content and metadata as structured JSON")
@click.option("--provider", "-p", help="LLM provider (e.g., openai/gpt-4)")
@click.option("--token", "-t", help="API token for the LLM provider")
def extract_cmd(url, instruction, provider, token):
    """Extract structured data from a website using LLM"""
    # Import and run the crawl_demo.py script's extract command
    from crawl_demo import run_extract
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


# Core Feature Commands

@cli.command("js")
@click.argument("url", required=True)
@click.argument("js_code", required=True)
@click.option("--wait-for", "-w", default="2000", help="Condition to wait for after JS execution")
def js_cmd(url, js_code, wait_for):
    """Execute JavaScript code on a webpage"""
    from javascript_demo import run_custom_js_demo
    run_custom_js_demo(url, js_code, wait_for)


@cli.command("click")
@click.argument("url", required=True)
@click.argument("selector", required=True)
@click.option("--wait-for", "-w", default="2000", help="Condition to wait for after clicking")
def click_cmd(url, selector, wait_for):
    """Click an element and show the resulting content"""
    from javascript_demo import run_click_demo
    run_click_demo(url, selector, wait_for)


@cli.command("css-extract")
@click.argument("url", required=True)
@click.option("--schema", "-s", type=click.Choice(["news", "products", "social"]), default="news", 
              help="Predefined schema type")
def css_extract_cmd(url, schema):
    """Extract data using CSS selectors"""
    from extraction_strategies_demo import run_css_demo
    run_css_demo(url, schema, None)


@cli.command("regex-extract")
@click.argument("url", required=True)
@click.option("--patterns", "-p", type=click.Choice(["contact", "financial"]), default="contact",
              help="Predefined pattern type")
def regex_extract_cmd(url, patterns):
    """Extract data using regular expressions"""
    from extraction_strategies_demo import run_regex_demo
    run_regex_demo(url, patterns, None)


@cli.command("filter")
@click.argument("url", required=True)
@click.option("--type", "-t", type=click.Choice(["pruning", "bm25"]), default="pruning",
              help="Content filter type")
def filter_cmd(url, type):
    """Apply content filtering to webpage"""
    from content_processing_demo import run_filter_demo
    run_filter_demo(url, type, 500)


@cli.command("screenshot")
@click.argument("url", required=True)
@click.option("--width", "-w", default=1280, type=int, help="Viewport width")
@click.option("--height", "-h", default=800, type=int, help="Viewport height")
@click.option("--output-dir", "-o", default="screenshots", help="Output directory")
def screenshot_cmd(url, width, height, output_dir):
    """Capture a screenshot of a webpage"""
    from screenshot_pdf_demo import run_screenshot_demo
    run_screenshot_demo(url, output_dir, width, height, 2.0, True)


@cli.command("pdf")
@click.argument("url", required=True)
@click.option("--output-dir", "-o", default="pdfs", help="Output directory")
def pdf_cmd(url, output_dir):
    """Generate a PDF of a webpage"""
    from screenshot_pdf_demo import run_pdf_demo
    run_pdf_demo(url, output_dir, True)


# Advanced Crawling Commands (Priority 2)

@cli.command("deep-crawl")
@click.argument("start_url", required=True)
@click.option("--strategy", "-s", type=click.Choice(["bfs", "dfs", "best-first"]), default="bfs",
              help="Deep crawling strategy")
@click.option("--max-depth", "-d", default=2, type=int, help="Maximum crawl depth")
@click.option("--max-pages", "-p", default=10, type=int, help="Maximum pages to crawl")
@click.option("--keywords", "-k", help="Keywords for best-first strategy (comma-separated)")
def deep_crawl_cmd(start_url, strategy, max_depth, max_pages, keywords):
    """Perform deep crawling with BFS/DFS/Best-First strategies"""
    from deep_crawling_demo import run_bfs_demo, run_dfs_demo, run_best_first_demo
    
    if strategy == "bfs":
        run_bfs_demo(start_url, max_depth, max_pages, "", "", False)
    elif strategy == "dfs":
        run_dfs_demo(start_url, max_depth, max_pages, False)
    elif strategy == "best-first":
        if not keywords:
            keywords = "python,programming,tutorial"
        run_best_first_demo(start_url, keywords, max_depth, max_pages, False)


@cli.command("batch-crawl")
@click.argument("urls", required=True)
@click.option("--strategy", "-s", type=click.Choice(["simple", "domain", "rate-limited"]), default="simple",
              help="Batch processing strategy")
@click.option("--max-concurrent", "-c", default=3, type=int, help="Maximum concurrent requests")
@click.option("--rate", "-r", default=1.0, type=float, help="Rate limit (requests per second)")
def batch_crawl_cmd(urls, strategy, max_concurrent, rate):
    """Batch process multiple URLs with different strategies"""
    from batch_processing_demo import run_simple_batch_demo, run_domain_batch_demo, run_rate_limited_demo
    
    if strategy == "simple":
        run_simple_batch_demo(urls, max_concurrent, False)
    elif strategy == "domain":
        run_domain_batch_demo(urls, False)
    elif strategy == "rate-limited":
        run_rate_limited_demo(urls, rate, False)


@cli.command("proxy-crawl")
@click.argument("url", required=True)
@click.option("--proxy-file", "-f", help="JSON file containing proxy list")
@click.option("--strategy", "-s", type=click.Choice(["rotation", "failover"]), default="rotation",
              help="Proxy usage strategy")
def proxy_crawl_cmd(url, proxy_file, strategy):
    """Crawl using proxy rotation or failover strategies"""
    from proxy_demo import run_rotation_demo, run_failover_demo
    
    if strategy == "rotation":
        run_rotation_demo(url, proxy_file, "round_robin", False)
    elif strategy == "failover":
        run_failover_demo(url, proxy_file, False)


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
- **JavaScript Interaction**: Execute JavaScript, click elements, handle dynamic content
- **Advanced Extraction**: CSS selectors, XPath, regex-based data extraction
- **Content Processing**: Content filtering, chunking, markdown generation
- **Visual Capture**: Screenshots and PDF generation with custom settings
- **MCP Integration**: Use the Model Context Protocol to connect Crawl4ai to AI models

## Basic Commands

- `crawl`: Basic web crawling and content extraction
- `extract`: Extract structured data using LLMs
- `tables`: Extract tables from websites
- `pandas`: Convert extracted tables to pandas DataFrames

## Core Feature Commands

- `js`: Execute JavaScript code on webpages
- `click`: Click elements and show resulting content
- `css-extract`: Extract data using CSS selectors
- `regex-extract`: Extract data using regular expressions
- `filter`: Apply content filtering (pruning, BM25)
- `screenshot`: Capture webpage screenshots
- `pdf`: Generate PDF documents from webpages

## Advanced Crawling Commands

- `deep-crawl`: Perform deep crawling with BFS/DFS/Best-First strategies
- `batch-crawl`: Batch process multiple URLs with different strategies
- `proxy-crawl`: Crawl using proxy rotation or failover strategies

## MCP Commands

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