#!/usr/bin/env python3
"""
Crawl4ai Reference Application - Basic Crawling Demo

This script demonstrates the basic usage of Crawl4ai for web crawling and content extraction.
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

from crawl4ai.async_configs import BrowserConfig, CrawlerRunConfig
from crawl4ai.async_crawler import AsyncWebCrawler
from crawl4ai.extraction import LLMExtractionStrategy, LLMConfig
from crawl4ai.models import CrawlResult

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
    browser_cfg = BrowserConfig.from_kwargs(browser_config or DEFAULT_BROWSER_CONFIG)
    crawler_cfg = CrawlerRunConfig.from_kwargs(crawler_config or DEFAULT_CRAWLER_CONFIG)
    
    # Initialize the crawler
    crawler = AsyncWebCrawler(config=browser_cfg)
    await crawler.start()
    
    try:
        # Execute the crawl
        results = await crawler.arun(url=url, config=crawler_cfg)
        # Return the first result (there's only one since we're crawling a single URL)
        return results[0] if isinstance(results, list) else results
    finally:
        # Always stop the crawler to clean up resources
        await crawler.stop()


async def extract_structured_data(url: str, instruction: str, llm_provider: str = None, api_token: str = None) -> Dict:
    """
    Extract structured data from a webpage using LLM extraction.
    
    Args:
        url: The URL to extract data from
        instruction: Instructions for the LLM on what data to extract
        llm_provider: Optional LLM provider (e.g., "openai/gpt-4")
        api_token: Optional API token for the LLM provider
        
    Returns:
        Dict: The extracted structured data
    """
    # Use environment variables if not provided
    provider = llm_provider or os.getenv("CRAWL4AI_LLM_PROVIDER", "openai/gpt-3.5-turbo")
    token = api_token or os.getenv("CRAWL4AI_LLM_TOKEN")
    
    if not token:
        console.print("[bold red]Error:[/] LLM API token not provided. Set CRAWL4AI_LLM_TOKEN environment variable.")
        return {}
    
    # Configure browser and crawler
    browser_cfg = BrowserConfig.from_kwargs(DEFAULT_BROWSER_CONFIG)
    
    # Create LLM extraction strategy
    extraction_strategy = LLMExtractionStrategy(
        llm_config=LLMConfig(provider=provider, api_token=token),
        instruction=instruction,
        extraction_type="schema",
        force_json_response=True,
        verbose=True,
    )
    
    # Configure crawler with extraction strategy
    crawler_cfg = CrawlerRunConfig.from_kwargs({
        **DEFAULT_CRAWLER_CONFIG,
        "extraction_strategy": extraction_strategy,
    })
    
    # Initialize the crawler
    crawler = AsyncWebCrawler(config=browser_cfg)
    await crawler.start()
    
    try:
        # Execute the crawl with extraction
        results = await crawler.arun(url=url, config=crawler_cfg)
        result = results[0] if isinstance(results, list) else results
        
        # Parse and return the extracted content
        if result.success and result.extracted_content:
            return json.loads(result.extracted_content)
        else:
            console.print(f"[bold red]Error:[/] Failed to extract data: {result.error_message}")
            return {}
    finally:
        # Always stop the crawler
        await crawler.stop()


@click.group()
def cli():
    """Crawl4ai Reference Application - Basic Crawling Demo"""
    pass


@cli.command("crawl")
@click.argument("url", required=True)
@click.option("--output", "-o", type=click.Choice(["json", "markdown"]), default="markdown", 
              help="Output format (json or markdown)")
@click.option("--headless/--no-headless", default=True, help="Run in headless mode")
@click.option("--verbose/--no-verbose", default=False, help="Enable verbose output")
def crawl_cmd(url: str, output: str, headless: bool, verbose: bool):
    """Crawl a website and display its content"""
    # Update config with command line options
    browser_config = {**DEFAULT_BROWSER_CONFIG, "headless": headless}
    crawler_config = {**DEFAULT_CRAWLER_CONFIG, "verbose": verbose}
    
    console.print(f"[bold blue]Crawling:[/] {url}")
    
    # Run the crawl
    result = asyncio.run(perform_crawl(url, browser_config, crawler_config))
    
    if result.success:
        console.print("[bold green]Crawl successful![/]")
        
        if output == "json":
            # Display as JSON
            console.print(Panel(json.dumps(result.model_dump(), indent=2), 
                               title="Crawl Result", expand=False))
        else:
            # Display as Markdown
            if result.markdown and result.markdown.raw_markdown:
                console.print(Panel(Markdown(result.markdown.raw_markdown), 
                                   title=f"Content from {url}", expand=False))
            else:
                console.print("[yellow]No markdown content available[/]")
    else:
        console.print(f"[bold red]Crawl failed:[/] {result.error_message}")


@cli.command("extract")
@click.argument("url", required=True)
@click.argument("instruction", required=False, default="Extract main content and metadata as structured JSON")
@click.option("--provider", "-p", help="LLM provider (e.g., openai/gpt-4)")
@click.option("--token", "-t", help="API token for the LLM provider")
def extract_cmd(url: str, instruction: str, provider: str, token: str):
    """Extract structured data from a website using LLM"""
    console.print(f"[bold blue]Extracting data from:[/] {url}")
    console.print(f"[bold blue]Instruction:[/] {instruction}")
    
    # Run the extraction
    result = asyncio.run(extract_structured_data(url, instruction, provider, token))
    
    if result:
        console.print("[bold green]Extraction successful![/]")
        console.print(Panel(json.dumps(result, indent=2), 
                           title="Extracted Data", expand=False))
    else:
        console.print("[bold red]Extraction failed[/]")


if __name__ == "__main__":
    try:
        cli()
    except Exception as e:
        console.print(f"[bold red]Error:[/] {str(e)}")
        sys.exit(1) 