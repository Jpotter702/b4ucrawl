#!/usr/bin/env python3
"""
Crawl4ai Reference Application - Content Processing Demo

This script demonstrates content processing strategies available in Crawl4ai:
- Content Filtering: PruningContentFilter, BM25ContentFilter, LLMContentFilter
- Chunking Strategies: RegexChunking for large content processing
- Markdown Generation: DefaultMarkdownGenerator with different filters
"""

import asyncio
import json
import os
import sys
from typing import Dict, List, Optional, Any

import click
from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from crawl4ai import (
    AsyncWebCrawler, 
    BrowserConfig, 
    CrawlerRunConfig, 
    CrawlResult,
    DefaultMarkdownGenerator,
    PruningContentFilter,
    BM25ContentFilter,
    LLMContentFilter,
    LLMConfig,
    RegexChunking
)

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


async def perform_content_crawl(url: str, markdown_generator=None, chunking_strategy=None) -> CrawlResult:
    """
    Perform a web crawl with specific content processing strategies.
    
    Args:
        url: The URL to crawl
        markdown_generator: The markdown generation strategy to use
        chunking_strategy: The chunking strategy to use
        
    Returns:
        CrawlResult: The result of the crawl operation
    """
    browser_config = DEFAULT_BROWSER_CONFIG.copy()
    crawler_config = DEFAULT_CRAWLER_CONFIG.copy()
    
    if markdown_generator:
        crawler_config["markdown_generator"] = markdown_generator
    if chunking_strategy:
        crawler_config["chunking_strategy"] = chunking_strategy
    
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


async def markdown_generation_demo(url: str, filter_type: str = None) -> Dict[str, CrawlResult]:
    """
    Demonstrate different markdown generation with content filters.
    
    Args:
        url: The URL to crawl
        filter_type: Type of filter to apply ('pruning', 'bm25', 'llm', or None)
        
    Returns:
        Dict of filter names to CrawlResult objects
    """
    results = {}
    
    # Default markdown generation (no filter)
    console.print("[bold yellow]1. Default Markdown Generation (no filter)[/]")
    default_generator = DefaultMarkdownGenerator()
    results["default"] = await perform_content_crawl(url, markdown_generator=default_generator)
    
    if filter_type is None or filter_type == "pruning":
        # With PruningContentFilter
        console.print("[bold yellow]2. Markdown with Pruning Content Filter[/]")
        pruning_generator = DefaultMarkdownGenerator(content_filter=PruningContentFilter())
        results["pruning"] = await perform_content_crawl(url, markdown_generator=pruning_generator)
    
    if filter_type is None or filter_type == "bm25":
        # With BM25ContentFilter
        console.print("[bold yellow]3. Markdown with BM25 Content Filter[/]")
        bm25_generator = DefaultMarkdownGenerator(content_filter=BM25ContentFilter())
        results["bm25"] = await perform_content_crawl(url, markdown_generator=bm25_generator)
    
    if filter_type == "llm":
        # With LLMContentFilter (requires API key)
        llm_provider = os.getenv("CRAWL4AI_LLM_PROVIDER", "openai/gpt-3.5-turbo")
        llm_token = os.getenv("CRAWL4AI_LLM_TOKEN")
        
        if llm_token:
            console.print("[bold yellow]4. Markdown with LLM Content Filter[/]")
            llm_config = LLMConfig(provider=llm_provider, api_token=llm_token)
            llm_filter = LLMContentFilter(llm_config=llm_config)
            llm_generator = DefaultMarkdownGenerator(content_filter=llm_filter)
            results["llm"] = await perform_content_crawl(url, markdown_generator=llm_generator)
        else:
            console.print("[bold red]Skipping LLM filter: CRAWL4AI_LLM_TOKEN not set[/]")
    
    return results


async def chunking_demo(url: str, chunk_size: int = 1000) -> CrawlResult:
    """
    Demonstrate content chunking for large documents.
    
    Args:
        url: The URL to crawl
        chunk_size: Size of each chunk in characters
        
    Returns:
        CrawlResult: The crawl result with chunked content
    """
    console.print(f"[bold yellow]Chunking Content (chunk size: {chunk_size})[/]")
    
    # Use RegexChunking strategy
    chunking_strategy = RegexChunking(chunk_size=chunk_size)
    
    result = await perform_content_crawl(url, chunking_strategy=chunking_strategy)
    return result


def display_content_comparison(results: Dict[str, CrawlResult]):
    """Display comparison of different content processing results."""
    
    # Create comparison table
    table = Table(title="Content Processing Comparison")
    table.add_column("Filter Type", style="cyan")
    table.add_column("Success", style="green")
    table.add_column("Raw Markdown Length", style="yellow")
    table.add_column("Fit Markdown Length", style="blue")
    table.add_column("Content Reduction", style="magenta")
    
    for filter_name, result in results.items():
        if result.success:
            raw_len = len(result.markdown.raw_markdown) if result.markdown and result.markdown.raw_markdown else 0
            fit_len = len(result.markdown.fit_markdown) if result.markdown and result.markdown.fit_markdown else 0
            
            if raw_len > 0:
                reduction = f"{((raw_len - fit_len) / raw_len * 100):.1f}%"
            else:
                reduction = "N/A"
            
            table.add_row(
                filter_name.title(),
                "✓",
                f"{raw_len:,}",
                f"{fit_len:,}",
                reduction
            )
        else:
            table.add_row(filter_name.title(), "✗", "0", "0", "N/A")
    
    console.print(table)


def display_markdown_sample(result: CrawlResult, filter_name: str, sample_length: int = 500):
    """Display a sample of the markdown content."""
    if result.success and result.markdown:
        # Show fit_markdown if available, otherwise raw_markdown
        content = result.markdown.fit_markdown or result.markdown.raw_markdown
        
        if content:
            sample = content[:sample_length] + "..." if len(content) > sample_length else content
            console.print(Panel(
                Markdown(sample),
                title=f"Sample Content ({filter_name})",
                expand=False
            ))
        else:
            console.print(f"[yellow]No markdown content available for {filter_name}[/]")
    else:
        console.print(f"[red]Failed to process content with {filter_name} filter[/]")


def display_chunks(result: CrawlResult, max_chunks: int = 3):
    """Display information about content chunks."""
    if result.success and hasattr(result, 'chunked_content') and result.chunked_content:
        console.print(f"[bold green]Content chunked into {len(result.chunked_content)} pieces[/]")
        
        for i, chunk in enumerate(result.chunked_content[:max_chunks]):
            console.print(f"\n[bold]Chunk {i+1}:[/]")
            sample = chunk[:200] + "..." if len(chunk) > 200 else chunk
            console.print(Panel(sample, expand=False))
        
        if len(result.chunked_content) > max_chunks:
            console.print(f"\n[yellow]... and {len(result.chunked_content) - max_chunks} more chunks[/]")
    else:
        console.print("[yellow]No chunked content available[/]")


# CLI Commands

def run_filter_demo(url: str, filter_type: str, sample_length: int):
    """Core filter demo function without Click decorators"""
    console.print(f"[bold blue]Content Filter Demo: {url}[/]")
    
    results = asyncio.run(markdown_generation_demo(url, filter_type))
    
    # Display comparison table
    display_content_comparison(results)
    
    # Display samples
    for filter_name, result in results.items():
        console.print(f"\n[bold cyan]--- {filter_name.title()} Filter Sample ---[/]")
        display_markdown_sample(result, filter_name, sample_length)


def run_chunking_demo(url: str, chunk_size: int, max_chunks: int):
    """Core chunking demo function without Click decorators"""
    console.print(f"[bold blue]Content Chunking Demo: {url}[/]")
    console.print(f"[bold blue]Chunk size: {chunk_size} characters[/]")
    
    result = asyncio.run(chunking_demo(url, chunk_size))
    
    if result.success:
        console.print("[bold green]Chunking successful![/]")
        
        # Display basic info
        if result.markdown and result.markdown.raw_markdown:
            total_length = len(result.markdown.raw_markdown)
            console.print(f"[bold]Total content length:[/] {total_length:,} characters")
        
        # Display chunks
        display_chunks(result, max_chunks)
        
        # Show markdown sample too
        display_markdown_sample(result, "chunked", 300)
        
    else:
        console.print(f"[bold red]Chunking failed: {result.error_message}[/]")


def run_comparison_demo(url: str):
    """Compare all content processing methods"""
    console.print(f"[bold blue]Full Content Processing Comparison: {url}[/]")
    
    # Test all filter types
    results = asyncio.run(markdown_generation_demo(url))
    
    # Display comparison
    display_content_comparison(results)
    
    # Show the most filtered version
    if "pruning" in results and results["pruning"].success:
        console.print("\n[bold cyan]--- Most Filtered Content (Pruning) ---[/]")
        display_markdown_sample(results["pruning"], "pruning", 800)
    
    # Test chunking
    console.print("\n[bold cyan]--- Chunking Demo ---[/]")
    chunk_result = asyncio.run(chunking_demo(url, 500))
    display_chunks(chunk_result, 2)


def run_custom_filter_demo(url: str):
    """Demonstrate custom filter configuration"""
    console.print(f"[bold blue]Custom Filter Configuration Demo: {url}[/]")
    
    # Create custom BM25 filter with specific parameters
    console.print("[bold yellow]Custom BM25 Filter (top 5 most relevant sections)[/]")
    
    async def custom_bm25_demo():
        custom_bm25_filter = BM25ContentFilter(k=5)  # Keep only top 5 most relevant sections
        custom_generator = DefaultMarkdownGenerator(content_filter=custom_bm25_filter)
        return await perform_content_crawl(url, markdown_generator=custom_generator)
    
    result = asyncio.run(custom_bm25_demo())
    
    if result.success:
        console.print("[bold green]Custom filtering successful![/]")
        display_markdown_sample(result, "custom BM25", 600)
        
        if result.markdown:
            raw_len = len(result.markdown.raw_markdown) if result.markdown.raw_markdown else 0
            fit_len = len(result.markdown.fit_markdown) if result.markdown.fit_markdown else 0
            console.print(f"[bold]Content reduction:[/] {raw_len:,} → {fit_len:,} characters")
    else:
        console.print(f"[bold red]Custom filtering failed: {result.error_message}[/]")


@click.group()
def cli():
    """Crawl4ai Reference Application - Content Processing Demo"""
    pass


@cli.command("filter")
@click.argument("url", required=True)
@click.option("--type", "-t", type=click.Choice(["pruning", "bm25", "llm"]), 
              help="Specific filter type to test")
@click.option("--sample-length", "-s", default=500, type=int, 
              help="Length of content sample to display")
def filter_cmd(url: str, type: str, sample_length: int):
    """Demonstrate content filtering strategies"""
    run_filter_demo(url, type, sample_length)


@cli.command("chunk")
@click.argument("url", required=True)
@click.option("--size", "-s", default=1000, type=int, help="Chunk size in characters")
@click.option("--max-chunks", "-m", default=3, type=int, help="Maximum chunks to display")
def chunk_cmd(url: str, size: int, max_chunks: int):
    """Demonstrate content chunking"""
    run_chunking_demo(url, size, max_chunks)


@cli.command("compare")
@click.argument("url", required=True)
def compare_cmd(url: str):
    """Compare all content processing methods"""
    run_comparison_demo(url)


@cli.command("custom")
@click.argument("url", required=True)
def custom_cmd(url: str):
    """Demonstrate custom filter configurations"""
    run_custom_filter_demo(url)


@cli.command("info")
def info_cmd():
    """Show information about content processing strategies"""
    console.print("[bold blue]Crawl4AI Content Processing Strategies[/]\n")
    
    console.print("[bold yellow]Content Filters:[/]")
    console.print("• [bold]PruningContentFilter[/]: Removes boilerplate content like headers, footers, navigation")
    console.print("• [bold]BM25ContentFilter[/]: Keeps most relevant content sections using BM25 scoring")
    console.print("• [bold]LLMContentFilter[/]: Uses LLM to identify and filter relevant content")
    
    console.print("\n[bold yellow]Chunking Strategies:[/]")
    console.print("• [bold]RegexChunking[/]: Splits content into chunks using regex patterns")
    
    console.print("\n[bold yellow]Markdown Generation:[/]")
    console.print("• [bold]DefaultMarkdownGenerator[/]: Converts HTML to clean markdown")
    console.print("• Can be combined with any content filter for enhanced processing")
    
    console.print("\n[bold green]When to Use Each:[/]")
    console.print("• [bold]Pruning[/]: General web pages with standard layout elements")
    console.print("• [bold]BM25[/]: Academic papers, long articles, research documents")
    console.print("• [bold]LLM[/]: Complex layouts, mixed content types, custom filtering needs")
    console.print("• [bold]Chunking[/]: Very long documents, batch processing, memory constraints")
    
    console.print("\n[bold green]Usage Examples:[/]")
    console.print("  python content_processing_demo.py filter https://example.com --type pruning")
    console.print("  python content_processing_demo.py chunk https://example.com --size 500")
    console.print("  python content_processing_demo.py compare https://example.com")


if __name__ == "__main__":
    try:
        cli()
    except Exception as e:
        console.print(f"[bold red]Error: {str(e)}[/]")
        sys.exit(1)