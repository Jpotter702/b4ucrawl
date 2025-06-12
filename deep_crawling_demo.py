#!/usr/bin/env python3
"""
Crawl4ai Reference Application - Deep Crawling Demo

This script demonstrates advanced crawling strategies available in Crawl4ai:
- Deep crawling with BFS/DFS strategies
- Link following and discovery
- URL filtering and scoring
- Crawl depth and breadth control
- Custom crawling patterns
"""

import asyncio
import json
import os
import sys
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path

import click
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.tree import Tree

from crawl4ai import (
    AsyncWebCrawler,
    BrowserConfig,
    CrawlerRunConfig,
    CrawlResult,
    # Deep crawling strategies
    BFSDeepCrawlStrategy,
    DFSDeepCrawlStrategy,
    BestFirstCrawlingStrategy,
    # Filters
    FilterChain,
    URLPatternFilter,
    DomainFilter,
    ContentTypeFilter,
    URLFilter,
    FilterStats,
    # Scorers
    KeywordRelevanceScorer,
    PathDepthScorer,
    CompositeScorer,
    URLScorer
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
    "delay_before_return_html": 1,
}


async def perform_deep_crawl(start_url: str, deep_crawl_strategy, max_depth: int = 2, max_pages: int = 10) -> List[CrawlResult]:
    """
    Perform a deep crawl using specified strategy.
    
    Args:
        start_url: The starting URL for deep crawling
        deep_crawl_strategy: The deep crawling strategy to use
        max_depth: Maximum depth to crawl
        max_pages: Maximum number of pages to crawl
        
    Returns:
        List[CrawlResult]: Results from all crawled pages
    """
    browser_config = DEFAULT_BROWSER_CONFIG.copy()
    crawler_config = DEFAULT_CRAWLER_CONFIG.copy()
    
    # Add deep crawling configuration
    crawler_config.update({
        "deep_crawl_strategy": deep_crawl_strategy,
    })
    
    browser_cfg = BrowserConfig(**browser_config)
    crawler_cfg = CrawlerRunConfig(**crawler_config)
    
    # Initialize the crawler
    crawler = AsyncWebCrawler(config=browser_cfg)
    await crawler.start()
    
    try:
        # Execute the deep crawl
        result = await crawler.arun(start_url, config=crawler_cfg)
        
        # Check for deep crawl results
        if hasattr(result, 'deep_crawl_results') and result.deep_crawl_results:
            return result.deep_crawl_results
        elif isinstance(result, list):
            return result
        else:
            return [result]
    finally:
        # Always close the crawler to clean up resources
        await crawler.close()


async def bfs_crawl_demo(start_url: str, max_depth: int = 2, max_pages: int = 10, 
                        include_patterns: List[str] = None, exclude_patterns: List[str] = None) -> List[CrawlResult]:
    """
    Demonstrate Breadth-First Search crawling.
    
    Args:
        start_url: Starting URL
        max_depth: Maximum crawl depth
        max_pages: Maximum pages to crawl
        include_patterns: URL patterns to include
        exclude_patterns: URL patterns to exclude
        
    Returns:
        List of crawl results
    """
    # Create filter chain
    filters = []
    
    if include_patterns:
        for pattern in include_patterns:
            filters.append(URLPatternFilter(pattern=pattern, action="include"))
    
    if exclude_patterns:
        for pattern in exclude_patterns:
            filters.append(URLPatternFilter(pattern=pattern, action="exclude"))
    
    # Add domain filter to stay within the same domain
    from urllib.parse import urlparse
    domain = urlparse(start_url).netloc
    filters.append(DomainFilter(allowed_domains=[domain]))
    
    filter_chain = FilterChain(filters=filters) if filters else None
    
    # Create BFS strategy
    bfs_strategy = BFSDeepCrawlStrategy(
        max_depth=max_depth,
        max_pages=max_pages,
        filter_chain=filter_chain
    )
    
    return await perform_deep_crawl(start_url, bfs_strategy, max_depth, max_pages)


async def dfs_crawl_demo(start_url: str, max_depth: int = 2, max_pages: int = 10) -> List[CrawlResult]:
    """
    Demonstrate Depth-First Search crawling.
    
    Args:
        start_url: Starting URL
        max_depth: Maximum crawl depth
        max_pages: Maximum pages to crawl
        
    Returns:
        List of crawl results
    """
    # Create filter chain with domain restriction
    from urllib.parse import urlparse
    domain = urlparse(start_url).netloc
    domain_filter = DomainFilter(allowed_domains=[domain])
    filter_chain = FilterChain(filters=[domain_filter])
    
    # Create DFS strategy
    dfs_strategy = DFSDeepCrawlStrategy(
        max_depth=max_depth,
        max_pages=max_pages,
        filter_chain=filter_chain
    )
    
    return await perform_deep_crawl(start_url, dfs_strategy, max_depth, max_pages)


async def best_first_crawl_demo(start_url: str, keywords: List[str], max_depth: int = 2, max_pages: int = 10) -> List[CrawlResult]:
    """
    Demonstrate Best-First crawling with keyword scoring.
    
    Args:
        start_url: Starting URL
        keywords: Keywords for relevance scoring
        max_depth: Maximum crawl depth
        max_pages: Maximum pages to crawl
        
    Returns:
        List of crawl results
    """
    # Create scorers
    keyword_scorer = KeywordRelevanceScorer(keywords=keywords)
    depth_scorer = PathDepthScorer()
    
    # Combine scorers (keyword relevance weighted higher)
    composite_scorer = CompositeScorer(
        scorers=[keyword_scorer, depth_scorer],
        weights=[0.8, 0.2]  # 80% keyword relevance, 20% depth preference
    )
    
    # Create filter chain
    from urllib.parse import urlparse
    domain = urlparse(start_url).netloc
    domain_filter = DomainFilter(allowed_domains=[domain])
    filter_chain = FilterChain(filters=[domain_filter])
    
    # Create Best-First strategy
    best_first_strategy = BestFirstCrawlingStrategy(
        max_depth=max_depth,
        max_pages=max_pages,
        filter_chain=filter_chain,
        scorer=composite_scorer
    )
    
    return await perform_deep_crawl(start_url, best_first_strategy, max_depth, max_pages)


def create_crawl_tree(results: List[CrawlResult], strategy_name: str) -> Tree:
    """Create a tree visualization of crawled pages."""
    tree = Tree(f"[bold blue]{strategy_name} Crawl Results[/]")
    
    if not results:
        tree.add("[red]No results[/]")
        return tree
    
    # Group by depth (approximated by URL path depth)
    depth_groups = {}
    for result in results:
        # Simple depth calculation based on URL path segments
        depth = len([p for p in result.url.split('/') if p]) - 2  # Subtract for protocol and domain
        depth = max(0, depth)
        
        if depth not in depth_groups:
            depth_groups[depth] = []
        depth_groups[depth].append(result)
    
    # Add to tree by depth
    for depth in sorted(depth_groups.keys()):
        depth_node = tree.add(f"[yellow]Depth {depth}[/]")
        
        for result in depth_groups[depth]:
            status = "[green]✓[/]" if result.success else "[red]✗[/]"
            url_display = result.url if len(result.url) <= 60 else result.url[:57] + "..."
            content_len = len(result.markdown.raw_markdown) if result.markdown and result.markdown.raw_markdown else 0
            
            depth_node.add(f"{status} {url_display} ({content_len:,} chars)")
    
    return tree


def display_crawl_results(results: List[CrawlResult], strategy_name: str):
    """Display comprehensive crawl results."""
    # Summary table
    table = Table(title=f"{strategy_name} Crawl Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="yellow")
    
    total_pages = len(results)
    successful_pages = len([r for r in results if r.success])
    total_content = sum(len(r.markdown.raw_markdown) if r.markdown and r.markdown.raw_markdown else 0 for r in results)
    unique_domains = len(set(r.url.split('/')[2] for r in results if r.url))
    
    table.add_row("Total Pages", str(total_pages))
    table.add_row("Successful", str(successful_pages))
    table.add_row("Success Rate", f"{(successful_pages/total_pages*100):.1f}%" if total_pages > 0 else "0%")
    table.add_row("Total Content", f"{total_content:,} characters")
    table.add_row("Unique Domains", str(unique_domains))
    table.add_row("Avg Content/Page", f"{total_content//total_pages:,} chars" if total_pages > 0 else "0")
    
    console.print(table)
    
    # Tree visualization
    tree = create_crawl_tree(results, strategy_name)
    console.print(tree)


def save_crawl_results(results: List[CrawlResult], strategy_name: str, output_dir: str = "crawl_results"):
    """Save crawl results to files."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save summary
    summary = {
        "strategy": strategy_name,
        "timestamp": timestamp,
        "total_pages": len(results),
        "successful_pages": len([r for r in results if r.success]),
        "pages": []
    }
    
    # Save individual results
    for i, result in enumerate(results):
        page_data = {
            "index": i,
            "url": result.url,
            "success": result.success,
            "content_length": len(result.markdown.raw_markdown) if result.markdown and result.markdown.raw_markdown else 0,
            "links_found": len(result.links.internal) + len(result.links.external) if result.links else 0,
            "error": result.error_message if not result.success else None
        }
        summary["pages"].append(page_data)
        
        # Save content if successful
        if result.success and result.markdown and result.markdown.raw_markdown:
            content_file = Path(output_dir) / f"{strategy_name}_{timestamp}_page_{i:03d}.md"
            with open(content_file, 'w', encoding='utf-8') as f:
                f.write(f"# {result.url}\n\n")
                f.write(result.markdown.raw_markdown)
    
    # Save summary JSON
    summary_file = Path(output_dir) / f"{strategy_name}_{timestamp}_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    console.print(f"[bold green]Results saved to {output_dir}/[/]")


# CLI Commands

def run_bfs_demo(start_url: str, max_depth: int, max_pages: int, include_patterns: str, exclude_patterns: str, save: bool):
    """Core BFS demo function without Click decorators"""
    console.print(f"[bold blue]BFS Deep Crawl Demo: {start_url}[/]")
    console.print(f"[bold blue]Max Depth: {max_depth}, Max Pages: {max_pages}[/]")
    
    # Parse patterns
    include_list = [p.strip() for p in include_patterns.split(',')] if include_patterns else None
    exclude_list = [p.strip() for p in exclude_patterns.split(',')] if exclude_patterns else None
    
    if include_list:
        console.print(f"[bold yellow]Include patterns: {include_list}[/]")
    if exclude_list:
        console.print(f"[bold yellow]Exclude patterns: {exclude_list}[/]")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
        task = progress.add_task("BFS Crawling...", total=max_pages)
        
        async def crawl_with_progress():
            results = await bfs_crawl_demo(start_url, max_depth, max_pages, include_list, exclude_list)
            progress.update(task, completed=len(results))
            return results
        
        results = asyncio.run(crawl_with_progress())
    
    display_crawl_results(results, "BFS")
    
    if save:
        save_crawl_results(results, "bfs", "deep_crawl_results")


def run_dfs_demo(start_url: str, max_depth: int, max_pages: int, save: bool):
    """Core DFS demo function without Click decorators"""
    console.print(f"[bold blue]DFS Deep Crawl Demo: {start_url}[/]")
    console.print(f"[bold blue]Max Depth: {max_depth}, Max Pages: {max_pages}[/]")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
        task = progress.add_task("DFS Crawling...", total=max_pages)
        
        async def crawl_with_progress():
            results = await dfs_crawl_demo(start_url, max_depth, max_pages)
            progress.update(task, completed=len(results))
            return results
        
        results = asyncio.run(crawl_with_progress())
    
    display_crawl_results(results, "DFS")
    
    if save:
        save_crawl_results(results, "dfs", "deep_crawl_results")


def run_best_first_demo(start_url: str, keywords: str, max_depth: int, max_pages: int, save: bool):
    """Core Best-First demo function without Click decorators"""
    keyword_list = [k.strip() for k in keywords.split(',') if k.strip()]
    
    console.print(f"[bold blue]Best-First Deep Crawl Demo: {start_url}[/]")
    console.print(f"[bold blue]Keywords: {keyword_list}[/]")
    console.print(f"[bold blue]Max Depth: {max_depth}, Max Pages: {max_pages}[/]")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Best-First Crawling...", total=max_pages)
        
        async def crawl_with_progress():
            results = await best_first_crawl_demo(start_url, keyword_list, max_depth, max_pages)
            progress.update(task, completed=len(results))
            return results
        
        results = asyncio.run(crawl_with_progress())
    
    display_crawl_results(results, "Best-First")
    
    if save:
        save_crawl_results(results, "best_first", "deep_crawl_results")


def run_comparison_demo(start_url: str, keywords: str, max_depth: int, max_pages: int):
    """Compare all deep crawling strategies"""
    keyword_list = [k.strip() for k in keywords.split(',') if k.strip()]
    
    console.print(f"[bold blue]Deep Crawling Strategy Comparison: {start_url}[/]")
    console.print(f"[bold blue]Keywords: {keyword_list}[/]")
    
    strategies = [
        ("BFS", lambda: bfs_crawl_demo(start_url, max_depth, max_pages)),
        ("DFS", lambda: dfs_crawl_demo(start_url, max_depth, max_pages)),
        ("Best-First", lambda: best_first_crawl_demo(start_url, keyword_list, max_depth, max_pages))
    ]
    
    all_results = {}
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        
        for strategy_name, strategy_func in strategies:
            task = progress.add_task(f"Running {strategy_name}...", total=None)
            
            results = asyncio.run(strategy_func())
            all_results[strategy_name] = results
            
            progress.remove_task(task)
    
    # Create comparison table
    comparison_table = Table(title="Strategy Comparison")
    comparison_table.add_column("Strategy", style="cyan")
    comparison_table.add_column("Pages Found", style="yellow")
    comparison_table.add_column("Success Rate", style="green")
    comparison_table.add_column("Total Content", style="blue")
    comparison_table.add_column("Unique Domains", style="magenta")
    
    for strategy_name, results in all_results.items():
        total_pages = len(results)
        successful_pages = len([r for r in results if r.success])
        success_rate = f"{(successful_pages/total_pages*100):.1f}%" if total_pages > 0 else "0%"
        total_content = sum(len(r.markdown.raw_markdown) if r.markdown and r.markdown.raw_markdown else 0 for r in results)
        unique_domains = len(set(r.url.split('/')[2] for r in results if r.url))
        
        comparison_table.add_row(
            strategy_name,
            str(total_pages),
            success_rate,
            f"{total_content:,}",
            str(unique_domains)
        )
    
    console.print(comparison_table)
    
    # Show individual results
    for strategy_name, results in all_results.items():
        console.print(f"\n[bold cyan]--- {strategy_name} Results ---[/]")
        tree = create_crawl_tree(results, strategy_name)
        console.print(tree)


@click.group()
def cli():
    """Crawl4ai Reference Application - Deep Crawling Demo"""
    pass


@cli.command("bfs")
@click.argument("start_url", required=True)
@click.option("--max-depth", "-d", default=2, type=int, help="Maximum crawl depth")
@click.option("--max-pages", "-p", default=10, type=int, help="Maximum pages to crawl")
@click.option("--include", "-i", default="", help="Include URL patterns (comma-separated)")
@click.option("--exclude", "-e", default="", help="Exclude URL patterns (comma-separated)")
@click.option("--save/--no-save", default=False, help="Save results to files")
def bfs_cmd(start_url: str, max_depth: int, max_pages: int, include: str, exclude: str, save: bool):
    """Perform Breadth-First Search deep crawling"""
    run_bfs_demo(start_url, max_depth, max_pages, include, exclude, save)


@cli.command("dfs")
@click.argument("start_url", required=True)
@click.option("--max-depth", "-d", default=2, type=int, help="Maximum crawl depth")
@click.option("--max-pages", "-p", default=10, type=int, help="Maximum pages to crawl")
@click.option("--save/--no-save", default=False, help="Save results to files")
def dfs_cmd(start_url: str, max_depth: int, max_pages: int, save: bool):
    """Perform Depth-First Search deep crawling"""
    run_dfs_demo(start_url, max_depth, max_pages, save)


@cli.command("best-first")
@click.argument("start_url", required=True)
@click.argument("keywords", required=True)
@click.option("--max-depth", "-d", default=2, type=int, help="Maximum crawl depth")
@click.option("--max-pages", "-p", default=10, type=int, help="Maximum pages to crawl")
@click.option("--save/--no-save", default=False, help="Save results to files")
def best_first_cmd(start_url: str, keywords: str, max_depth: int, max_pages: int, save: bool):
    """Perform Best-First deep crawling with keyword scoring"""
    run_best_first_demo(start_url, keywords, max_depth, max_pages, save)


@cli.command("compare")
@click.argument("start_url", required=True)
@click.option("--keywords", "-k", default="python,programming,tutorial", help="Keywords for best-first strategy")
@click.option("--max-depth", "-d", default=2, type=int, help="Maximum crawl depth")
@click.option("--max-pages", "-p", default=8, type=int, help="Maximum pages to crawl")
def compare_cmd(start_url: str, keywords: str, max_depth: int, max_pages: int):
    """Compare all deep crawling strategies"""
    run_comparison_demo(start_url, keywords, max_depth, max_pages)


@cli.command("info")
def info_cmd():
    """Show information about deep crawling strategies"""
    console.print("[bold blue]Crawl4AI Deep Crawling Strategies[/]\n")
    
    console.print("[bold yellow]Crawling Strategies:[/]")
    console.print("• [bold]BFS (Breadth-First Search)[/]: Explores all pages at current depth before going deeper")
    console.print("• [bold]DFS (Depth-First Search)[/]: Explores as far as possible along each branch before backtracking")
    console.print("• [bold]Best-First[/]: Prioritizes pages based on relevance scoring (keywords, domain authority, etc.)")
    
    console.print("\n[bold yellow]Filtering Options:[/]")
    console.print("• [bold]URL Pattern Filters[/]: Include/exclude based on URL patterns")
    console.print("• [bold]Domain Filters[/]: Restrict crawling to specific domains")
    console.print("• [bold]Content Type Filters[/]: Filter by MIME type (HTML, PDF, etc.)")
    
    console.print("\n[bold yellow]Scoring Methods:[/]")
    console.print("• [bold]Keyword Relevance[/]: Score pages based on keyword matching")
    console.print("• [bold]Path Depth[/]: Prefer pages closer to the root")
    console.print("• [bold]Composite Scoring[/]: Combine multiple scoring methods with weights")
    
    console.print("\n[bold green]When to Use Each Strategy:[/]")
    console.print("• [bold]BFS[/]: Site mapping, finding all pages at specific depths")
    console.print("• [bold]DFS[/]: Following specific paths, deep exploration of branches")
    console.print("• [bold]Best-First[/]: Finding most relevant content, research-focused crawling")
    
    console.print("\n[bold green]Usage Examples:[/]")
    console.print("  python deep_crawling_demo.py bfs https://example.com --max-depth 2")
    console.print("  python deep_crawling_demo.py best-first https://docs.python.org 'python,tutorial'")
    console.print("  python deep_crawling_demo.py compare https://example.com")


if __name__ == "__main__":
    try:
        cli()
    except Exception as e:
        console.print(f"[bold red]Error: {str(e)}[/]")
        sys.exit(1)