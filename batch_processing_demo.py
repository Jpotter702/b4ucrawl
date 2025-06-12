#!/usr/bin/env python3
"""
Crawl4ai Reference Application - Batch Processing Demo

This script demonstrates batch processing capabilities in Crawl4ai:
- Multiple URL handling with different strategies
- Request dispatchers and queue management
- Rate limiting and throttling
- Parallel processing with concurrency control
- Batch result aggregation and analysis
"""

import asyncio
import json
import os
import sys
import time
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse
from dataclasses import dataclass

import click
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
from rich.live import Live
from rich.layout import Layout

from crawl4ai import (
    AsyncWebCrawler,
    BrowserConfig,
    CrawlerRunConfig,
    CrawlResult,
    # Rate limiting and concurrency
    RequestThrottler,
    ConcurrencyManager,
    # Dispatchers
    URLDispatcher,
    DomainBasedDispatcher,
    PriorityDispatcher,
    RoundRobinDispatcher,
    # Queue management
    CrawlQueue,
    PriorityQueue,
    URLBatch
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

@dataclass
class BatchConfig:
    """Configuration for batch processing operations."""
    max_concurrent: int = 5
    rate_limit: float = 1.0  # Requests per second
    retry_attempts: int = 3
    timeout: int = 30
    respect_robots_txt: bool = True
    delay_between_batches: float = 2.0


class BatchProcessor:
    """Handles batch processing of multiple URLs with advanced features."""
    
    def __init__(self, config: BatchConfig = None):
        self.config = config or BatchConfig()
        self.results: List[CrawlResult] = []
        self.errors: List[Dict[str, Any]] = []
        self.start_time: Optional[float] = None
        self.throttler = RequestThrottler(rate_limit=self.config.rate_limit)
        self.concurrency_manager = ConcurrencyManager(max_concurrent=self.config.max_concurrent)
    
    async def process_urls(self, urls: List[str], dispatcher_type: str = "round_robin") -> List[CrawlResult]:
        """Process multiple URLs using specified dispatcher strategy."""
        self.start_time = time.time()
        self.results = []
        self.errors = []
        
        # Create URL batches
        batches = self._create_batches(urls, dispatcher_type)
        
        # Process each batch
        for i, batch in enumerate(batches):
            console.print(f"[bold yellow]Processing batch {i+1}/{len(batches)} ({len(batch.urls)} URLs)[/]")
            
            batch_results = await self._process_batch(batch)
            self.results.extend(batch_results)
            
            # Delay between batches if configured
            if i < len(batches) - 1 and self.config.delay_between_batches > 0:
                await asyncio.sleep(self.config.delay_between_batches)
        
        return self.results
    
    def _create_batches(self, urls: List[str], dispatcher_type: str) -> List[URLBatch]:
        """Create URL batches using specified dispatcher strategy."""
        
        if dispatcher_type == "domain":
            dispatcher = DomainBasedDispatcher()
        elif dispatcher_type == "priority":
            # Assign priorities based on URL characteristics
            dispatcher = PriorityDispatcher()
        else:  # round_robin (default)
            dispatcher = RoundRobinDispatcher(batch_size=self.config.max_concurrent)
        
        return dispatcher.create_batches(urls)
    
    async def _process_batch(self, batch: URLBatch) -> List[CrawlResult]:
        """Process a single batch of URLs with concurrency control."""
        browser_cfg = BrowserConfig(**DEFAULT_BROWSER_CONFIG)
        crawler_cfg = CrawlerRunConfig(**DEFAULT_CRAWLER_CONFIG)
        
        # Initialize crawler
        crawler = AsyncWebCrawler(config=browser_cfg)
        await crawler.start()
        
        try:
            # Create semaphore for concurrency control
            semaphore = asyncio.Semaphore(self.config.max_concurrent)
            
            # Process URLs concurrently within the batch
            tasks = [
                self._process_single_url(crawler, url, crawler_cfg, semaphore)
                for url in batch.urls
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Separate successful results from errors
            batch_results = []
            for url, result in zip(batch.urls, results):
                if isinstance(result, Exception):
                    self.errors.append({
                        "url": url,
                        "error": str(result),
                        "timestamp": datetime.now().isoformat()
                    })
                else:
                    batch_results.append(result)
            
            return batch_results
            
        finally:
            await crawler.close()
    
    async def _process_single_url(self, crawler: AsyncWebCrawler, url: str, 
                                 config: CrawlerRunConfig, semaphore: asyncio.Semaphore) -> CrawlResult:
        """Process a single URL with rate limiting and error handling."""
        async with semaphore:
            # Apply rate limiting
            await self.throttler.wait()
            
            # Retry logic
            for attempt in range(self.config.retry_attempts):
                try:
                    result = await asyncio.wait_for(
                        crawler.arun(url=url, config=config),
                        timeout=self.config.timeout
                    )
                    return result
                    
                except asyncio.TimeoutError:
                    if attempt == self.config.retry_attempts - 1:
                        raise Exception(f"Timeout after {self.config.retry_attempts} attempts")
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    
                except Exception as e:
                    if attempt == self.config.retry_attempts - 1:
                        raise e
                    await asyncio.sleep(2 ** attempt)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics."""
        if not self.start_time:
            return {}
        
        elapsed_time = time.time() - self.start_time
        total_urls = len(self.results) + len(self.errors)
        successful_urls = len(self.results)
        
        # Calculate content statistics
        total_content = sum(
            len(result.markdown.raw_markdown) 
            if result.markdown and result.markdown.raw_markdown else 0
            for result in self.results
        )
        
        # Domain statistics
        domains = {}
        for result in self.results:
            domain = urlparse(result.url).netloc
            domains[domain] = domains.get(domain, 0) + 1
        
        return {
            "total_urls": total_urls,
            "successful_urls": successful_urls,
            "failed_urls": len(self.errors),
            "success_rate": (successful_urls / total_urls * 100) if total_urls > 0 else 0,
            "elapsed_time": elapsed_time,
            "urls_per_second": total_urls / elapsed_time if elapsed_time > 0 else 0,
            "total_content_chars": total_content,
            "avg_content_per_url": total_content / successful_urls if successful_urls > 0 else 0,
            "unique_domains": len(domains),
            "domain_distribution": domains
        }


async def simple_batch_demo(urls: List[str], max_concurrent: int = 3) -> List[CrawlResult]:
    """
    Simple batch processing demo with basic concurrency.
    
    Args:
        urls: List of URLs to process
        max_concurrent: Maximum concurrent requests
        
    Returns:
        List of crawl results
    """
    config = BatchConfig(max_concurrent=max_concurrent, rate_limit=2.0)
    processor = BatchProcessor(config)
    
    return await processor.process_urls(urls, "round_robin")


async def domain_based_batch_demo(urls: List[str]) -> List[CrawlResult]:
    """
    Domain-based batch processing demo.
    Groups URLs by domain for efficient processing.
    
    Args:
        urls: List of URLs to process
        
    Returns:
        List of crawl results
    """
    config = BatchConfig(max_concurrent=4, rate_limit=1.5, delay_between_batches=3.0)
    processor = BatchProcessor(config)
    
    return await processor.process_urls(urls, "domain")


async def priority_batch_demo(urls: List[str]) -> List[CrawlResult]:
    """
    Priority-based batch processing demo.
    Processes URLs based on assigned priorities.
    
    Args:
        urls: List of URLs to process
        
    Returns:
        List of crawl results
    """
    config = BatchConfig(max_concurrent=3, rate_limit=1.0, retry_attempts=2)
    processor = BatchProcessor(config)
    
    return await processor.process_urls(urls, "priority")


async def rate_limited_batch_demo(urls: List[str], rate_limit: float = 0.5) -> List[CrawlResult]:
    """
    Rate-limited batch processing demo.
    Demonstrates careful rate limiting for sensitive sites.
    
    Args:
        urls: List of URLs to process
        rate_limit: Requests per second limit
        
    Returns:
        List of crawl results
    """
    config = BatchConfig(
        max_concurrent=2, 
        rate_limit=rate_limit, 
        retry_attempts=3,
        delay_between_batches=5.0,
        respect_robots_txt=True
    )
    processor = BatchProcessor(config)
    
    return await processor.process_urls(urls, "round_robin")


def load_urls_from_file(filepath: str) -> List[str]:
    """Load URLs from a text file (one per line)."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            urls = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        return urls
    except FileNotFoundError:
        console.print(f"[bold red]Error: File {filepath} not found[/]")
        return []


def save_batch_results(results: List[CrawlResult], errors: List[Dict], 
                      statistics: Dict, output_dir: str = "batch_results"):
    """Save batch processing results to files."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save summary
    summary = {
        "timestamp": timestamp,
        "statistics": statistics,
        "results_count": len(results),
        "errors_count": len(errors),
        "results": [],
        "errors": errors
    }
    
    # Process results
    for i, result in enumerate(results):
        result_data = {
            "index": i,
            "url": result.url,
            "success": result.success,
            "content_length": len(result.markdown.raw_markdown) if result.markdown and result.markdown.raw_markdown else 0,
            "links_found": len(result.links.internal) + len(result.links.external) if result.links else 0,
            "processing_time": getattr(result, 'processing_time', None),
            "domain": urlparse(result.url).netloc
        }
        summary["results"].append(result_data)
        
        # Save content if successful
        if result.success and result.markdown and result.markdown.raw_markdown:
            safe_filename = f"batch_{timestamp}_result_{i:03d}.md"
            content_file = Path(output_dir) / safe_filename
            with open(content_file, 'w', encoding='utf-8') as f:
                f.write(f"# {result.url}\n\n")
                f.write(f"**Domain:** {urlparse(result.url).netloc}\n")
                f.write(f"**Content Length:** {len(result.markdown.raw_markdown):,} characters\n\n")
                f.write("---\n\n")
                f.write(result.markdown.raw_markdown)
    
    # Save summary JSON
    summary_file = Path(output_dir) / f"batch_{timestamp}_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    console.print(f"[bold green]Batch results saved to {output_dir}/[/]")


def display_batch_results(results: List[CrawlResult], errors: List[Dict], statistics: Dict):
    """Display comprehensive batch processing results."""
    
    # Statistics table
    stats_table = Table(title="Batch Processing Statistics")
    stats_table.add_column("Metric", style="cyan")
    stats_table.add_column("Value", style="yellow")
    
    stats_table.add_row("Total URLs", str(statistics.get("total_urls", 0)))
    stats_table.add_row("Successful", str(statistics.get("successful_urls", 0)))
    stats_table.add_row("Failed", str(statistics.get("failed_urls", 0)))
    stats_table.add_row("Success Rate", f"{statistics.get('success_rate', 0):.1f}%")
    stats_table.add_row("Processing Time", f"{statistics.get('elapsed_time', 0):.2f} seconds")
    stats_table.add_row("URLs/Second", f"{statistics.get('urls_per_second', 0):.2f}")
    stats_table.add_row("Total Content", f"{statistics.get('total_content_chars', 0):,} chars")
    stats_table.add_row("Avg Content/URL", f"{statistics.get('avg_content_per_url', 0):,.0f} chars")
    stats_table.add_row("Unique Domains", str(statistics.get("unique_domains", 0)))
    
    console.print(stats_table)
    
    # Domain distribution
    if statistics.get("domain_distribution"):
        domain_table = Table(title="Domain Distribution")
        domain_table.add_column("Domain", style="cyan")
        domain_table.add_column("URLs Processed", style="yellow")
        domain_table.add_column("Percentage", style="green")
        
        total_processed = statistics.get("successful_urls", 0)
        for domain, count in sorted(statistics["domain_distribution"].items(), 
                                  key=lambda x: x[1], reverse=True)[:10]:
            percentage = (count / total_processed * 100) if total_processed > 0 else 0
            domain_table.add_row(domain, str(count), f"{percentage:.1f}%")
        
        console.print(domain_table)
    
    # Results table (top 10)
    if results:
        results_table = Table(title="Top 10 Results by Content Length")
        results_table.add_column("URL", style="cyan")
        results_table.add_column("Domain", style="blue")
        results_table.add_column("Content Length", style="yellow")
        results_table.add_column("Links Found", style="green")
        
        # Sort by content length
        sorted_results = sorted(results, 
                              key=lambda r: len(r.markdown.raw_markdown) if r.markdown and r.markdown.raw_markdown else 0, 
                              reverse=True)
        
        for result in sorted_results[:10]:
            url_display = result.url if len(result.url) <= 50 else result.url[:47] + "..."
            domain = urlparse(result.url).netloc
            content_len = len(result.markdown.raw_markdown) if result.markdown and result.markdown.raw_markdown else 0
            links_count = len(result.links.internal) + len(result.links.external) if result.links else 0
            
            results_table.add_row(
                url_display,
                domain,
                f"{content_len:,}",
                str(links_count)
            )
        
        console.print(results_table)
    
    # Errors summary
    if errors:
        error_table = Table(title="Processing Errors")
        error_table.add_column("URL", style="red")
        error_table.add_column("Error", style="yellow")
        
        for error in errors[:5]:  # Show first 5 errors
            url_display = error["url"] if len(error["url"]) <= 50 else error["url"][:47] + "..."
            error_msg = error["error"][:100] + "..." if len(error["error"]) > 100 else error["error"]
            error_table.add_row(url_display, error_msg)
        
        if len(errors) > 5:
            error_table.add_row("...", f"and {len(errors) - 5} more errors")
        
        console.print(error_table)


# CLI Commands

def run_simple_batch_demo(urls_input: str, max_concurrent: int, save: bool):
    """Core simple batch demo function without Click decorators"""
    urls = parse_urls_input(urls_input)
    if not urls:
        return
    
    console.print(f"[bold blue]Simple Batch Processing Demo[/]")
    console.print(f"[bold blue]URLs: {len(urls)}, Max Concurrent: {max_concurrent}[/]")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Processing URLs...", total=len(urls))
        
        async def process_with_progress():
            processor = BatchProcessor(BatchConfig(max_concurrent=max_concurrent, rate_limit=2.0))
            results = await processor.process_urls(urls, "round_robin")
            progress.update(task, completed=len(urls))
            return results, processor.errors, processor.get_statistics()
        
        results, errors, stats = asyncio.run(process_with_progress())
    
    display_batch_results(results, errors, stats)
    
    if save:
        save_batch_results(results, errors, stats, "simple_batch_results")


def run_domain_batch_demo(urls_input: str, save: bool):
    """Core domain-based batch demo function without Click decorators"""
    urls = parse_urls_input(urls_input)
    if not urls:
        return
    
    console.print(f"[bold blue]Domain-Based Batch Processing Demo[/]")
    console.print(f"[bold blue]URLs: {len(urls)}, Strategy: Domain-Based Grouping[/]")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Processing URLs by domain...", total=None)
        
        results = asyncio.run(domain_based_batch_demo(urls))
        progress.remove_task(task)
    
    # Get statistics (simplified for demo)
    processor = BatchProcessor()
    processor.results = results
    stats = processor.get_statistics()
    
    display_batch_results(results, [], stats)
    
    if save:
        save_batch_results(results, [], stats, "domain_batch_results")


def run_rate_limited_demo(urls_input: str, rate_limit: float, save: bool):
    """Core rate-limited demo function without Click decorators"""
    urls = parse_urls_input(urls_input)
    if not urls:
        return
    
    console.print(f"[bold blue]Rate-Limited Batch Processing Demo[/]")
    console.print(f"[bold blue]URLs: {len(urls)}, Rate Limit: {rate_limit} req/sec[/]")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Processing with rate limiting...", total=None)
        
        results = asyncio.run(rate_limited_batch_demo(urls, rate_limit))
        progress.remove_task(task)
    
    # Get statistics
    processor = BatchProcessor()
    processor.results = results
    stats = processor.get_statistics()
    
    display_batch_results(results, [], stats)
    
    if save:
        save_batch_results(results, [], stats, "rate_limited_results")


def run_comparison_demo(urls_input: str):
    """Compare different batch processing strategies"""
    urls = parse_urls_input(urls_input)
    if not urls:
        return
    
    # Limit URLs for comparison demo
    if len(urls) > 8:
        urls = urls[:8]
        console.print(f"[yellow]Limited to first 8 URLs for comparison demo[/]")
    
    console.print(f"[bold blue]Batch Processing Strategy Comparison[/]")
    console.print(f"[bold blue]URLs: {len(urls)}[/]")
    
    strategies = [
        ("Simple (Concurrent=3)", lambda: simple_batch_demo(urls, 3)),
        ("Domain-Based", lambda: domain_based_batch_demo(urls)),
        ("Rate-Limited (0.5/sec)", lambda: rate_limited_batch_demo(urls, 0.5))
    ]
    
    all_results = {}
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        
        for strategy_name, strategy_func in strategies:
            task = progress.add_task(f"Running {strategy_name}...", total=None)
            
            start_time = time.time()
            results = asyncio.run(strategy_func())
            elapsed_time = time.time() - start_time
            
            all_results[strategy_name] = {
                "results": results,
                "elapsed_time": elapsed_time,
                "success_count": len([r for r in results if r.success])
            }
            
            progress.remove_task(task)
    
    # Create comparison table
    comparison_table = Table(title="Strategy Comparison")
    comparison_table.add_column("Strategy", style="cyan")
    comparison_table.add_column("Processing Time", style="yellow")
    comparison_table.add_column("Successful URLs", style="green")
    comparison_table.add_column("URLs/Second", style="blue")
    comparison_table.add_column("Total Content", style="magenta")
    
    for strategy_name, data in all_results.items():
        results = data["results"]
        elapsed_time = data["elapsed_time"]
        success_count = data["success_count"]
        urls_per_sec = len(results) / elapsed_time if elapsed_time > 0 else 0
        total_content = sum(
            len(r.markdown.raw_markdown) if r.markdown and r.markdown.raw_markdown else 0
            for r in results
        )
        
        comparison_table.add_row(
            strategy_name,
            f"{elapsed_time:.2f}s",
            str(success_count),
            f"{urls_per_sec:.2f}",
            f"{total_content:,}"
        )
    
    console.print(comparison_table)


def parse_urls_input(urls_input: str) -> List[str]:
    """Parse URLs from input (comma-separated string or file path)."""
    if urls_input.endswith('.txt'):
        urls = load_urls_from_file(urls_input)
    else:
        urls = [url.strip() for url in urls_input.split(',') if url.strip()]
    
    if not urls:
        console.print("[bold red]Error: No URLs provided or file not found[/]")
        return []
    
    return urls


@click.group()
def cli():
    """Crawl4ai Reference Application - Batch Processing Demo"""
    pass


@cli.command("simple")
@click.argument("urls", required=True)
@click.option("--max-concurrent", "-c", default=3, type=int, help="Maximum concurrent requests")
@click.option("--save/--no-save", default=False, help="Save results to files")
def simple_cmd(urls: str, max_concurrent: int, save: bool):
    """Simple batch processing with basic concurrency"""
    run_simple_batch_demo(urls, max_concurrent, save)


@cli.command("domain")
@click.argument("urls", required=True)
@click.option("--save/--no-save", default=False, help="Save results to files")
def domain_cmd(urls: str, save: bool):
    """Domain-based batch processing"""
    run_domain_batch_demo(urls, save)


@cli.command("rate-limited")
@click.argument("urls", required=True)
@click.option("--rate", "-r", default=0.5, type=float, help="Rate limit (requests per second)")
@click.option("--save/--no-save", default=False, help="Save results to files")
def rate_limited_cmd(urls: str, rate: float, save: bool):
    """Rate-limited batch processing"""
    run_rate_limited_demo(urls, rate, save)


@cli.command("compare")
@click.argument("urls", required=True)
def compare_cmd(urls: str):
    """Compare different batch processing strategies"""
    run_comparison_demo(urls)


@cli.command("info")
def info_cmd():
    """Show information about batch processing capabilities"""
    console.print("[bold blue]Crawl4AI Batch Processing Capabilities[/]\n")
    
    console.print("[bold yellow]Processing Strategies:[/]")
    console.print("• [bold]Simple Concurrent[/]: Basic parallel processing with concurrency limits")
    console.print("• [bold]Domain-Based[/]: Groups URLs by domain for efficient processing")
    console.print("• [bold]Priority-Based[/]: Processes URLs based on assigned priorities")
    console.print("• [bold]Rate-Limited[/]: Careful rate limiting for sensitive websites")
    
    console.print("\n[bold yellow]Key Features:[/]")
    console.print("• [bold]Request Throttling[/]: Control request rate to avoid overwhelming servers")
    console.print("• [bold]Concurrency Management[/]: Limit simultaneous connections")
    console.print("• [bold]Retry Logic[/]: Automatic retry with exponential backoff")
    console.print("• [bold]Error Handling[/]: Comprehensive error tracking and reporting")
    console.print("• [bold]Progress Tracking[/]: Real-time progress updates")
    
    console.print("\n[bold yellow]Dispatchers:[/]")
    console.print("• [bold]Round Robin[/]: Distributes URLs evenly across batches")
    console.print("• [bold]Domain-Based[/]: Groups URLs by domain for better cache utilization")
    console.print("• [bold]Priority-Based[/]: Processes high-priority URLs first")
    
    console.print("\n[bold green]Best Practices:[/]")
    console.print("• Use domain-based grouping for better performance")
    console.print("• Apply rate limiting when crawling sensitive sites")
    console.print("• Monitor robots.txt compliance")
    console.print("• Use appropriate retry strategies for different error types")
    console.print("• Save results incrementally for large batches")
    
    console.print("\n[bold green]Usage Examples:[/]")
    console.print("  python batch_processing_demo.py simple 'https://site1.com,https://site2.com'")
    console.print("  python batch_processing_demo.py domain urls.txt --save")
    console.print("  python batch_processing_demo.py rate-limited urls.txt --rate 0.2")
    console.print("  python batch_processing_demo.py compare 'https://example1.com,https://example2.com'")


if __name__ == "__main__":
    try:
        cli()
    except Exception as e:
        console.print(f"[bold red]Error: {str(e)}[/]")
        sys.exit(1)