#!/usr/bin/env python3
"""
Crawl4ai Reference Application - Proxy Demo

This script demonstrates proxy usage and rotation strategies in Crawl4ai:
- HTTP/HTTPS proxy configuration
- SOCKS proxy support
- Proxy rotation strategies
- Proxy health checking and failover
- Geographic proxy selection
- Anonymous proxy usage
"""

import asyncio
import json
import os
import sys
import time
import random
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass
from urllib.parse import urlparse

import click
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.live import Live

from crawl4ai import (
    AsyncWebCrawler,
    BrowserConfig,
    CrawlerRunConfig,
    CrawlResult,
    # Proxy classes
    ProxyConfig,
    ProxyRotator,
    ProxyHealthChecker,
    ProxyPool,
    # Proxy strategies
    RoundRobinProxyStrategy,
    RandomProxyStrategy,
    GeographicProxyStrategy,
    FailoverProxyStrategy,
    # Proxy types
    HTTPProxy,
    HTTPSProxy,
    SOCKSProxy
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

@dataclass
class ProxyInfo:
    """Information about a proxy server."""
    host: str
    port: int
    proxy_type: str = "http"  # http, https, socks4, socks5
    username: Optional[str] = None
    password: Optional[str] = None
    country: Optional[str] = None
    region: Optional[str] = None
    is_anonymous: bool = True
    response_time: Optional[float] = None
    last_checked: Optional[datetime] = None
    is_working: bool = True
    
    def to_url(self) -> str:
        """Convert proxy info to URL format."""
        if self.username and self.password:
            auth = f"{self.username}:{self.password}@"
        else:
            auth = ""
        
        return f"{self.proxy_type}://{auth}{self.host}:{self.port}"
    
    def __str__(self) -> str:
        return f"{self.proxy_type}://{self.host}:{self.port}"


class ProxyManager:
    """Manages proxy pools and rotation strategies."""
    
    def __init__(self):
        self.proxy_pool: List[ProxyInfo] = []
        self.health_checker = ProxyHealthChecker()
        self.rotation_strategy = "round_robin"
        self.current_index = 0
        self.failed_proxies: List[ProxyInfo] = []
        
    def add_proxy(self, proxy: ProxyInfo):
        """Add a proxy to the pool."""
        self.proxy_pool.append(proxy)
    
    def add_proxies_from_list(self, proxies: List[Dict[str, Any]]):
        """Add multiple proxies from a list of dictionaries."""
        for proxy_data in proxies:
            proxy = ProxyInfo(**proxy_data)
            self.add_proxy(proxy)
    
    def load_proxies_from_file(self, filepath: str):
        """Load proxies from a JSON file."""
        try:
            with open(filepath, 'r') as f:
                proxy_data = json.load(f)
                
            if isinstance(proxy_data, list):
                self.add_proxies_from_list(proxy_data)
            elif isinstance(proxy_data, dict) and "proxies" in proxy_data:
                self.add_proxies_from_list(proxy_data["proxies"])
            else:
                console.print(f"[bold red]Invalid proxy file format: {filepath}[/]")
                
        except FileNotFoundError:
            console.print(f"[bold red]Proxy file not found: {filepath}[/]")
        except json.JSONDecodeError:
            console.print(f"[bold red]Invalid JSON in proxy file: {filepath}[/]")
    
    async def check_proxy_health(self, proxy: ProxyInfo, test_url: str = "http://httpbin.org/ip") -> bool:
        """Check if a proxy is working."""
        try:
            start_time = time.time()
            
            # Create browser config with proxy
            browser_config = DEFAULT_BROWSER_CONFIG.copy()
            browser_config["proxy"] = proxy.to_url()
            
            browser_cfg = BrowserConfig(**browser_config)
            crawler_cfg = CrawlerRunConfig(**DEFAULT_CRAWLER_CONFIG)
            
            # Test the proxy
            crawler = AsyncWebCrawler(config=browser_cfg)
            await crawler.start()
            
            try:
                result = await asyncio.wait_for(
                    crawler.arun(url=test_url, config=crawler_cfg),
                    timeout=10
                )
                
                proxy.response_time = time.time() - start_time
                proxy.last_checked = datetime.now()
                proxy.is_working = result.success
                
                return result.success
                
            finally:
                await crawler.close()
                
        except Exception as e:
            proxy.is_working = False
            proxy.last_checked = datetime.now()
            return False
    
    async def health_check_all_proxies(self, test_url: str = "http://httpbin.org/ip"):
        """Check health of all proxies in the pool."""
        console.print(f"[bold yellow]Checking health of {len(self.proxy_pool)} proxies...[/]")
        
        tasks = [
            self.check_proxy_health(proxy, test_url)
            for proxy in self.proxy_pool
        ]
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:
            task = progress.add_task("Testing proxies...", total=len(tasks))
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            working_count = sum(1 for result in results if result is True)
            progress.update(task, completed=len(tasks))
            
            console.print(f"[bold green]{working_count}/{len(self.proxy_pool)} proxies are working[/]")
    
    def get_next_proxy(self) -> Optional[ProxyInfo]:
        """Get the next proxy based on rotation strategy."""
        working_proxies = [p for p in self.proxy_pool if p.is_working]
        
        if not working_proxies:
            return None
        
        if self.rotation_strategy == "round_robin":
            proxy = working_proxies[self.current_index % len(working_proxies)]
            self.current_index += 1
            return proxy
            
        elif self.rotation_strategy == "random":
            return random.choice(working_proxies)
            
        elif self.rotation_strategy == "fastest":
            # Sort by response time
            fastest_proxies = sorted(
                [p for p in working_proxies if p.response_time is not None],
                key=lambda x: x.response_time
            )
            return fastest_proxies[0] if fastest_proxies else working_proxies[0]
            
        elif self.rotation_strategy == "geographic":
            # Prefer proxies from specific regions (for demo, prefer US)
            us_proxies = [p for p in working_proxies if p.country == "US"]
            if us_proxies:
                return random.choice(us_proxies)
            return random.choice(working_proxies)
        
        return working_proxies[0]
    
    def mark_proxy_failed(self, proxy: ProxyInfo):
        """Mark a proxy as failed and move to failed list."""
        proxy.is_working = False
        if proxy not in self.failed_proxies:
            self.failed_proxies.append(proxy)
    
    def get_proxy_statistics(self) -> Dict[str, Any]:
        """Get statistics about the proxy pool."""
        total_proxies = len(self.proxy_pool)
        working_proxies = len([p for p in self.proxy_pool if p.is_working])
        failed_proxies = len(self.failed_proxies)
        
        # Response time statistics
        response_times = [p.response_time for p in self.proxy_pool 
                         if p.response_time is not None and p.is_working]
        
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        
        # Geographic distribution
        countries = {}
        for proxy in self.proxy_pool:
            if proxy.country:
                countries[proxy.country] = countries.get(proxy.country, 0) + 1
        
        # Proxy types
        types = {}
        for proxy in self.proxy_pool:
            types[proxy.proxy_type] = types.get(proxy.proxy_type, 0) + 1
        
        return {
            "total_proxies": total_proxies,
            "working_proxies": working_proxies,
            "failed_proxies": failed_proxies,
            "success_rate": (working_proxies / total_proxies * 100) if total_proxies > 0 else 0,
            "avg_response_time": avg_response_time,
            "countries": countries,
            "proxy_types": types
        }


async def crawl_with_proxy(url: str, proxy: ProxyInfo) -> Tuple[CrawlResult, bool]:
    """
    Crawl a URL using a specific proxy.
    
    Args:
        url: The URL to crawl
        proxy: The proxy to use
        
    Returns:
        Tuple of (CrawlResult, success_flag)
    """
    browser_config = DEFAULT_BROWSER_CONFIG.copy()
    browser_config["proxy"] = proxy.to_url()
    
    browser_cfg = BrowserConfig(**browser_config)
    crawler_cfg = CrawlerRunConfig(**DEFAULT_CRAWLER_CONFIG)
    
    crawler = AsyncWebCrawler(config=browser_cfg)
    await crawler.start()
    
    try:
        result = await asyncio.wait_for(
            crawler.arun(url=url, config=crawler_cfg),
            timeout=30
        )
        return result, result.success
        
    except Exception as e:
        # Create a failed result
        failed_result = CrawlResult(
            url=url,
            success=False,
            error_message=str(e)
        )
        return failed_result, False
        
    finally:
        await crawler.close()


async def proxy_rotation_demo(urls: List[str], proxy_manager: ProxyManager) -> List[Dict[str, Any]]:
    """
    Demonstrate proxy rotation across multiple URLs.
    
    Args:
        urls: List of URLs to crawl
        proxy_manager: Configured proxy manager
        
    Returns:
        List of crawl results with proxy information
    """
    results = []
    
    for i, url in enumerate(urls):
        proxy = proxy_manager.get_next_proxy()
        if not proxy:
            console.print("[bold red]No working proxies available[/]")
            break
            
        console.print(f"[bold yellow]Crawling {url} via {proxy}[/]")
        
        result, success = await crawl_with_proxy(url, proxy)
        
        result_info = {
            "url": url,
            "proxy": str(proxy),
            "proxy_country": proxy.country,
            "success": success,
            "error": result.error_message if not success else None,
            "content_length": len(result.markdown.raw_markdown) if result.markdown and result.markdown.raw_markdown else 0,
            "response_time": proxy.response_time
        }
        
        results.append(result_info)
        
        if not success:
            proxy_manager.mark_proxy_failed(proxy)
            console.print(f"[bold red]Proxy {proxy} failed, marking as unavailable[/]")
        
        # Small delay between requests
        await asyncio.sleep(1)
    
    return results


async def failover_demo(url: str, proxy_manager: ProxyManager) -> Dict[str, Any]:
    """
    Demonstrate proxy failover when proxies fail.
    
    Args:
        url: URL to crawl
        proxy_manager: Configured proxy manager
        
    Returns:
        Result information with failover details
    """
    attempts = []
    max_attempts = min(5, len(proxy_manager.proxy_pool))
    
    for attempt in range(max_attempts):
        proxy = proxy_manager.get_next_proxy()
        if not proxy:
            break
            
        console.print(f"[bold yellow]Attempt {attempt + 1}: Using proxy {proxy}[/]")
        
        result, success = await crawl_with_proxy(url, proxy)
        
        attempt_info = {
            "attempt": attempt + 1,
            "proxy": str(proxy),
            "success": success,
            "error": result.error_message if not success else None
        }
        attempts.append(attempt_info)
        
        if success:
            console.print(f"[bold green]Success on attempt {attempt + 1}![/]")
            return {
                "url": url,
                "attempts": attempts,
                "final_success": True,
                "working_proxy": str(proxy),
                "content_length": len(result.markdown.raw_markdown) if result.markdown and result.markdown.raw_markdown else 0
            }
        else:
            proxy_manager.mark_proxy_failed(proxy)
            console.print(f"[bold red]Failed with proxy {proxy}[/]")
    
    return {
        "url": url,
        "attempts": attempts,
        "final_success": False,
        "error": "All proxies failed"
    }


def create_sample_proxy_pool() -> List[Dict[str, Any]]:
    """Create a sample proxy pool for demonstration."""
    return [
        {
            "host": "proxy1.example.com",
            "port": 8080,
            "proxy_type": "http",
            "country": "US",
            "region": "East Coast",
            "is_anonymous": True
        },
        {
            "host": "proxy2.example.com", 
            "port": 3128,
            "proxy_type": "http",
            "username": "user1",
            "password": "pass1",
            "country": "UK",
            "region": "London",
            "is_anonymous": False
        },
        {
            "host": "proxy3.example.com",
            "port": 1080,
            "proxy_type": "socks5",
            "country": "DE",
            "region": "Frankfurt",
            "is_anonymous": True
        },
        {
            "host": "proxy4.example.com",
            "port": 8888,
            "proxy_type": "https",
            "country": "JP",
            "region": "Tokyo",
            "is_anonymous": True
        },
        {
            "host": "proxy5.example.com",
            "port": 9050,
            "proxy_type": "socks5",
            "country": "CA",
            "region": "Toronto",
            "is_anonymous": True
        }
    ]


def save_proxy_results(results: List[Dict], proxy_stats: Dict, output_dir: str = "proxy_results"):
    """Save proxy testing results to files."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save detailed results
    results_file = Path(output_dir) / f"proxy_results_{timestamp}.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({
            "timestamp": timestamp,
            "proxy_statistics": proxy_stats,
            "crawl_results": results
        }, f, indent=2)
    
    # Save proxy statistics
    stats_file = Path(output_dir) / f"proxy_stats_{timestamp}.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(proxy_stats, f, indent=2)
    
    console.print(f"[bold green]Proxy results saved to {output_dir}/[/]")


def display_proxy_results(results: List[Dict], proxy_stats: Dict):
    """Display comprehensive proxy testing results."""
    
    # Proxy statistics table
    stats_table = Table(title="Proxy Pool Statistics")
    stats_table.add_column("Metric", style="cyan")
    stats_table.add_column("Value", style="yellow")
    
    stats_table.add_row("Total Proxies", str(proxy_stats.get("total_proxies", 0)))
    stats_table.add_row("Working Proxies", str(proxy_stats.get("working_proxies", 0)))
    stats_table.add_row("Failed Proxies", str(proxy_stats.get("failed_proxies", 0)))
    stats_table.add_row("Success Rate", f"{proxy_stats.get('success_rate', 0):.1f}%")
    stats_table.add_row("Avg Response Time", f"{proxy_stats.get('avg_response_time', 0):.2f}s")
    
    console.print(stats_table)
    
    # Geographic distribution
    if proxy_stats.get("countries"):
        geo_table = Table(title="Geographic Distribution")
        geo_table.add_column("Country", style="cyan")
        geo_table.add_column("Proxy Count", style="yellow")
        
        for country, count in proxy_stats["countries"].items():
            geo_table.add_row(country, str(count))
        
        console.print(geo_table)
    
    # Proxy types
    if proxy_stats.get("proxy_types"):
        types_table = Table(title="Proxy Types")
        types_table.add_column("Type", style="cyan")
        types_table.add_column("Count", style="yellow")
        
        for proxy_type, count in proxy_stats["proxy_types"].items():
            types_table.add_row(proxy_type, str(count))
        
        console.print(types_table)
    
    # Crawl results
    if results:
        results_table = Table(title="Crawl Results")
        results_table.add_column("URL", style="cyan")
        results_table.add_column("Proxy", style="blue")
        results_table.add_column("Country", style="green")
        results_table.add_column("Success", style="yellow")
        results_table.add_column("Content Length", style="magenta")
        
        for result in results:
            url_display = result["url"] if len(result["url"]) <= 40 else result["url"][:37] + "..."
            proxy_display = result["proxy"] if len(result["proxy"]) <= 25 else result["proxy"][:22] + "..."
            success = "✓" if result["success"] else "✗"
            content_len = f"{result.get('content_length', 0):,}"
            
            results_table.add_row(
                url_display,
                proxy_display,
                result.get("proxy_country", "Unknown"),
                success,
                content_len
            )
        
        console.print(results_table)


# CLI Commands

def run_rotation_demo(urls_input: str, proxy_file: str, strategy: str, save: bool):
    """Core proxy rotation demo function without Click decorators"""
    # Parse URLs
    if urls_input.endswith('.txt'):
        try:
            with open(urls_input, 'r') as f:
                urls = [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            console.print(f"[bold red]Error: File {urls_input} not found[/]")
            return
    else:
        urls = [url.strip() for url in urls_input.split(',') if url.strip()]
    
    if not urls:
        console.print("[bold red]Error: No URLs provided[/]")
        return
    
    # Setup proxy manager
    proxy_manager = ProxyManager()
    proxy_manager.rotation_strategy = strategy
    
    if proxy_file:
        proxy_manager.load_proxies_from_file(proxy_file)
    else:
        # Use sample proxies for demo
        sample_proxies = create_sample_proxy_pool()
        proxy_manager.add_proxies_from_list(sample_proxies)
        console.print("[bold yellow]Using sample proxy pool for demonstration[/]")
    
    console.print(f"[bold blue]Proxy Rotation Demo[/]")
    console.print(f"[bold blue]URLs: {len(urls)}, Strategy: {strategy}[/]")
    console.print(f"[bold blue]Proxies: {len(proxy_manager.proxy_pool)}[/]")
    
    async def run_demo():
        # Health check proxies first
        await proxy_manager.health_check_all_proxies()
        
        # Run rotation demo
        results = await proxy_rotation_demo(urls, proxy_manager)
        
        return results, proxy_manager.get_proxy_statistics()
    
    results, proxy_stats = asyncio.run(run_demo())
    
    display_proxy_results(results, proxy_stats)
    
    if save:
        save_proxy_results(results, proxy_stats)


def run_failover_demo(url: str, proxy_file: str, save: bool):
    """Core proxy failover demo function without Click decorators"""
    # Setup proxy manager
    proxy_manager = ProxyManager()
    
    if proxy_file:
        proxy_manager.load_proxies_from_file(proxy_file)
    else:
        # Use sample proxies for demo
        sample_proxies = create_sample_proxy_pool()
        proxy_manager.add_proxies_from_list(sample_proxies)
        console.print("[bold yellow]Using sample proxy pool for demonstration[/]")
    
    console.print(f"[bold blue]Proxy Failover Demo[/]")
    console.print(f"[bold blue]URL: {url}[/]")
    console.print(f"[bold blue]Proxies: {len(proxy_manager.proxy_pool)}[/]")
    
    async def run_demo():
        # Health check proxies first
        await proxy_manager.health_check_all_proxies()
        
        # Run failover demo
        result = await failover_demo(url, proxy_manager)
        
        return result, proxy_manager.get_proxy_statistics()
    
    result, proxy_stats = asyncio.run(run_demo())
    
    # Display results
    console.print(f"\n[bold cyan]Failover Results for {url}:[/]")
    
    attempts_table = Table(title="Failover Attempts")
    attempts_table.add_column("Attempt", style="cyan")
    attempts_table.add_column("Proxy", style="blue")
    attempts_table.add_column("Success", style="green")
    attempts_table.add_column("Error", style="red")
    
    for attempt in result["attempts"]:
        success = "✓" if attempt["success"] else "✗"
        error = attempt.get("error", "")[:50] + "..." if attempt.get("error") and len(attempt.get("error", "")) > 50 else attempt.get("error", "")
        
        attempts_table.add_row(
            str(attempt["attempt"]),
            attempt["proxy"],
            success,
            error
        )
    
    console.print(attempts_table)
    
    if result["final_success"]:
        console.print(f"[bold green]✓ Final result: Success with {result['working_proxy']}[/]")
        console.print(f"[bold green]Content length: {result.get('content_length', 0):,} characters[/]")
    else:
        console.print(f"[bold red]✗ Final result: All proxies failed[/]")
    
    # Display proxy statistics
    display_proxy_results([], proxy_stats)
    
    if save:
        save_proxy_results([result], proxy_stats, "failover_results")


def run_health_check_demo(proxy_file: str, test_url: str):
    """Check health of all proxies in a file"""
    proxy_manager = ProxyManager()
    
    if proxy_file:
        proxy_manager.load_proxies_from_file(proxy_file)
    else:
        # Use sample proxies for demo
        sample_proxies = create_sample_proxy_pool()
        proxy_manager.add_proxies_from_list(sample_proxies)
        console.print("[bold yellow]Using sample proxy pool for demonstration[/]")
    
    console.print(f"[bold blue]Proxy Health Check Demo[/]")
    console.print(f"[bold blue]Test URL: {test_url}[/]")
    console.print(f"[bold blue]Proxies to test: {len(proxy_manager.proxy_pool)}[/]")
    
    async def run_health_check():
        await proxy_manager.health_check_all_proxies(test_url)
        return proxy_manager.get_proxy_statistics()
    
    proxy_stats = asyncio.run(run_health_check())
    
    # Display detailed results
    health_table = Table(title="Proxy Health Check Results")
    health_table.add_column("Proxy", style="cyan")
    health_table.add_column("Type", style="blue")
    health_table.add_column("Country", style="green")
    health_table.add_column("Status", style="yellow")
    health_table.add_column("Response Time", style="magenta")
    health_table.add_column("Last Checked", style="white")
    
    for proxy in proxy_manager.proxy_pool:
        status = "✓ Working" if proxy.is_working else "✗ Failed"
        response_time = f"{proxy.response_time:.2f}s" if proxy.response_time else "N/A"
        last_checked = proxy.last_checked.strftime("%H:%M:%S") if proxy.last_checked else "Never"
        
        health_table.add_row(
            str(proxy),
            proxy.proxy_type,
            proxy.country or "Unknown",
            status,
            response_time,
            last_checked
        )
    
    console.print(health_table)
    
    # Display statistics
    display_proxy_results([], proxy_stats)


@click.group()
def cli():
    """Crawl4ai Reference Application - Proxy Demo"""
    pass


@cli.command("rotation")
@click.argument("urls", required=True)
@click.option("--proxy-file", "-f", help="JSON file containing proxy list")
@click.option("--strategy", "-s", type=click.Choice(["round_robin", "random", "fastest", "geographic"]), 
              default="round_robin", help="Proxy rotation strategy")
@click.option("--save/--no-save", default=False, help="Save results to files")
def rotation_cmd(urls: str, proxy_file: str, strategy: str, save: bool):
    """Demonstrate proxy rotation across multiple URLs"""
    run_rotation_demo(urls, proxy_file, strategy, save)


@cli.command("failover")
@click.argument("url", required=True)
@click.option("--proxy-file", "-f", help="JSON file containing proxy list")
@click.option("--save/--no-save", default=False, help="Save results to files")
def failover_cmd(url: str, proxy_file: str, save: bool):
    """Demonstrate proxy failover when proxies fail"""
    run_failover_demo(url, proxy_file, save)


@cli.command("health-check")
@click.option("--proxy-file", "-f", help="JSON file containing proxy list")
@click.option("--test-url", "-u", default="http://httpbin.org/ip", help="URL to test proxy connectivity")
def health_check_cmd(proxy_file: str, test_url: str):
    """Check health of all proxies in the pool"""
    run_health_check_demo(proxy_file, test_url)


@cli.command("create-sample")
@click.option("--output", "-o", default="sample_proxies.json", help="Output file for sample proxy list")
def create_sample_cmd(output: str):
    """Create a sample proxy configuration file"""
    sample_proxies = {
        "proxies": create_sample_proxy_pool(),
        "description": "Sample proxy pool for Crawl4AI proxy demo",
        "created": datetime.now().isoformat()
    }
    
    with open(output, 'w', encoding='utf-8') as f:
        json.dump(sample_proxies, f, indent=2)
    
    console.print(f"[bold green]Sample proxy file created: {output}[/]")
    console.print("[bold yellow]Note: These are example proxies and will not work for actual crawling[/]")


@cli.command("info")
def info_cmd():
    """Show information about proxy capabilities"""
    console.print("[bold blue]Crawl4AI Proxy Capabilities[/]\n")
    
    console.print("[bold yellow]Supported Proxy Types:[/]")
    console.print("• [bold]HTTP Proxies[/]: Standard HTTP proxy protocol")
    console.print("• [bold]HTTPS Proxies[/]: Secure HTTP proxy with SSL/TLS")
    console.print("• [bold]SOCKS4/5 Proxies[/]: Socket-level proxy protocols")
    console.print("• [bold]Authenticated Proxies[/]: Username/password authentication")
    
    console.print("\n[bold yellow]Rotation Strategies:[/]")
    console.print("• [bold]Round Robin[/]: Cycles through proxies in order")
    console.print("• [bold]Random[/]: Selects proxies randomly")
    console.print("• [bold]Fastest[/]: Prefers proxies with lowest response time")
    console.print("• [bold]Geographic[/]: Selects proxies based on location")
    
    console.print("\n[bold yellow]Key Features:[/]")
    console.print("• [bold]Health Checking[/]: Automatic proxy health monitoring")
    console.print("• [bold]Failover Support[/]: Automatic retry with different proxies")
    console.print("• [bold]Response Time Tracking[/]: Monitor proxy performance")
    console.print("• [bold]Geographic Filtering[/]: Filter proxies by country/region")
    console.print("• [bold]Anonymous Proxy Support[/]: Use anonymous proxy pools")
    
    console.print("\n[bold green]Common Use Cases:[/]")
    console.print("• Avoiding IP-based rate limiting")
    console.print("• Geographic content access")
    console.print("• Large-scale web scraping")
    console.print("• Anonymizing web requests")
    console.print("• Load distribution across proxy pools")
    
    console.print("\n[bold green]Best Practices:[/]")
    console.print("• Regularly health check your proxy pool")
    console.print("• Use appropriate rotation strategies for your use case")
    console.print("• Respect website terms of service and robots.txt")
    console.print("• Monitor proxy performance and response times")
    console.print("• Have backup proxies for failover scenarios")
    
    console.print("\n[bold green]Usage Examples:[/]")
    console.print("  python proxy_demo.py create-sample --output my_proxies.json")
    console.print("  python proxy_demo.py health-check --proxy-file my_proxies.json")
    console.print("  python proxy_demo.py rotation 'https://site1.com,https://site2.com' --strategy random")
    console.print("  python proxy_demo.py failover https://example.com --proxy-file my_proxies.json")


if __name__ == "__main__":
    try:
        cli()
    except Exception as e:
        console.print(f"[bold red]Error: {str(e)}[/]")
        sys.exit(1)