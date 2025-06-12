#!/usr/bin/env python3
"""
Crawl4ai Reference Application - Screenshot and PDF Demo

This script demonstrates visual capture and document generation capabilities in Crawl4ai:
- Screenshot capture in various formats and configurations
- PDF generation with different settings
- Batch processing for multiple URLs
- Custom viewport and styling options
"""

import asyncio
import base64
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
from rich.progress import Progress, SpinnerColumn, TextColumn

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


async def perform_visual_crawl(url: str, screenshot: bool = False, pdf: bool = False, 
                               browser_config: Dict = None, crawler_config: Dict = None) -> CrawlResult:
    """
    Perform a web crawl with visual capture options.
    
    Args:
        url: The URL to crawl
        screenshot: Whether to capture a screenshot
        pdf: Whether to generate a PDF
        browser_config: Optional browser configuration
        crawler_config: Optional crawler configuration
        
    Returns:
        CrawlResult: The result of the crawl operation
    """
    browser_cfg_dict = browser_config or DEFAULT_BROWSER_CONFIG.copy()
    crawler_cfg_dict = crawler_config or DEFAULT_CRAWLER_CONFIG.copy()
    
    # Add visual capture options
    crawler_cfg_dict.update({
        "screenshot": screenshot,
        "pdf": pdf,
    })
    
    browser_cfg = BrowserConfig(**browser_cfg_dict)
    crawler_cfg = CrawlerRunConfig(**crawler_cfg_dict)
    
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


async def screenshot_demo(url: str, output_dir: str = "screenshots", 
                         viewport_width: int = 1280, viewport_height: int = 800,
                         wait_time: float = 2.0) -> CrawlResult:
    """
    Capture a screenshot with custom settings.
    
    Args:
        url: The URL to capture
        output_dir: Directory to save screenshots
        viewport_width: Browser viewport width
        viewport_height: Browser viewport height  
        wait_time: Time to wait before capture
        
    Returns:
        CrawlResult: The crawl result with screenshot data
    """
    browser_config = DEFAULT_BROWSER_CONFIG.copy()
    browser_config.update({
        "viewport_width": viewport_width,
        "viewport_height": viewport_height,
    })
    
    crawler_config = DEFAULT_CRAWLER_CONFIG.copy()
    crawler_config.update({
        "screenshot": True,
        "screenshot_wait_for": wait_time,
    })
    
    return await perform_visual_crawl(url, screenshot=True, browser_config=browser_config, crawler_config=crawler_config)


async def pdf_demo(url: str, output_dir: str = "pdfs") -> CrawlResult:
    """
    Generate a PDF document of the webpage.
    
    Args:
        url: The URL to convert to PDF
        output_dir: Directory to save PDFs
        
    Returns:
        CrawlResult: The crawl result with PDF data
    """
    return await perform_visual_crawl(url, pdf=True)


async def batch_visual_capture(urls: List[str], screenshot: bool = True, pdf: bool = False) -> List[CrawlResult]:
    """
    Capture screenshots or PDFs for multiple URLs.
    
    Args:
        urls: List of URLs to process
        screenshot: Whether to capture screenshots
        pdf: Whether to generate PDFs
        
    Returns:
        List of CrawlResult objects
    """
    results = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Processing URLs...", total=len(urls))
        
        for i, url in enumerate(urls):
            progress.update(task, description=f"Processing {i+1}/{len(urls)}: {url}")
            
            result = await perform_visual_crawl(url, screenshot=screenshot, pdf=pdf)
            results.append(result)
            
            progress.advance(task)
    
    return results


def save_screenshot(screenshot_data: str, filename: str, output_dir: str = "screenshots") -> str:
    """
    Save base64 encoded screenshot data to file.
    
    Args:
        screenshot_data: Base64 encoded screenshot
        filename: Output filename
        output_dir: Output directory
        
    Returns:
        Path to saved file
    """
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Decode and save
    filepath = Path(output_dir) / filename
    with open(filepath, "wb") as f:
        f.write(base64.b64decode(screenshot_data))
    
    return str(filepath)


def save_pdf(pdf_data: bytes, filename: str, output_dir: str = "pdfs") -> str:
    """
    Save PDF data to file.
    
    Args:
        pdf_data: Raw PDF bytes
        filename: Output filename
        output_dir: Output directory
        
    Returns:
        Path to saved file
    """
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save PDF
    filepath = Path(output_dir) / filename
    with open(filepath, "wb") as f:
        f.write(pdf_data)
    
    return str(filepath)


def generate_filename(url: str, extension: str) -> str:
    """Generate a safe filename from URL."""
    from urllib.parse import urlparse
    import re
    
    # Parse URL
    parsed = urlparse(url)
    domain = parsed.netloc.replace("www.", "")
    path = parsed.path.replace("/", "_").replace(".", "_")
    
    # Clean up and create filename
    safe_name = re.sub(r'[^\w\-_]', '', f"{domain}{path}")
    if not safe_name:
        safe_name = "webpage"
    
    # Add timestamp to avoid conflicts
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    return f"{safe_name}_{timestamp}.{extension}"


def display_visual_results(results: List[CrawlResult], operation: str):
    """Display results of visual capture operations."""
    
    # Create results table
    table = Table(title=f"{operation.title()} Results")
    table.add_column("URL", style="cyan")
    table.add_column("Success", style="green")
    table.add_column("Screenshot", style="yellow")
    table.add_column("PDF", style="blue")
    table.add_column("Content Length", style="magenta")
    
    for result in results:
        success = "✓" if result.success else "✗"
        has_screenshot = "✓" if result.screenshot else "✗"
        has_pdf = "✓" if result.pdf else "✗"
        content_len = len(result.markdown.raw_markdown) if result.markdown and result.markdown.raw_markdown else 0
        
        # Truncate long URLs
        display_url = result.url if len(result.url) <= 50 else result.url[:47] + "..."
        
        table.add_row(
            display_url,
            success,
            has_screenshot,
            has_pdf,
            f"{content_len:,}"
        )
    
    console.print(table)


# CLI Commands

def run_screenshot_demo(url: str, output_dir: str, width: int, height: int, wait_time: float, save: bool):
    """Core screenshot demo function without Click decorators"""
    console.print(f"[bold blue]Screenshot Demo: {url}[/]")
    console.print(f"[bold blue]Viewport: {width}x{height}, Wait: {wait_time}s[/]")
    
    result = asyncio.run(screenshot_demo(url, output_dir, width, height, wait_time))
    
    if result.success:
        console.print("[bold green]Screenshot capture successful![/]")
        
        if result.screenshot:
            console.print(f"[bold]Screenshot data size:[/] {len(result.screenshot):,} characters (base64)")
            
            if save:
                filename = generate_filename(url, "png")
                filepath = save_screenshot(result.screenshot, filename, output_dir)
                console.print(f"[bold green]Screenshot saved to:[/] {filepath}")
            else:
                console.print("[yellow]Screenshot captured but not saved (use --save to save)[/]")
        else:
            console.print("[yellow]No screenshot data available[/]")
    else:
        console.print(f"[bold red]Screenshot capture failed: {result.error_message}[/]")


def run_pdf_demo(url: str, output_dir: str, save: bool):
    """Core PDF demo function without Click decorators"""
    console.print(f"[bold blue]PDF Demo: {url}[/]")
    
    result = asyncio.run(pdf_demo(url, output_dir))
    
    if result.success:
        console.print("[bold green]PDF generation successful![/]")
        
        if result.pdf:
            console.print(f"[bold]PDF data size:[/] {len(result.pdf):,} bytes")
            
            if save:
                filename = generate_filename(url, "pdf")
                filepath = save_pdf(result.pdf, filename, output_dir)
                console.print(f"[bold green]PDF saved to:[/] {filepath}")
            else:
                console.print("[yellow]PDF generated but not saved (use --save to save)[/]")
        else:
            console.print("[yellow]No PDF data available[/]")
    else:
        console.print(f"[bold red]PDF generation failed: {result.error_message}[/]")


def run_batch_demo(urls_input: str, screenshot: bool, pdf: bool, output_dir: str, save: bool):
    """Core batch processing demo function without Click decorators"""
    # Parse URLs (comma-separated or from file)
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
    
    operation = []
    if screenshot:
        operation.append("screenshot")
    if pdf:
        operation.append("PDF")
    operation_name = " and ".join(operation)
    
    console.print(f"[bold blue]Batch {operation_name} capture for {len(urls)} URLs[/]")
    
    results = asyncio.run(batch_visual_capture(urls, screenshot, pdf))
    
    # Display results
    display_visual_results(results, operation_name)
    
    # Save files if requested
    if save:
        saved_files = []
        for result in results:
            if result.success:
                if screenshot and result.screenshot:
                    filename = generate_filename(result.url, "png")
                    filepath = save_screenshot(result.screenshot, filename, f"{output_dir}/screenshots")
                    saved_files.append(filepath)
                
                if pdf and result.pdf:
                    filename = generate_filename(result.url, "pdf")
                    filepath = save_pdf(result.pdf, filename, f"{output_dir}/pdfs")
                    saved_files.append(filepath)
        
        if saved_files:
            console.print(f"\n[bold green]Saved {len(saved_files)} files to {output_dir}/[/]")
        else:
            console.print("[yellow]No files were saved[/]")


def run_comparison_demo(url: str):
    """Compare different viewport sizes for screenshots"""
    console.print(f"[bold blue]Viewport Size Comparison: {url}[/]")
    
    viewports = [
        (1280, 800, "Desktop"),
        (768, 1024, "Tablet"),
        (375, 667, "Mobile"),
        (1920, 1080, "HD Desktop")
    ]
    
    async def capture_all_viewports():
        results = []
        for width, height, name in viewports:
            console.print(f"[yellow]Capturing {name} ({width}x{height})...[/]")
            result = await screenshot_demo(url, "comparison", width, height)
            results.append((result, name, width, height))
        return results
    
    results = asyncio.run(capture_all_viewports())
    
    # Create comparison table
    table = Table(title="Viewport Comparison")
    table.add_column("Device", style="cyan")
    table.add_column("Resolution", style="yellow")
    table.add_column("Success", style="green")
    table.add_column("Screenshot Size", style="blue")
    
    for result, name, width, height in results:
        success = "✓" if result.success else "✗"
        screenshot_size = f"{len(result.screenshot):,}" if result.screenshot else "0"
        
        table.add_row(
            name,
            f"{width}x{height}",
            success,
            screenshot_size
        )
    
    console.print(table)
    
    # Save comparison files
    output_dir = "viewport_comparison"
    saved_files = []
    for result, name, width, height in results:
        if result.success and result.screenshot:
            filename = f"{name.lower().replace(' ', '_')}_{width}x{height}.png"
            filepath = save_screenshot(result.screenshot, filename, output_dir)
            saved_files.append(filepath)
    
    console.print(f"\n[bold green]Comparison screenshots saved to {output_dir}/[/]")


@click.group()
def cli():
    """Crawl4ai Reference Application - Screenshot and PDF Demo"""
    pass


@cli.command("screenshot")
@click.argument("url", required=True)
@click.option("--output-dir", "-o", default="screenshots", help="Output directory")
@click.option("--width", "-w", default=1280, type=int, help="Viewport width")
@click.option("--height", "-h", default=800, type=int, help="Viewport height")
@click.option("--wait", default=2.0, type=float, help="Wait time before capture (seconds)")
@click.option("--save/--no-save", default=True, help="Save screenshot to file")
def screenshot_cmd(url: str, output_dir: str, width: int, height: int, wait: float, save: bool):
    """Capture a screenshot of a webpage"""
    run_screenshot_demo(url, output_dir, width, height, wait, save)


@cli.command("pdf")
@click.argument("url", required=True)
@click.option("--output-dir", "-o", default="pdfs", help="Output directory")
@click.option("--save/--no-save", default=True, help="Save PDF to file")
def pdf_cmd(url: str, output_dir: str, save: bool):
    """Generate a PDF of a webpage"""
    run_pdf_demo(url, output_dir, save)


@cli.command("batch")
@click.argument("urls", required=True)
@click.option("--screenshot/--no-screenshot", default=True, help="Capture screenshots")
@click.option("--pdf/--no-pdf", default=False, help="Generate PDFs")
@click.option("--output-dir", "-o", default="batch_output", help="Output directory")
@click.option("--save/--no-save", default=True, help="Save files to disk")
def batch_cmd(urls: str, screenshot: bool, pdf: bool, output_dir: str, save: bool):
    """Batch process multiple URLs (comma-separated or .txt file)"""
    run_batch_demo(urls, screenshot, pdf, output_dir, save)


@cli.command("compare")
@click.argument("url", required=True)
def compare_cmd(url: str):
    """Compare screenshots across different viewport sizes"""
    run_comparison_demo(url)


@cli.command("both")
@click.argument("url", required=True)
@click.option("--output-dir", "-o", default="visual_output", help="Output directory")
def both_cmd(url: str, output_dir: str):
    """Capture both screenshot and PDF of a webpage"""
    console.print(f"[bold blue]Capturing both screenshot and PDF: {url}[/]")
    
    async def capture_both():
        return await perform_visual_crawl(url, screenshot=True, pdf=True)
    
    result = asyncio.run(capture_both())
    
    if result.success:
        console.print("[bold green]Both captures successful![/]")
        
        saved_files = []
        
        if result.screenshot:
            filename = generate_filename(url, "png")
            filepath = save_screenshot(result.screenshot, filename, f"{output_dir}/screenshots")
            saved_files.append(filepath)
            console.print(f"[bold green]Screenshot saved:[/] {filepath}")
        
        if result.pdf:
            filename = generate_filename(url, "pdf")
            filepath = save_pdf(result.pdf, filename, f"{output_dir}/pdfs")
            saved_files.append(filepath)
            console.print(f"[bold green]PDF saved:[/] {filepath}")
        
        console.print(f"\n[bold]Total files saved:[/] {len(saved_files)}")
    else:
        console.print(f"[bold red]Capture failed: {result.error_message}[/]")


@cli.command("info")
def info_cmd():
    """Show information about visual capture capabilities"""
    console.print("[bold blue]Crawl4AI Visual Capture Capabilities[/]\n")
    
    console.print("[bold yellow]Screenshot Features:[/]")
    console.print("• Full page screenshots in PNG format")
    console.print("• Custom viewport sizes (mobile, tablet, desktop)")
    console.print("• Configurable wait times for dynamic content")
    console.print("• Base64 encoded data or direct file saving")
    
    console.print("\n[bold yellow]PDF Features:[/]")
    console.print("• Full webpage to PDF conversion")
    console.print("• Preserves formatting and layout")
    console.print("• Suitable for archiving and sharing")
    
    console.print("\n[bold yellow]Batch Processing:[/]")
    console.print("• Process multiple URLs simultaneously")
    console.print("• Progress tracking for large batches")
    console.print("• Automatic filename generation")
    
    console.print("\n[bold green]Common Use Cases:[/]")
    console.print("• Website monitoring and archiving")
    console.print("• Responsive design testing")
    console.print("• Documentation generation")
    console.print("• Visual regression testing")
    console.print("• Content backup and preservation")
    
    console.print("\n[bold green]Usage Examples:[/]")
    console.print("  python screenshot_pdf_demo.py screenshot https://example.com")
    console.print("  python screenshot_pdf_demo.py pdf https://example.com")
    console.print("  python screenshot_pdf_demo.py both https://example.com")
    console.print("  python screenshot_pdf_demo.py batch 'https://site1.com,https://site2.com'")
    console.print("  python screenshot_pdf_demo.py compare https://example.com")


if __name__ == "__main__":
    try:
        cli()
    except Exception as e:
        console.print(f"[bold red]Error: {str(e)}[/]")
        sys.exit(1)