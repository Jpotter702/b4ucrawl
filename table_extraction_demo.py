#!/usr/bin/env python3
"""
Crawl4ai Reference Application - Table Extraction Demo

This script demonstrates how to extract tables from websites using Crawl4ai.
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
from rich.table import Table as RichTable

from crawl4ai.async_configs import BrowserConfig, CrawlerRunConfig
from crawl4ai.async_crawler import AsyncWebCrawler
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
    "table_score_threshold": 8,  # Higher threshold for more accurate table detection
}


async def extract_tables(url: str, browser_config: Dict = None, crawler_config: Dict = None) -> List[Dict]:
    """
    Extract tables from a webpage using Crawl4ai.
    
    Args:
        url: The URL to extract tables from
        browser_config: Optional browser configuration
        crawler_config: Optional crawler configuration
        
    Returns:
        List[Dict]: A list of extracted tables
    """
    # Use default configs if not provided
    browser_cfg = BrowserConfig.from_kwargs(browser_config or DEFAULT_BROWSER_CONFIG)
    
    # Configure crawler with table extraction settings
    crawler_cfg_dict = crawler_config or DEFAULT_CRAWLER_CONFIG.copy()
    crawler_cfg_dict["table_score_threshold"] = crawler_cfg_dict.get("table_score_threshold", 8)
    crawler_cfg = CrawlerRunConfig.from_kwargs(crawler_cfg_dict)
    
    # Initialize the crawler
    crawler = AsyncWebCrawler(config=browser_cfg)
    await crawler.start()
    
    try:
        # Execute the crawl
        results = await crawler.arun(url=url, config=crawler_cfg)
        result = results[0] if isinstance(results, list) else results
        
        # Extract tables from the result
        tables = []
        if result.success and result.media and "tables" in result.media:
            tables = result.media["tables"]
        
        return tables
    finally:
        # Always stop the crawler to clean up resources
        await crawler.stop()


def display_table(table_data: Dict):
    """
    Display a table using rich.
    
    Args:
        table_data: The table data to display
    """
    if not table_data or not table_data.get("headers") or not table_data.get("rows"):
        console.print("[yellow]No valid table data to display[/]")
        return
    
    # Create a rich table
    table = RichTable(title=table_data.get("caption", "Extracted Table"))
    
    # Add headers
    for header in table_data["headers"]:
        table.add_column(header)
    
    # Add rows
    for row in table_data["rows"]:
        # Convert all values to strings
        string_row = [str(cell) if cell is not None else "" for cell in row]
        table.add_row(*string_row)
    
    # Display the table
    console.print(table)


def export_table_to_csv(table_data: Dict, filename: str):
    """
    Export a table to CSV.
    
    Args:
        table_data: The table data to export
        filename: The filename to save the CSV to
    """
    import csv
    
    if not table_data or not table_data.get("headers") or not table_data.get("rows"):
        console.print("[yellow]No valid table data to export[/]")
        return
    
    try:
        with open(filename, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            # Write headers
            writer.writerow(table_data["headers"])
            # Write rows
            writer.writerows(table_data["rows"])
        
        console.print(f"[bold green]Table exported to {filename}[/]")
    except Exception as e:
        console.print(f"[bold red]Error exporting table to CSV:[/] {str(e)}")


@click.group()
def cli():
    """Crawl4ai Reference Application - Table Extraction Demo"""
    pass


@cli.command("extract")
@click.argument("url", required=True)
@click.option("--export", "-e", is_flag=True, help="Export tables to CSV files")
@click.option("--output-dir", "-o", default=".", help="Directory to save exported tables")
@click.option("--threshold", "-t", default=8, type=int, help="Table score threshold (1-10)")
@click.option("--headless/--no-headless", default=True, help="Run in headless mode")
@click.option("--verbose/--no-verbose", default=False, help="Enable verbose output")
def extract_cmd(url: str, export: bool, output_dir: str, threshold: int, headless: bool, verbose: bool):
    """Extract tables from a website"""
    # Update config with command line options
    browser_config = {**DEFAULT_BROWSER_CONFIG, "headless": headless}
    crawler_config = {
        **DEFAULT_CRAWLER_CONFIG, 
        "verbose": verbose,
        "table_score_threshold": threshold
    }
    
    console.print(f"[bold blue]Extracting tables from:[/] {url}")
    console.print(f"[bold blue]Table score threshold:[/] {threshold} (higher = stricter detection)")
    
    # Run the extraction
    tables = asyncio.run(extract_tables(url, browser_config, crawler_config))
    
    if not tables:
        console.print("[bold yellow]No tables found on the page[/]")
        return
    
    console.print(f"[bold green]Found {len(tables)} tables![/]")
    
    # Display and optionally export each table
    for i, table in enumerate(tables):
        console.print(f"\n[bold]Table {i+1}:[/]")
        
        # Display table information
        console.print(f"Caption: {table.get('caption', 'N/A')}")
        console.print(f"Size: {len(table.get('rows', []))} rows × {len(table.get('headers', []))} columns")
        console.print(f"Score: {table.get('score', 'N/A')}")
        
        # Display the table
        display_table(table)
        
        # Export the table if requested
        if export:
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate filename
            base_filename = os.path.join(output_dir, f"table_{i+1}")
            csv_filename = f"{base_filename}.csv"
            
            # Export to CSV
            export_table_to_csv(table, csv_filename)


@cli.command("pandas")
@click.argument("url", required=True)
@click.option("--threshold", "-t", default=8, type=int, help="Table score threshold (1-10)")
def pandas_cmd(url: str, threshold: int):
    """Extract tables and convert to pandas DataFrame"""
    try:
        import pandas as pd
    except ImportError:
        console.print("[bold red]Error:[/] pandas is not installed. Install it with 'pip install pandas'")
        return
    
    # Update config with command line options
    crawler_config = {
        **DEFAULT_CRAWLER_CONFIG,
        "table_score_threshold": threshold
    }
    
    console.print(f"[bold blue]Extracting tables from:[/] {url}")
    
    # Run the extraction
    tables = asyncio.run(extract_tables(url, None, crawler_config))
    
    if not tables:
        console.print("[bold yellow]No tables found on the page[/]")
        return
    
    console.print(f"[bold green]Found {len(tables)} tables![/]")
    
    # Convert each table to a pandas DataFrame
    for i, table in enumerate(tables):
        if not table.get("headers") or not table.get("rows"):
            console.print(f"[yellow]Table {i+1} has no valid data[/]")
            continue
        
        console.print(f"\n[bold]Table {i+1}:[/]")
        
        # Create DataFrame
        df = pd.DataFrame(table["rows"], columns=table["headers"])
        
        # Display DataFrame info
        console.print(f"DataFrame shape: {df.shape}")
        console.print(f"DataFrame columns: {', '.join(df.columns)}")
        
        # Display DataFrame head
        console.print("\nDataFrame preview:")
        console.print(df.head().to_string())
        
        # Display DataFrame description
        console.print("\nDataFrame statistics:")
        try:
            console.print(df.describe().to_string())
        except:
            console.print("[yellow]Could not generate statistics for this DataFrame[/]")


if __name__ == "__main__":
    try:
        cli()
    except Exception as e:
        console.print(f"[bold red]Error:[/] {str(e)}")
        sys.exit(1) 