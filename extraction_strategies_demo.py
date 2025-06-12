#!/usr/bin/env python3
"""
Crawl4ai Reference Application - Extraction Strategies Demo

This script demonstrates different extraction strategies available in Crawl4ai:
- JsonCssExtractionStrategy: Extract data using CSS selectors
- JsonXPathExtractionStrategy: Extract data using XPath expressions  
- RegexExtractionStrategy: Extract data using regular expressions
- LLMExtractionStrategy: Extract data using LLM reasoning (covered in crawl_demo.py)
"""

import asyncio
import json
import os
import sys
from typing import Dict, List, Optional, Any

import click
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from crawl4ai import (
    AsyncWebCrawler, 
    BrowserConfig, 
    CrawlerRunConfig, 
    CrawlResult,
    JsonCssExtractionStrategy,
    JsonXPathExtractionStrategy, 
    RegexExtractionStrategy
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


async def perform_extraction_crawl(url: str, extraction_strategy) -> CrawlResult:
    """
    Perform a web crawl with a specific extraction strategy.
    
    Args:
        url: The URL to crawl
        extraction_strategy: The extraction strategy to use
        
    Returns:
        CrawlResult: The result of the crawl operation
    """
    browser_config = DEFAULT_BROWSER_CONFIG.copy()
    crawler_config = DEFAULT_CRAWLER_CONFIG.copy()
    crawler_config["extraction_strategy"] = extraction_strategy
    
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


async def css_extraction_demo(url: str, schema: Dict) -> CrawlResult:
    """
    Demonstrate CSS selector-based extraction.
    
    Args:
        url: The URL to crawl
        schema: Schema defining what to extract
        
    Returns:
        CrawlResult: The crawl result with extracted data
    """
    extraction_strategy = JsonCssExtractionStrategy(schema=schema)
    return await perform_extraction_crawl(url, extraction_strategy)


async def xpath_extraction_demo(url: str, schema: Dict) -> CrawlResult:
    """
    Demonstrate XPath-based extraction.
    
    Args:
        url: The URL to crawl
        schema: Schema defining what to extract using XPath
        
    Returns:
        CrawlResult: The crawl result with extracted data
    """
    extraction_strategy = JsonXPathExtractionStrategy(schema=schema)
    return await perform_extraction_crawl(url, extraction_strategy)


async def regex_extraction_demo(url: str, patterns: Dict[str, str]) -> CrawlResult:
    """
    Demonstrate regex-based extraction.
    
    Args:
        url: The URL to crawl
        patterns: Dictionary of field names to regex patterns
        
    Returns:
        CrawlResult: The crawl result with extracted data
    """
    extraction_strategy = RegexExtractionStrategy(patterns=patterns)
    return await perform_extraction_crawl(url, extraction_strategy)


def display_extraction_result(result: CrawlResult, strategy_name: str):
    """Display the extraction result in a formatted way."""
    if result.success:
        console.print(f"[bold green]{strategy_name} extraction successful![/]")
        
        if result.extracted_content:
            try:
                extracted_data = json.loads(result.extracted_content)
                console.print(Panel(
                    json.dumps(extracted_data, indent=2),
                    title=f"Extracted Data ({strategy_name})",
                    expand=False
                ))
            except json.JSONDecodeError:
                console.print(Panel(
                    result.extracted_content,
                    title=f"Extracted Content ({strategy_name})",
                    expand=False
                ))
        else:
            console.print("[yellow]No extracted content available[/]")
    else:
        console.print(f"[bold red]{strategy_name} extraction failed:[/] {result.error_message}")


# Predefined schemas for common websites

def get_news_schema_css():
    """CSS schema for extracting news articles."""
    return {
        "name": "NewsArticles",
        "baseSelector": "article, .article, .post, [data-testid*='article']",
        "fields": [
            {"name": "title", "selector": "h1, h2, .title, .headline", "type": "text"},
            {"name": "author", "selector": ".author, .byline, [data-testid*='author']", "type": "text"},
            {"name": "date", "selector": ".date, .published, time", "type": "text"},
            {"name": "content", "selector": ".content, .body, p", "type": "text"},
            {"name": "link", "selector": "a", "type": "attribute", "attribute": "href"}
        ]
    }

def get_news_schema_xpath():
    """XPath schema for extracting news articles."""
    return {
        "name": "NewsArticles", 
        "baseSelector": "//article | //div[contains(@class, 'article')] | //div[contains(@class, 'post')]",
        "fields": [
            {"name": "title", "selector": ".//h1 | .//h2 | .//*[contains(@class, 'title')]", "type": "text"},
            {"name": "author", "selector": ".//*[contains(@class, 'author')] | .//*[contains(@class, 'byline')]", "type": "text"},
            {"name": "date", "selector": ".//*[contains(@class, 'date')] | .//time", "type": "text"},
            {"name": "content", "selector": ".//*[contains(@class, 'content')] | .//p", "type": "text"}
        ]
    }

def get_product_schema_css():
    """CSS schema for extracting product information."""
    return {
        "name": "Products",
        "baseSelector": ".product, .item, [data-testid*='product']",
        "fields": [
            {"name": "name", "selector": ".product-name, .title, h2, h3", "type": "text"},
            {"name": "price", "selector": ".price, .cost, .amount", "type": "text"},
            {"name": "description", "selector": ".description, .summary", "type": "text"},
            {"name": "image", "selector": "img", "type": "attribute", "attribute": "src"},
            {"name": "rating", "selector": ".rating, .stars, .score", "type": "text"},
            {"name": "link", "selector": "a", "type": "attribute", "attribute": "href"}
        ]
    }

def get_social_schema_css():
    """CSS schema for extracting social media posts."""
    return {
        "name": "SocialPosts",
        "baseSelector": ".post, .tweet, .status, [data-testid*='post']",
        "fields": [
            {"name": "author", "selector": ".author, .username, .user", "type": "text"},
            {"name": "content", "selector": ".content, .text, .message", "type": "text"},
            {"name": "timestamp", "selector": ".time, .date, .timestamp", "type": "text"},
            {"name": "likes", "selector": ".likes, .hearts, .reactions", "type": "text"},
            {"name": "shares", "selector": ".shares, .retweets", "type": "text"}
        ]
    }

def get_email_regex_patterns():
    """Regex patterns for extracting email addresses and contact info."""
    return {
        "emails": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        "phone_numbers": r'(\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
        "urls": r'https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:[\w.])*)?)?',
        "dates": r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b'
    }

def get_price_regex_patterns():
    """Regex patterns for extracting prices and financial data."""
    return {
        "prices": r'\$\d+(?:,\d{3})*(?:\.\d{2})?|\d+(?:,\d{3})*(?:\.\d{2})?\s*(?:USD|EUR|GBP)',
        "percentages": r'\d+(?:\.\d+)?%',
        "numbers": r'\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b'
    }


# CLI Commands

def run_css_demo(url: str, schema_type: str, custom_schema: str):
    """Core CSS extraction demo function without Click decorators"""
    console.print(f"[bold blue]CSS Extraction Demo: {url}[/]")
    
    # Select schema
    if custom_schema:
        try:
            schema = json.loads(custom_schema)
        except json.JSONDecodeError:
            console.print("[bold red]Error: Invalid JSON schema[/]")
            return
    elif schema_type == "news":
        schema = get_news_schema_css()
    elif schema_type == "products":
        schema = get_product_schema_css()
    elif schema_type == "social":
        schema = get_social_schema_css()
    else:
        console.print("[bold red]Error: Unknown schema type. Use 'news', 'products', 'social', or provide custom schema[/]")
        return
    
    # Display schema
    from rich.text import Text
    schema_text = Text(json.dumps(schema, indent=2))
    console.print(Panel(
        schema_text,
        title="CSS Extraction Schema",
        expand=False
    ))
    
    result = asyncio.run(css_extraction_demo(url, schema))
    display_extraction_result(result, "CSS")


def run_xpath_demo(url: str, schema_type: str, custom_schema: str):
    """Core XPath extraction demo function without Click decorators"""
    console.print(f"[bold blue]XPath Extraction Demo: {url}[/]")
    
    # Select schema
    if custom_schema:
        try:
            schema = json.loads(custom_schema)
        except json.JSONDecodeError:
            console.print("[bold red]Error: Invalid JSON schema[/]")
            return
    elif schema_type == "news":
        schema = get_news_schema_xpath()
    else:
        console.print("[bold red]Error: For XPath demo, use 'news' schema type or provide custom schema[/]")
        return
    
    # Display schema
    from rich.text import Text
    schema_text = Text(json.dumps(schema, indent=2))
    console.print(Panel(
        schema_text,
        title="XPath Extraction Schema",
        expand=False
    ))
    
    result = asyncio.run(xpath_extraction_demo(url, schema))
    display_extraction_result(result, "XPath")


def run_regex_demo(url: str, pattern_type: str, custom_patterns: str):
    """Core Regex extraction demo function without Click decorators"""
    console.print(f"[bold blue]Regex Extraction Demo: {url}[/]")
    
    # Select patterns
    if custom_patterns:
        try:
            patterns = json.loads(custom_patterns)
        except json.JSONDecodeError:
            console.print("[bold red]Error: Invalid JSON patterns[/]")
            return
    elif pattern_type == "contact":
        patterns = get_email_regex_patterns()
    elif pattern_type == "financial":
        patterns = get_price_regex_patterns()
    else:
        console.print("[bold red]Error: Unknown pattern type. Use 'contact', 'financial', or provide custom patterns[/]")
        return
    
    # Display patterns
    from rich.text import Text
    patterns_text = Text(json.dumps(patterns, indent=2))
    console.print(Panel(
        patterns_text,
        title="Regex Extraction Patterns",
        expand=False
    ))
    
    result = asyncio.run(regex_extraction_demo(url, patterns))
    display_extraction_result(result, "Regex")


def run_compare_demo(url: str):
    """Compare different extraction strategies on the same URL"""
    console.print(f"[bold blue]Comparing Extraction Strategies on: {url}[/]")
    
    # CSS extraction
    console.print("\n[bold yellow]1. CSS Selector Extraction[/]")
    css_schema = get_news_schema_css()
    css_result = asyncio.run(css_extraction_demo(url, css_schema))
    
    # XPath extraction  
    console.print("\n[bold yellow]2. XPath Extraction[/]")
    xpath_schema = get_news_schema_xpath()
    xpath_result = asyncio.run(xpath_extraction_demo(url, xpath_schema))
    
    # Regex extraction
    console.print("\n[bold yellow]3. Regex Extraction[/]")
    regex_patterns = get_email_regex_patterns()
    regex_result = asyncio.run(regex_extraction_demo(url, regex_patterns))
    
    # Create comparison table
    table = Table(title="Extraction Strategy Comparison")
    table.add_column("Strategy", style="cyan")
    table.add_column("Success", style="green")
    table.add_column("Data Extracted", style="yellow")
    
    for name, result in [("CSS", css_result), ("XPath", xpath_result), ("Regex", regex_result)]:
        success = "✓" if result.success else "✗"
        data_count = "0"
        if result.success and result.extracted_content:
            try:
                data = json.loads(result.extracted_content)
                if isinstance(data, list):
                    data_count = str(len(data))
                elif isinstance(data, dict):
                    data_count = str(len(data))
            except:
                data_count = "Error parsing"
        
        table.add_row(name, success, data_count)
    
    console.print(table)


@click.group()
def cli():
    """Crawl4ai Reference Application - Extraction Strategies Demo"""
    pass


@cli.command("css")
@click.argument("url", required=True)
@click.option("--schema", "-s", type=click.Choice(["news", "products", "social"]), default="news", 
              help="Predefined schema type")
@click.option("--custom", "-c", help="Custom JSON schema")
def css_cmd(url: str, schema: str, custom: str):
    """Extract data using CSS selectors"""
    run_css_demo(url, schema, custom)


@cli.command("xpath")
@click.argument("url", required=True)
@click.option("--schema", "-s", type=click.Choice(["news"]), default="news",
              help="Predefined schema type")
@click.option("--custom", "-c", help="Custom JSON schema")
def xpath_cmd(url: str, schema: str, custom: str):
    """Extract data using XPath expressions"""
    run_xpath_demo(url, schema, custom)


@cli.command("regex")
@click.argument("url", required=True)
@click.option("--patterns", "-p", type=click.Choice(["contact", "financial"]), default="contact",
              help="Predefined pattern type")
@click.option("--custom", "-c", help="Custom JSON patterns")
def regex_cmd(url: str, patterns: str, custom: str):
    """Extract data using regular expressions"""
    run_regex_demo(url, patterns, custom)


@cli.command("compare")
@click.argument("url", required=True)
def compare_cmd(url: str):
    """Compare all extraction strategies on the same URL"""
    run_compare_demo(url)


@cli.command("schemas")
def schemas_cmd():
    """Show available predefined schemas and patterns"""
    console.print("[bold blue]Available Predefined Schemas and Patterns[/]\n")
    
    # CSS Schemas
    console.print("[bold yellow]CSS Schemas:[/]")
    console.print("• news: Extract news articles (title, author, date, content)")
    console.print("• products: Extract product info (name, price, description)")  
    console.print("• social: Extract social media posts (author, content, timestamp)")
    
    # XPath Schemas
    console.print("\n[bold yellow]XPath Schemas:[/]")
    console.print("• news: Extract news articles using XPath")
    
    # Regex Patterns
    console.print("\n[bold yellow]Regex Patterns:[/]")
    console.print("• contact: Extract emails, phone numbers, URLs")
    console.print("• financial: Extract prices, percentages, numbers")
    
    console.print("\n[bold green]Usage Examples:[/]")
    console.print("  python extraction_strategies_demo.py css https://news.ycombinator.com --schema news")
    console.print("  python extraction_strategies_demo.py regex https://example.com --patterns contact")
    console.print("  python extraction_strategies_demo.py compare https://example.com")


if __name__ == "__main__":
    try:
        cli()
    except Exception as e:
        console.print(f"[bold red]Error: {str(e)}[/]")
        sys.exit(1)