#!/usr/bin/env python3
"""
Crawl4ai Reference Application - JavaScript Interaction Demo

This script demonstrates JavaScript execution, page interaction, session management,
and dynamic content handling with Crawl4ai.
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
from rich.syntax import Syntax

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


async def perform_js_crawl(url: str, js_code: str, wait_for: str = None, session_id: str = None, js_only: bool = False) -> CrawlResult:
    """
    Perform a web crawl with JavaScript execution.
    
    Args:
        url: The URL to crawl
        js_code: JavaScript code to execute
        wait_for: Condition to wait for after JS execution
        session_id: Session ID for maintaining state across calls
        js_only: Whether to only execute JS without page navigation
        
    Returns:
        CrawlResult: The result of the crawl operation
    """
    browser_config = DEFAULT_BROWSER_CONFIG.copy()
    crawler_config = DEFAULT_CRAWLER_CONFIG.copy()
    
    # Add JavaScript execution parameters
    crawler_config.update({
        "js_code": js_code,
        "session_id": session_id,
        "js_only": js_only,
    })
    
    if wait_for:
        crawler_config["wait_for"] = wait_for
    
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


async def click_and_extract(url: str, click_selector: str, extract_selector: str = None, wait_for: str = None) -> CrawlResult:
    """
    Click an element and extract content from the resulting page.
    
    Args:
        url: The URL to crawl
        click_selector: CSS selector for element to click
        extract_selector: CSS selector for content to extract after click
        wait_for: Condition to wait for after clicking
        
    Returns:
        CrawlResult: The result after clicking and optional extraction
    """
    # JavaScript to click element
    js_click = f"""
    (() => {{
        const element = document.querySelector('{click_selector}');
        if (element) {{
            element.click();
            return true;
        }}
        return false;
    }})();
    """
    
    result = await perform_js_crawl(
        url=url,
        js_code=js_click,
        wait_for=wait_for or f"css:{click_selector}",
        session_id="click_session"
    )
    
    return result


async def infinite_scroll_demo(url: str, scroll_count: int = 3) -> CrawlResult:
    """
    Demonstrate infinite scrolling to load more content.
    
    Args:
        url: The URL to crawl
        scroll_count: Number of times to scroll
        
    Returns:
        CrawlResult: The result after scrolling
    """
    session_id = "scroll_session"
    
    # Initial page load
    browser_config = DEFAULT_BROWSER_CONFIG.copy()
    crawler_config = DEFAULT_CRAWLER_CONFIG.copy()
    crawler_config["session_id"] = session_id
    
    browser_cfg = BrowserConfig(**browser_config)
    crawler_cfg = CrawlerRunConfig(**crawler_config)
    
    crawler = AsyncWebCrawler(config=browser_cfg)
    await crawler.start()
    
    try:
        # Initial load
        result = await crawler.arun(url=url, config=crawler_cfg)
        
        # Scroll multiple times
        for i in range(scroll_count):
            console.print(f"[bold blue]Scroll {i+1}/{scroll_count}[/]")
            
            scroll_js = """
            window.scrollTo(0, document.body.scrollHeight);
            """
            
            scroll_config = CrawlerRunConfig(
                **DEFAULT_CRAWLER_CONFIG,
                js_code=scroll_js,
                session_id=session_id,
                js_only=True,
                wait_for=2000  # Wait 2 seconds after scroll
            )
            
            result = await crawler.arun(url=url, config=scroll_config)
        
        return result
    finally:
        await crawler.close()


async def form_interaction_demo(url: str, form_data: Dict[str, str]) -> CrawlResult:
    """
    Demonstrate form filling and submission.
    
    Args:
        url: The URL containing the form
        form_data: Dictionary of field names to values
        
    Returns:
        CrawlResult: The result after form submission
    """
    # Generate JavaScript to fill form
    js_lines = []
    for field_name, value in form_data.items():
        js_lines.append(f"""
        const {field_name}_field = document.querySelector('input[name="{field_name}"], textarea[name="{field_name}"], select[name="{field_name}"]');
        if ({field_name}_field) {{
            {field_name}_field.value = '{value}';
            {field_name}_field.dispatchEvent(new Event('input', {{ bubbles: true }}));
        }}
        """)
    
    # Add form submission
    js_lines.append("""
    const form = document.querySelector('form');
    if (form) {
        form.submit();
    }
    """)
    
    form_js = "\n".join(js_lines)
    
    result = await perform_js_crawl(
        url=url,
        js_code=form_js,
        wait_for=3000,  # Wait 3 seconds for form submission
        session_id="form_session"
    )
    
    return result


async def multi_step_interaction(url: str, steps: List[Dict]) -> List[CrawlResult]:
    """
    Demonstrate multi-step interaction maintaining session state.
    
    Args:
        url: The starting URL
        steps: List of step dictionaries with 'js_code', 'wait_for', etc.
        
    Returns:
        List[CrawlResult]: Results from each step
    """
    session_id = "multi_step_session"
    results = []
    
    browser_config = DEFAULT_BROWSER_CONFIG.copy()
    crawler = AsyncWebCrawler(config=BrowserConfig(**browser_config))
    await crawler.start()
    
    try:
        # Initial page load
        initial_config = CrawlerRunConfig(
            **DEFAULT_CRAWLER_CONFIG,
            session_id=session_id
        )
        
        result = await crawler.arun(url=url, config=initial_config)
        results.append(result)
        
        # Execute each step
        for i, step in enumerate(steps):
            console.print(f"[bold blue]Executing step {i+1}/{len(steps)}: {step.get('description', 'No description')}[/]")
            
            step_config = CrawlerRunConfig(
                **DEFAULT_CRAWLER_CONFIG,
                js_code=step['js_code'],
                session_id=session_id,
                js_only=True,
                wait_for=step.get('wait_for', 2000)
            )
            
            result = await crawler.arun(url=url, config=step_config)
            results.append(result)
        
        return results
    finally:
        await crawler.close()


# CLI Commands

def run_click_demo(url: str, selector: str, wait_for: str):
    """Core click demo function without Click decorators"""
    console.print(f"[bold blue]Clicking element:[/] {selector} on {url}")
    
    result = asyncio.run(click_and_extract(
        url=url,
        click_selector=selector,
        wait_for=wait_for
    ))
    
    if result.success:
        console.print("[bold green]Click successful![/]")
        if result.markdown and result.markdown.raw_markdown:
            console.print(Panel(
                Markdown(result.markdown.raw_markdown[:2000] + "..." if len(result.markdown.raw_markdown) > 2000 else result.markdown.raw_markdown),
                title=f"Content after clicking {selector}",
                expand=False
            ))
        else:
            console.print("[yellow]No markdown content available[/]")
    else:
        console.print(f"[bold red]Click failed:[/] {result.error_message}")


def run_scroll_demo(url: str, count: int):
    """Core scroll demo function without Click decorators"""
    console.print(f"[bold blue]Infinite scrolling:[/] {count} times on {url}")
    
    result = asyncio.run(infinite_scroll_demo(url, count))
    
    if result.success:
        console.print("[bold green]Scrolling successful![/]")
        console.print(f"[bold green]Final content length:[/] {len(result.markdown.raw_markdown) if result.markdown else 0} characters")
        
        # Show snippet of content
        if result.markdown and result.markdown.raw_markdown:
            snippet = result.markdown.raw_markdown[:1000] + "..." if len(result.markdown.raw_markdown) > 1000 else result.markdown.raw_markdown
            console.print(Panel(
                Markdown(snippet),
                title="Content snippet after scrolling",
                expand=False
            ))
    else:
        console.print(f"[bold red]Scrolling failed:[/] {result.error_message}")


def run_form_demo(url: str, form_fields: str):
    """Core form demo function without Click decorators"""
    try:
        # Parse form fields (expected format: "field1=value1,field2=value2")
        form_data = {}
        if form_fields:
            for pair in form_fields.split(','):
                if '=' in pair:
                    key, value = pair.split('=', 1)
                    form_data[key.strip()] = value.strip()
        
        console.print(f"[bold blue]Form interaction:[/] {form_data} on {url}")
        
        result = asyncio.run(form_interaction_demo(url, form_data))
        
        if result.success:
            console.print("[bold green]Form submission successful![/]")
            if result.markdown and result.markdown.raw_markdown:
                console.print(Panel(
                    Markdown(result.markdown.raw_markdown[:1500] + "..." if len(result.markdown.raw_markdown) > 1500 else result.markdown.raw_markdown),
                    title="Content after form submission",
                    expand=False
                ))
        else:
            console.print(f"[bold red]Form submission failed:[/] {result.error_message}")
            
    except Exception as e:
        console.print(f"[bold red]Error in form demo:[/] {str(e)}")


def run_custom_js_demo(url: str, js_code: str, wait_for: str):
    """Core custom JS demo function without Click decorators"""
    console.print(f"[bold blue]Executing custom JavaScript on:[/] {url}")
    console.print(Panel(
        Syntax(js_code, "javascript", theme="monokai"),
        title="JavaScript Code",
        expand=False
    ))
    
    result = asyncio.run(perform_js_crawl(
        url=url,
        js_code=js_code,
        wait_for=wait_for,
        session_id="custom_js_session"
    ))
    
    if result.success:
        console.print("[bold green]JavaScript execution successful![/]")
        if result.markdown and result.markdown.raw_markdown:
            console.print(Panel(
                Markdown(result.markdown.raw_markdown[:1500] + "..." if len(result.markdown.raw_markdown) > 1500 else result.markdown.raw_markdown),
                title="Content after JavaScript execution",
                expand=False
            ))
    else:
        console.print(f"[bold red]JavaScript execution failed:[/] {result.error_message}")


def run_multi_step_demo(url: str):
    """Demo of multi-step interaction with session management"""
    console.print(f"[bold blue]Multi-step interaction demo on:[/] {url}")
    
    # Example steps for a typical SPA navigation
    steps = [
        {
            "description": "Accept cookies",
            "js_code": """
            const cookieButton = document.querySelector('button[data-testid="cookie-accept"], .cookie-accept, #accept-cookies');
            if (cookieButton) cookieButton.click();
            """,
            "wait_for": 1000
        },
        {
            "description": "Open navigation menu",
            "js_code": """
            const menuButton = document.querySelector('.menu-toggle, .hamburger, [aria-label*="menu"]');
            if (menuButton) menuButton.click();
            """,
            "wait_for": 1000
        },
        {
            "description": "Click on About section",
            "js_code": """
            const aboutLink = document.querySelector('a[href*="about"], a:contains("About")');
            if (aboutLink) aboutLink.click();
            """,
            "wait_for": 2000
        }
    ]
    
    results = asyncio.run(multi_step_interaction(url, steps))
    
    console.print(f"[bold green]Completed {len(results)} steps[/]")
    
    # Show final result
    final_result = results[-1]
    if final_result.success and final_result.markdown:
        console.print(Panel(
            Markdown(final_result.markdown.raw_markdown[:1000] + "..." if len(final_result.markdown.raw_markdown) > 1000 else final_result.markdown.raw_markdown),
            title="Final content after all steps",
            expand=False
        ))


@click.group()
def cli():
    """Crawl4ai Reference Application - JavaScript Interaction Demo"""
    pass


@cli.command("click")
@click.argument("url", required=True)
@click.argument("selector", required=True)
@click.option("--wait-for", "-w", default="2000", help="Condition to wait for after clicking (CSS selector or milliseconds)")
def click_cmd(url: str, selector: str, wait_for: str):
    """Click an element and show the resulting content"""
    run_click_demo(url, selector, wait_for)


@cli.command("scroll")
@click.argument("url", required=True)
@click.option("--count", "-c", default=3, type=int, help="Number of times to scroll")
def scroll_cmd(url: str, count: int):
    """Perform infinite scrolling to load more content"""
    run_scroll_demo(url, count)


@cli.command("form")
@click.argument("url", required=True)
@click.option("--fields", "-f", default="", help="Form fields as 'field1=value1,field2=value2'")
def form_cmd(url: str, fields: str):
    """Fill and submit a form"""
    run_form_demo(url, fields)


@cli.command("js")
@click.argument("url", required=True)
@click.argument("js_code", required=True)
@click.option("--wait-for", "-w", default="2000", help="Condition to wait for after JS execution")
def js_cmd(url: str, js_code: str, wait_for: str):
    """Execute custom JavaScript code"""
    run_custom_js_demo(url, js_code, wait_for)


@cli.command("multi-step")
@click.argument("url", required=True)
def multi_step_cmd(url: str):
    """Demonstrate multi-step interaction with session management"""
    run_multi_step_demo(url)


if __name__ == "__main__":
    try:
        cli()
    except Exception as e:
        console.print(f"[bold red]Error:[/] {str(e)}")
        sys.exit(1)