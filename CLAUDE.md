# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

B4UCrawl is a reference application demonstrating Crawl4ai, an open-source LLM-friendly web crawler and scraper. The project provides a unified CLI interface with multiple demo scripts showcasing different crawling capabilities.


## Complete Reference Implementation

B4UCrawl now provides comprehensive reference implementations for **ALL major Crawl4AI capabilities**:

### ✅ Implemented Features

#### **Basic Operations**
- Web crawling with markdown output
- LLM-based structured data extraction
- Table extraction with CSV export and pandas integration

#### **JavaScript & Dynamic Content** (`javascript_demo.py`)
- Element clicking and page interaction
- Infinite scrolling and "load more" functionality
- Form filling and submission
- Multi-step interactions with session management
- Custom JavaScript execution with wait conditions

#### **Advanced Extraction** (`extraction_strategies_demo.py`)
- **CSS Selector Extraction**: Predefined schemas for news, products, social media
- **XPath Extraction**: Complex path-based data extraction
- **Regex Extraction**: Pattern-based extraction for contacts, financial data
- **Comparison Tools**: Side-by-side strategy evaluation

#### **Content Processing** (`content_processing_demo.py`)
- **Content Filtering**: Pruning, BM25, LLM-based filtering
- **Chunking Strategies**: RegexChunking for large documents
- **Markdown Generation**: Enhanced markdown with content filters
- **Performance Analysis**: Content reduction metrics and comparisons

#### **Visual Capture** (`screenshot_pdf_demo.py`)
- **Screenshot Capture**: Custom viewports, mobile/tablet/desktop sizes
- **PDF Generation**: Full webpage to PDF conversion
- **Batch Processing**: Multiple URL processing with progress tracking
- **Viewport Comparison**: Responsive design testing across device sizes

#### **MCP Integration**
- Server setup and management
- Client interaction patterns
- Tool discovery and usage

### **Ready for Abstractions**

This comprehensive reference implementation covers virtually all Crawl4AI functionality, providing a solid foundation for building higher-level abstractions. Any custom tools should be built on top of these proven patterns.

## Context

The `/docs` folder contains detailed documentation about Crawl4AI's capabilities and "vibe coding" approach:

### Vibe Coding Concept
Crawl4AI supports "vibe coding" - a methodology for working with AI assistants where users describe their web scraping goals in natural language rather than writing detailed code. This enables:
- Users to articulate data extraction needs to AI assistants
- AI assistants to generate effective Crawl4AI code from high-level intentions
- Rapid prototyping and development without deep API knowledge

### Core Crawl4AI Components (for AI Assistant Context)
All classes are imported directly from the main `crawl4ai` package:
```python
from crawl4ai import (
    AsyncWebCrawler,        # Main crawling tool using async patterns
    BrowserConfig,          # Browser-level settings (headless mode, proxy, user agent)
    CrawlerRunConfig,       # Configuration for individual crawl runs
    LLMConfig,              # LLM provider configuration for extraction strategies
    LLMExtractionStrategy,  # LLM-based data extraction
    JsonCssExtractionStrategy, # CSS selector-based extraction
    CrawlResult,            # Data object containing crawl results
    DefaultMarkdownGenerator, # Markdown generation
    PruningContentFilter,   # Content filtering for cleaner output
    CacheMode              # Caching behavior control
)
```

### Key Capabilities to Communicate to AI Assistants
- Fetch any webpage content (static or JavaScript-heavy applications)
- Convert web content to clean Markdown (ideal for LLM input)
- Extract structured data as JSON using CSS selectors or LLM extraction
- Process multiple URLs concurrently with `arun_many`
- Capture screenshots and generate PDFs
- Handle page interactions with JavaScript execution
- Deep crawling with BFS/DFS strategies and filtering chains

## Common Commands

### Installation and Setup
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Main Application Commands
The primary entry point is `b4ucrawl.py`, which provides a unified CLI:

```bash
# Basic web crawling
python b4ucrawl.py crawl https://example.com
python b4ucrawl.py crawl https://example.com --output json

# Structured data extraction (requires LLM API token)
python b4ucrawl.py extract https://example.com "Extract main content and title"

# Table extraction
python b4ucrawl.py tables https://example.com
python b4ucrawl.py tables https://example.com --export --output-dir ./tables

# MCP (Model Context Protocol) server operations
python b4ucrawl.py mcp-start --port 11235
python b4ucrawl.py mcp-test https://example.com
python b4ucrawl.py mcp-tools
python b4ucrawl.py mcp-crawl https://example.com
```

### Individual Demo Scripts
Each demo script can also be run independently:
- `python crawl_demo.py crawl https://example.com`
- `python table_extraction_demo.py extract https://example.com`
- `python mcp_demo.py start --port 11235`
- `python mcp_client_demo.py list`

## Architecture

### Core Components

1. **b4ucrawl.py** - Main CLI entry point that unifies all demo functionality
2. **crawl_demo.py** - Basic crawling and LLM-based data extraction
3. **table_extraction_demo.py** - Specialized table extraction and CSV export
4. **javascript_demo.py** - Page interaction, session management, dynamic content
5. **extraction_strategies_demo.py** - CSS, XPath, and regex-based extraction methods
6. **content_processing_demo.py** - Content filtering, chunking, markdown generation
7. **screenshot_pdf_demo.py** - Visual capture and document generation
8. **mcp_demo.py** - MCP server management and testing
9. **mcp_client_demo.py** - Direct MCP client interactions

### Key Design Patterns

- **Async/Await**: All crawling operations use asyncio and AsyncWebCrawler
- **Direct Imports**: All Crawl4AI classes imported directly from main package (`from crawl4ai import ...`)
- **Proper Configuration Creation**: Configuration objects created with `BrowserConfig(**config_dict)` and `CrawlerRunConfig(**config_dict)`
- **Correct Async Lifecycle**: Uses `await crawler.start()` and `await crawler.close()` for proper resource management
- **Configuration Management**: Default configs are centralized and overridable via CLI options and environment variables
- **Rich UI**: All scripts use Rich library for formatted console output with panels, tables, and syntax highlighting
- **Click CLI**: Consistent command-line interface using Click decorators
- **Resource Management**: Proper cleanup with try/finally blocks for crawler start/stop

### Environment Configuration

The application supports configuration via `.env` file:
```
CRAWL4AI_VERBOSE=true
CRAWL4AI_HEADLESS=true
CRAWL4AI_LLM_PROVIDER=openai/gpt-4
CRAWL4AI_LLM_TOKEN=your_token_here
CRAWL4AI_MCP_PORT=11235
CRAWL4AI_MCP_HOST=localhost
```

### Common Configuration Objects

- **DEFAULT_BROWSER_CONFIG**: Headless mode, viewport settings, user agent randomization
- **DEFAULT_CRAWLER_CONFIG**: Verbose output, full page scanning, delay settings
- **Table extraction**: Uses `table_score_threshold` (1-10 scale) for filtering tables

### MCP Integration

The MCP (Model Context Protocol) functionality allows AI models to use Crawl4ai as an external tool:
- Server starts on configurable port (default 11235)
- Provides SSE endpoint for real-time communication
- Supports both direct HTTP calls and MCP protocol integration
- Tools available: `crawl_url`, `extract_data`

### Error Handling Patterns

- Rich console for user-friendly error messages
- Graceful degradation when optional dependencies missing
- Server availability checks before MCP operations
- Proper resource cleanup in async contexts
