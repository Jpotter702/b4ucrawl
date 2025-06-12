# B4UCrawl - Comprehensive Crawl4AI Reference Application

A complete reference implementation demonstrating all capabilities of [Crawl4ai](https://github.com/unclecode/crawl4ai), an open-source LLM-friendly web crawler and scraper. This application provides a unified CLI interface with comprehensive demo scripts showcasing everything from basic crawling to advanced enterprise-level features.

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Basic web crawling
python b4ucrawl.py crawl https://example.com

# Extract structured data with LLM
python b4ucrawl.py extract https://example.com "Extract main content and title"

# Capture screenshot
python b4ucrawl.py screenshot https://example.com

# Deep crawling with BFS strategy
python b4ucrawl.py deep-crawl https://example.com --strategy bfs --max-depth 2
```

## 📋 Complete Feature Set

### 🎯 Core Features (Priority 1)
- ✅ **Basic Crawling**: Web crawling with markdown/JSON output
- ✅ **JavaScript Interaction**: Element clicking, form submission, infinite scrolling
- ✅ **Advanced Extraction**: CSS selectors, XPath, regex patterns
- ✅ **Content Processing**: Filtering, chunking, markdown generation
- ✅ **Visual Capture**: Screenshots and PDF generation
- ✅ **Table Extraction**: CSV export and pandas integration
- ✅ **MCP Integration**: Model Context Protocol server and client

### 🔥 Advanced Features (Priority 2)
- ✅ **Deep Crawling**: BFS, DFS, and Best-First search strategies
- ✅ **Batch Processing**: Multiple URL handling with rate limiting
- ✅ **Proxy Support**: Rotation, failover, and geographic strategies

## 🛠 Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd b4ucrawl
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install Playwright browsers** (required for web crawling)
   ```bash
   playwright install
   playwright install-deps  # On Linux
   ```

## 🎮 Unified CLI Reference

The main `b4ucrawl.py` provides a unified interface to all functionality:

### Basic Commands
```bash
# Web crawling
python b4ucrawl.py crawl <url> [--output json|markdown] [--headless/--no-headless]

# LLM-based extraction (requires API token)
python b4ucrawl.py extract <url> <instruction> [--provider openai/gpt-4] [--token <token>]

# Table extraction
python b4ucrawl.py tables <url> [--export] [--threshold 8]
python b4ucrawl.py pandas <url> [--threshold 8]
```

### Core Feature Commands
```bash
# JavaScript interaction
python b4ucrawl.py js <url> <js_code> [--wait-for <condition>]
python b4ucrawl.py click <url> <selector> [--wait-for <condition>]

# Data extraction
python b4ucrawl.py css-extract <url> [--schema news|products|social]
python b4ucrawl.py regex-extract <url> [--patterns contact|financial]

# Content processing
python b4ucrawl.py filter <url> [--type pruning|bm25]

# Visual capture
python b4ucrawl.py screenshot <url> [--width 1280] [--height 800] [--output-dir screenshots]
python b4ucrawl.py pdf <url> [--output-dir pdfs]
```

### Advanced Crawling Commands
```bash
# Deep crawling
python b4ucrawl.py deep-crawl <start_url> [--strategy bfs|dfs|best-first] [--max-depth 2] [--max-pages 10] [--keywords <keywords>]

# Batch processing
python b4ucrawl.py batch-crawl <urls> [--strategy simple|domain|rate-limited] [--max-concurrent 3] [--rate 1.0]

# Proxy crawling
python b4ucrawl.py proxy-crawl <url> [--proxy-file <file>] [--strategy rotation|failover]
```

### MCP Commands
```bash
# MCP server operations
python b4ucrawl.py mcp-start [--port 11235]
python b4ucrawl.py mcp-test <url> [--port 11235]
python b4ucrawl.py mcp-tools [--host localhost] [--port 11235]
python b4ucrawl.py mcp-crawl <url> [--host localhost] [--port 11235] [--output markdown|json]
```

## 📚 Individual Demo Scripts

Each feature is also available as standalone demo scripts for detailed exploration:

### Core Features
| Script | Purpose | Key Commands |
|--------|---------|--------------|
| `crawl_demo.py` | Basic crawling and LLM extraction | `crawl`, `extract` |
| `table_extraction_demo.py` | Table extraction and CSV export | `extract`, `pandas` |
| `javascript_demo.py` | Page interaction and dynamic content | `click`, `form`, `scroll`, `custom-js` |
| `extraction_strategies_demo.py` | CSS, XPath, regex extraction | `css`, `xpath`, `regex`, `compare` |
| `content_processing_demo.py` | Content filtering and processing | `filter`, `chunk`, `compare`, `custom` |
| `screenshot_pdf_demo.py` | Visual capture capabilities | `screenshot`, `pdf`, `batch`, `compare` |

### Advanced Features
| Script | Purpose | Key Commands |
|--------|---------|--------------|
| `deep_crawling_demo.py` | Deep crawling strategies | `bfs`, `dfs`, `best-first`, `compare` |
| `batch_processing_demo.py` | Multiple URL processing | `simple`, `domain`, `rate-limited`, `compare` |
| `proxy_demo.py` | Proxy rotation and failover | `rotation`, `failover`, `health-check` |

### MCP Integration
| Script | Purpose | Key Commands |
|--------|---------|--------------|
| `mcp_demo.py` | MCP server management | `start`, `test` |
| `mcp_client_demo.py` | Direct MCP client usage | `list`, `crawl` |

## 💡 Usage Examples

### Basic Web Crawling
```bash
# Simple crawling
python b4ucrawl.py crawl https://example.com

# Get JSON output
python b4ucrawl.py crawl https://example.com --output json

# Non-headless mode (show browser)
python b4ucrawl.py crawl https://example.com --no-headless
```

### JavaScript Interaction
```bash
# Click a button and capture result
python b4ucrawl.py click https://example.com "#load-more-button"

# Execute custom JavaScript
python b4ucrawl.py js https://example.com "document.querySelector('#search').value='test'; document.querySelector('#search-btn').click();"

# Advanced interactions
python javascript_demo.py form https://example.com
python javascript_demo.py scroll https://example.com
```

### Data Extraction
```bash
# Extract with CSS selectors (news schema)
python b4ucrawl.py css-extract https://news.ycombinator.com --schema news

# Extract with regex patterns (contact info)
python b4ucrawl.py regex-extract https://example.com --patterns contact

# LLM-based extraction
python b4ucrawl.py extract https://example.com "Extract all product names and prices as JSON"
```

### Content Processing
```bash
# Apply content filtering
python b4ucrawl.py filter https://example.com --type pruning

# Compare filtering strategies
python content_processing_demo.py compare https://example.com

# Chunk large content
python content_processing_demo.py chunk https://example.com --size 1000
```

### Visual Capture
```bash
# Capture screenshot
python b4ucrawl.py screenshot https://example.com --width 1920 --height 1080

# Generate PDF
python b4ucrawl.py pdf https://example.com

# Batch processing
python screenshot_pdf_demo.py batch "https://site1.com,https://site2.com" --screenshot --pdf
```

### Deep Crawling
```bash
# BFS crawling
python b4ucrawl.py deep-crawl https://example.com --strategy bfs --max-depth 2 --max-pages 10

# Best-first with keywords
python b4ucrawl.py deep-crawl https://docs.python.org --strategy best-first --keywords "python,tutorial,guide"

# Compare strategies
python deep_crawling_demo.py compare https://example.com
```

### Batch Processing
```bash
# Simple batch processing
python b4ucrawl.py batch-crawl "https://site1.com,https://site2.com,https://site3.com" --max-concurrent 3

# Domain-based grouping
python b4ucrawl.py batch-crawl urls.txt --strategy domain

# Rate-limited processing
python b4ucrawl.py batch-crawl urls.txt --strategy rate-limited --rate 0.5
```

### Proxy Usage
```bash
# Create sample proxy file
python proxy_demo.py create-sample --output my_proxies.json

# Test proxy health
python proxy_demo.py health-check --proxy-file my_proxies.json

# Crawl with proxy rotation
python b4ucrawl.py proxy-crawl https://example.com --proxy-file my_proxies.json --strategy rotation
```

## ⚙️ Configuration

### Environment Variables
Create a `.env` file for configuration:

```bash
# Basic settings
CRAWL4AI_VERBOSE=true
CRAWL4AI_HEADLESS=true

# LLM settings (for extraction features)
CRAWL4AI_LLM_PROVIDER=openai/gpt-4
CRAWL4AI_LLM_TOKEN=your_token_here

# MCP settings
CRAWL4AI_MCP_PORT=11235
CRAWL4AI_MCP_HOST=localhost
```

### Browser Configuration
- **Headless mode**: `--headless/--no-headless`
- **Viewport size**: `--width` and `--height` options
- **User agent**: Randomized by default
- **Proxy support**: Via `--proxy-file` option

### Performance Tuning
- **Concurrency**: `--max-concurrent` for batch operations
- **Rate limiting**: `--rate` for respectful crawling
- **Timeouts**: Configurable per operation
- **Caching**: Built-in response caching

## 🏗 Architecture Overview

### Core Components
- **b4ucrawl.py**: Unified CLI interface
- **Demo Scripts**: Individual feature demonstrations
- **Configuration**: Environment-based settings
- **Error Handling**: Comprehensive error management
- **Resource Management**: Proper async lifecycle

### Design Patterns
- **Async/Await**: All crawling operations use asyncio
- **Direct Imports**: Clean imports from `crawl4ai` package
- **Rich UI**: Formatted console output with progress tracking
- **Click CLI**: Consistent command-line interfaces
- **Configuration Management**: Centralized and overridable settings

## 🎯 Use Cases

### Business Intelligence
- **E-commerce**: Product monitoring, price tracking, competitor analysis
- **News**: Content aggregation, sentiment analysis, trend tracking
- **Social Media**: Engagement metrics, content analysis, brand monitoring

### Research & Development
- **Academic**: Paper extraction, citation analysis, knowledge graphs
- **Market Research**: Data collection, trend analysis, competitor intelligence
- **Content Analysis**: Large-scale content processing and categorization

### Automation & Monitoring
- **Website Monitoring**: Change detection, uptime tracking, performance monitoring
- **Data Pipeline**: Automated data collection and processing
- **Quality Assurance**: Content validation, link checking, accessibility testing

## 🔧 Troubleshooting

### Common Issues

**Playwright not installed**
```bash
playwright install
playwright install-deps  # Linux only
```

**LLM extraction not working**
- Set `CRAWL4AI_LLM_TOKEN` environment variable
- Verify API key permissions
- Check provider format (e.g., `openai/gpt-4`)

**MCP server connection issues**
- Verify server is running: `python b4ucrawl.py mcp-start`
- Check port availability
- Ensure firewall allows connections

**Slow performance**
- Reduce concurrency: `--max-concurrent 2`
- Enable headless mode: `--headless`
- Use rate limiting: `--rate 1.0`

### Performance Optimization
- Use domain-based batch processing for better cache utilization
- Enable response caching for repeated requests
- Adjust viewport size for faster rendering
- Use appropriate filtering to reduce content size

## 📖 Additional Resources

- **Crawl4ai Documentation**: [GitHub Repository](https://github.com/unclecode/crawl4ai)
- **API Reference**: Check individual demo scripts for detailed examples
- **Community**: Issues and discussions on the Crawl4ai repository
- **Contributing**: Contributions welcome via pull requests

## 📄 License

This project is licensed under the Apache License 2.0 with attribution requirements. See the [Crawl4ai license](https://github.com/unclecode/crawl4ai/blob/main/LICENSE) for details.

**Powered by [Crawl4ai](https://github.com/unclecode/crawl4ai)**

---

## 🎉 Getting Help

1. **Check the examples** in this README for your use case
2. **Run with `--help`** to see all available options for any command
3. **Use the `info` command** in demo scripts for detailed feature information
4. **Check CLAUDE.md** for technical implementation details
5. **Open an issue** on the Crawl4ai repository for bugs or feature requests

Happy crawling! 🕷️