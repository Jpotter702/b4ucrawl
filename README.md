# Crawl4ai Reference Application

This is a reference application demonstrating the basic usage of [Crawl4ai](https://github.com/unclecode/crawl4ai), an open-source LLM-friendly web crawler and scraper.

## Features

- Basic web crawling and content extraction
- Markdown output generation
- JSON structured data extraction
- Example of using the MCP (Model Context Protocol) server

## Installation

1. Clone this repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Basic Crawling

```bash
python crawl_demo.py crawl https://example.com
```

### Extract JSON Data

```bash
python crawl_demo.py extract https://example.com "Extract main content and title"
```

### MCP Server Demo

```bash
python mcp_demo.py
```

## Configuration

You can configure the application by creating a `.env` file with the following options:

```
CRAWL4AI_VERBOSE=true
CRAWL4AI_HEADLESS=true
```

## License

This project is licensed under the Apache License 2.0 with attribution requirements. See the [Crawl4ai license](https://github.com/unclecode/crawl4ai/blob/main/LICENSE) for details.

Powered by [Crawl4ai](https://github.com/unclecode/crawl4ai) 