"""
Scraper Template
================
A template for creating web scraper scripts.

Usage:
    Customize the URL and parsing logic below.
    Run this script to scrape data and save to the project database.
"""
import requests
from bs4 import BeautifulSoup


def scrape(url: str) -> dict:
    """
    Scrape content from the given URL.
    
    Args:
        url: The URL to scrape
        
    Returns:
        A dictionary containing the scraped data
    """
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Customize your scraping logic here
    title = soup.find('title')
    title_text = title.get_text(strip=True) if title else ""
    
    paragraphs = soup.find_all('p')
    content = "\n".join(p.get_text(strip=True) for p in paragraphs)
    
    return {
        'url': url,
        'title': title_text,
        'content': content
    }


if __name__ == "__main__":
    # Example usage
    result = scrape("https://example.com")
    print(f"Title: {result['title']}")
    print(f"Content length: {len(result['content'])} chars")
