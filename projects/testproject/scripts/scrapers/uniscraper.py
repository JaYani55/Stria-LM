#!/usr/bin/env python3
"""
Scraper: scraperuni
Description: [Add description here]
"""

import requests
from bs4 import BeautifulSoup


def scrape(url: str) -> dict:
    """
    Scrape data from the given URL.
    
    Args:
        url: The URL to scrape
        
    Returns:
        Dictionary with scraped data
    """
    response = requests.get(url)
    response.raise_for_status()
    
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # TODO: Implement scraping logic
    data = {
        "url": url,
        "title": soup.title.string if soup.title else "",
        "content": ""
    }
    
    return data


if __name__ == "__main__":
    # Test the scraper
    result = scrape("https://example.com")
    print(result)

