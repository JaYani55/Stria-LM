import requests
import scrapy
from scrapy.crawler import CrawlerProcess
from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import xml.etree.ElementTree as ET
import logging
from typing import List, Dict, Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UniversalSpider(CrawlSpider):
    name = 'universal_spider'
    
    def __init__(self, start_url: str, max_pages: int = 10, *args, **kwargs):
        super(UniversalSpider, self).__init__(*args, **kwargs)
        self.start_url = start_url
        self.max_pages = max_pages
        self.pages_scraped = 0
        self.scraped_data = []
        
        # Parse the domain from start_url
        parsed_url = urlparse(start_url)
        self.allowed_domains = [parsed_url.netloc]
        self.start_urls = [start_url]
        
        # Define rules for following links within the same domain
        self.rules = (
            Rule(LinkExtractor(allow_domains=self.allowed_domains), 
                 callback='parse_page', follow=True),
        )
        super(UniversalSpider, self)._compile_rules()
    
    def parse_page(self, response):
        """Parse each page and extract text content"""
        if self.pages_scraped >= self.max_pages:
            return
        
        self.pages_scraped += 1
        
        # Use BeautifulSoup for better text extraction
        soup = BeautifulSoup(response.text, 'lxml')
        
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "header", "footer"]):
            script.decompose()
        
        # Get text content
        text_content = soup.get_text(separator=' ', strip=True)
        
        # Clean up text
        lines = (line.strip() for line in text_content.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text_content = ' '.join(chunk for chunk in chunks if chunk)
        
        page_data = {
            'url': response.url,
            'title': soup.title.string if soup.title else '',
            'content': text_content,
            'domain': urlparse(response.url).netloc
        }
        
        self.scraped_data.append(page_data)
        
        logger.info(f"Scraped page {self.pages_scraped}/{self.max_pages}: {response.url}")
        
        yield page_data

class WebScraper:
    def __init__(self):
        self.scraped_data = []
    
    def check_sitemap(self, base_url: str) -> Optional[List[str]]:
        """Check if sitemap.xml exists and extract URLs"""
        sitemap_urls = [
            urljoin(base_url, '/sitemap.xml'),
            urljoin(base_url, '/sitemap_index.xml'),
            urljoin(base_url, '/robots.txt')  # Check robots.txt for sitemap
        ]
        
        for sitemap_url in sitemap_urls:
            try:
                response = requests.get(sitemap_url, timeout=10)
                if response.status_code == 200:
                    if 'sitemap.xml' in sitemap_url:
                        return self._parse_sitemap(response.content)
                    elif 'robots.txt' in sitemap_url:
                        # Parse robots.txt for sitemap references
                        for line in response.text.split('\n'):
                            if line.lower().startswith('sitemap:'):
                                sitemap_ref = line.split(':', 1)[1].strip()
                                sitemap_response = requests.get(sitemap_ref, timeout=10)
                                if sitemap_response.status_code == 200:
                                    return self._parse_sitemap(sitemap_response.content)
            except Exception as e:
                logger.warning(f"Could not access {sitemap_url}: {e}")
                continue
        
        return None
    
    def _parse_sitemap(self, xml_content: bytes) -> List[str]:
        """Parse sitemap XML and extract URLs"""
        try:
            root = ET.fromstring(xml_content)
            urls = []
            
            # Handle different sitemap namespaces
            for url_elem in root.findall('.//{http://www.sitemaps.org/schemas/sitemap/0.9}url'):
                loc_elem = url_elem.find('{http://www.sitemaps.org/schemas/sitemap/0.9}loc')
                if loc_elem is not None:
                    urls.append(loc_elem.text)
            
            # Also check for sitemap index files
            for sitemap_elem in root.findall('.//{http://www.sitemaps.org/schemas/sitemap/0.9}sitemap'):
                loc_elem = sitemap_elem.find('{http://www.sitemaps.org/schemas/sitemap/0.9}loc')
                if loc_elem is not None:
                    # Recursively parse sitemap index
                    try:
                        response = requests.get(loc_elem.text, timeout=10)
                        if response.status_code == 200:
                            sub_urls = self._parse_sitemap(response.content)
                            urls.extend(sub_urls)
                    except Exception as e:
                        logger.warning(f"Could not parse sub-sitemap {loc_elem.text}: {e}")
            
            return urls[:10]  # Limit to 10 URLs
        except ET.ParseError as e:
            logger.error(f"Error parsing sitemap XML: {e}")
            return []
    
    def scrape_from_sitemap(self, urls: List[str]) -> List[Dict]:
        """Scrape content from sitemap URLs"""
        scraped_data = []
        
        for url in urls[:10]:  # Limit to 10 pages
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'lxml')
                    
                    # Remove unwanted elements
                    for script in soup(["script", "style", "nav", "header", "footer"]):
                        script.decompose()
                    
                    # Get text content
                    text_content = soup.get_text(separator=' ', strip=True)
                    
                    # Clean up text
                    lines = (line.strip() for line in text_content.splitlines())
                    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                    text_content = ' '.join(chunk for chunk in chunks if chunk)
                    
                    page_data = {
                        'url': url,
                        'title': soup.title.string if soup.title else '',
                        'content': text_content,
                        'domain': urlparse(url).netloc
                    }
                    
                    scraped_data.append(page_data)
                    logger.info(f"Scraped from sitemap: {url}")
                    
            except Exception as e:
                logger.error(f"Error scraping {url}: {e}")
                continue
        
        return scraped_data
    
    def scrape_with_crawler(self, start_url: str, max_pages: int = 10) -> List[Dict]:
        """Use Scrapy crawler to scrape website"""
        def run_spider():
            process = CrawlerProcess({
                'USER_AGENT': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'ROBOTSTXT_OBEY': True,
                'DOWNLOAD_DELAY': 1,
                'RANDOMIZE_DOWNLOAD_DELAY': True,
                'COOKIES_ENABLED': False,
                'LOG_LEVEL': 'WARNING'
            })
            
            spider = UniversalSpider(start_url, max_pages)
            process.crawl(spider)
            process.start()
            return spider.scraped_data
        
        # Run spider in a separate process to avoid reactor issues
        if hasattr(asyncio, '_get_running_loop') and asyncio._get_running_loop():
            # If we're in an async context, use ThreadPoolExecutor
            with ThreadPoolExecutor() as executor:
                future = executor.submit(run_spider)
                return future.result()
        else:
            return run_spider()
    
    def scrape_website(self, url: str, max_pages: int = 10) -> List[Dict]:
        """
        Main scraping function that intelligently chooses between sitemap and crawler
        """
        logger.info(f"Starting to scrape: {url}")
        
        # First, check for sitemap
        sitemap_urls = self.check_sitemap(url)
        
        if sitemap_urls:
            logger.info(f"Found sitemap with {len(sitemap_urls)} URLs")
            return self.scrape_from_sitemap(sitemap_urls)
        else:
            logger.info("No sitemap found, using crawler approach")
            return self.scrape_with_crawler(url, max_pages)

# Main function for external use
def scrape_website(url: str, max_pages: int = 10) -> List[Dict]:
    """
    Scrape a website intelligently using sitemap or crawler
    
    Args:
        url: The starting URL to scrape
        max_pages: Maximum number of pages to scrape (default: 10)
    
    Returns:
        List of dictionaries containing scraped page data
    """
    scraper = WebScraper()
    return scraper.scrape_website(url, max_pages)

if __name__ == "__main__":
    # Test the scraper
    test_url = "https://example.com"
    results = scrape_website(test_url, 5)
    print(f"Scraped {len(results)} pages from {test_url}")
    for result in results:
        print(f"URL: {result['url']}")
        print(f"Title: {result['title']}")
        print(f"Content length: {len(result['content'])} characters")
        print("---")
