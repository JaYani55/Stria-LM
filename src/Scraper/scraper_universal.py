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
            # This function will be run in a separate process
            q = multiprocessing.Queue()

            def spider_process(queue):
                try:
                    process = CrawlerProcess({
                        'USER_AGENT': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                        'ROBOTSTXT_OBEY': True,
                        'DOWNLOAD_DELAY': 1,
                        'RANDOMIZE_DOWNLOAD_DELAY': True,
                        'COOKIES_ENABLED': False,
                        'LOG_LEVEL': 'WARNING'
                    })
                    # Pass the spider *class* and its arguments to the crawler
                    process.crawl(UniversalSpider, start_url=start_url, max_pages=max_pages)
                    process.start() # This is a blocking call
                    # After the crawl is finished, we need to get the data.
                    # This part is tricky because the spider instance is not directly accessible.
                    # A common pattern is to use a custom pipeline to store data or use signals.
                    # For simplicity here, we'll rely on a workaround if possible, or recommend a redesign.
                    # Let's assume for now the spider is modified to output to a shared object.
                    # A queue is a more robust way to handle this.
                    # The spider would need to be modified to put its results in the queue.
                    # Let's modify the spider to accept a queue.
                    # This requires a change in UniversalSpider's __init__ and how it stores data.
                    # However, a simpler fix for the original error is just to pass the class.
                    # The data retrieval part is a separate problem.
                    # Let's assume the spider is designed to be run this way and data is retrieved later.
                    # The error is about passing an instance, not about data retrieval.
                    # The provided code `process.crawl(spider)` where `spider` is an instance is the issue.
                    # It should be `process.crawl(UniversalSpider, start_url=start_url, max_pages=max_pages)`
                    # The original code tries to get data back via `spider.scraped_data`, which won't work across processes.
                    # Let's fix the immediate error and address data passing.
                    
                    # The spider needs to be modified to handle this properly.
                    # Let's assume we can't change the spider for now and fix the call.
                    # The issue is that `process.start()` blocks and `spider.scraped_data` is in another process's memory.
                    
                    # A proper fix involves using a queue or another IPC mechanism.
                    # Let's implement that.
                    
                    # The spider needs to be modified to accept a queue.
                    # Let's make that change to UniversalSpider.
                    
                    # In UniversalSpider.__init__, add `self.queue = queue`
                    # In UniversalSpider.parse_page, instead of `self.scraped_data.append`, use `self.queue.put(page_data)`
                    
                    # But let's first try to fix the immediate error without deep refactoring.
                    # The error is `crawler_or_spidercls argument cannot be a spider object`.
                    # The fix is to pass the class.
                    
                    # The original code is:
                    # spider = UniversalSpider(start_url, max_pages)
                    # process.crawl(spider)
                    # process.start()
                    # return spider.scraped_data
                    
                    # This is fundamentally flawed for multiprocessing.
                    # Let's fix it the right way with a queue.
                    
                    # We will modify the spider to accept a queue.
                    # And then collect results from the queue.
                    
                    # This requires changing UniversalSpider.
                    # Let's do it.
                    
                    # The spider will be defined inside this function to make it self-contained.
                    
                    class SpiderWithQueue(UniversalSpider):
                        def __init__(self, *args, **kwargs):
                            self.queue = kwargs.pop('queue', None)
                            super().__init__(*args, **kwargs)
                            self.scraped_data = [] # Keep local copy for single-process case

                        def parse_page(self, response):
                            # This overrides the parent method
                            if self.pages_scraped >= self.max_pages:
                                return

                            self.pages_scraped += 1
                            
                            soup = BeautifulSoup(response.text, 'lxml')
                            for script in soup(["script", "style", "nav", "header", "footer"]):
                                script.decompose()
                            
                            text_content = soup.get_text(separator=' ', strip=True)
                            lines = (line.strip() for line in text_content.splitlines())
                            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                            text_content = ' '.join(chunk for chunk in chunks if chunk)
                            
                            page_data = {
                                'url': response.url,
                                'title': soup.title.string if soup.title else '',
                                'content': text_content,
                                'domain': urlparse(response.url).netloc
                            }
                            
                            if self.queue:
                                self.queue.put(page_data)
                            else:
                                self.scraped_data.append(page_data)
                            
                            logger.info(f"Scraped page {self.pages_scraped}/{self.max_pages}: {response.url}")
                            yield page_data

                    process.crawl(SpiderWithQueue, queue=queue, start_url=start_url, max_pages=max_pages)
                    process.start() # Blocking call
                    
                except Exception as e:
                    logger.error(f"Error in spider process: {e}")
                finally:
                    queue.put(None) # Sentinel value to indicate completion

            p = multiprocessing.Process(target=spider_process, args=(q,))
            p.start()
            
            results = []
            while True:
                item = q.get()
                if item is None:
                    break
                results.append(item)
            
            p.join()
            return results

        # Run spider in a separate process to avoid reactor issues
        # This is a more robust way to handle Scrapy in other applications
        ctx = multiprocessing.get_context('spawn')
        q = ctx.Queue()

        def spider_process(queue, start_url, max_pages):
            try:
                process = CrawlerProcess({
                    'USER_AGENT': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                    'ROBOTSTXT_OBEY': True,
                    'DOWNLOAD_DELAY': 1,
                    'RANDOMIZE_DOWNLOAD_DELAY': True,
                    'COOKIES_ENABLED': False,
                    'LOG_LEVEL': 'WARNING'
                })
                
                # The spider needs to be aware of the queue.
                # We'll pass it in the constructor.
                class SpiderForProcess(UniversalSpider):
                    def __init__(self, *args, **kwargs):
                        self.queue = kwargs.pop('queue', None)
                        super().__init__(*args, **kwargs)
                    
                    def parse_page(self, response):
                        # We override parse_page to put items in the queue
                        if self.pages_scraped >= self.max_pages:
                            return
                        
                        self.pages_scraped += 1
                        
                        soup = BeautifulSoup(response.text, 'lxml')
                        for script in soup(["script", "style", "nav", "header", "footer"]):
                            script.decompose()
                        
                        text_content = soup.get_text(separator=' ', strip=True)
                        lines = (line.strip() for line in text_content.splitlines())
                        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                        text_content = ' '.join(chunk for chunk in chunks if chunk)
                        
                        page_data = {
                            'url': response.url,
                            'title': soup.title.string if soup.title else '',
                            'content': text_content,
                            'domain': urlparse(response.url).netloc
                        }
                        
                        if self.queue:
                            self.queue.put(page_data)
                        
                        logger.info(f"Scraped page {self.pages_scraped}/{self.max_pages}: {response.url}")
                        # We don't need to yield here as we're using the queue
                
                process.crawl(SpiderForProcess, start_url=start_url, max_pages=max_pages, queue=q)
                process.start()
            except Exception as e:
                logger.error(f"Spider process failed: {e}")
            finally:
                # Signal that the process is done
                q.put(None)

        p = ctx.Process(target=spider_process, args=(q, start_url, max_pages))
        p.start()
        
        scraped_data = []
        while True:
            item = q.get()
            if item is None:
                break
            scraped_data.append(item)
        
        p.join()
        return scraped_data
    
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
