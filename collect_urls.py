import asyncio
import aiohttp
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

async def afrom_website(start_url, url_limit=100, concurrency=10, pattern=None, debug=False):
    """
    Asynchronously traverses a website and collects URLs up to the specified limit.
    
    Args:
        start_url (str): The starting URL to begin traversal
        url_limit (int): Maximum number of URLs to collect
        concurrency (int): Number of concurrent requests
        pattern (str): Optional string pattern that URLs must contain
        
    Returns:
        list: List of collected URLs
    """
    # Extract domain from start_url to stay on the same website
    base_domain = urlparse(start_url).netloc
    
    # Initialize collections for tracking
    collected_urls = []
    visited_urls = set()
    urls_to_visit = [start_url]
    
    # Semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(concurrency)
    
    async def process_url(url):
        # Skip if already visited or collected enough URLs
        if url in visited_urls or len(collected_urls) >= url_limit:
            return
        
        # Mark as visited
        visited_urls.add(url)
        
        try:
            # Use semaphore to limit concurrent requests
            async with semaphore:
                # Make request with a user agent to avoid being blocked
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
                
                async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=10) ) as response:
                    # Only process successful responses
                    if response.status == 200:
                        # Add to collected URLs if it matches the pattern (if specified)
                        if pattern is None or pattern in url:
                            collected_urls.append(url)
                            if debug:
                                print(f"Collected: {url} ({len(collected_urls)}/{url_limit})")
                        
                        # Don't process further if we've reached the limit
                        if len(collected_urls) >= url_limit:
                            return
                        
                        # Parse the page to find more links
                        html = await response.text()
                        soup = BeautifulSoup(html, 'html.parser')
                        
                        new_urls = []
                        for a_tag in soup.find_all('a', href=True):
                            href = a_tag['href']
                            full_url = urljoin(url, href)
                            
                            # Only follow links on the same domain that we haven't visited yet
                            parsed_url = urlparse(full_url)
                            if (parsed_url.netloc == base_domain and 
                                full_url not in visited_urls and 
                                full_url not in urls_to_visit and
                                full_url not in new_urls):
                                new_urls.append(full_url)
                        
                        # Add new URLs to the queue
                        urls_to_visit.extend(new_urls)
                        return new_urls
        
        except Exception as e:
            print(f"Error processing {url}: {e}")
            return []
    
    # Create a session for all requests
    async with aiohttp.ClientSession() as session:
        while urls_to_visit and len(collected_urls) < url_limit:
            # Get a batch of URLs to process
            batch_size = min(concurrency, len(urls_to_visit))
            current_batch = [urls_to_visit.pop(0) for _ in range(batch_size)]
            
            # Process the batch concurrently
            tasks = [process_url(url) for url in current_batch]
            await asyncio.gather(*tasks)
    
    return collected_urls
