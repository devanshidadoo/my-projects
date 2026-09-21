import trafilatura
import requests

class NewsFetcherTool:
    """
    LangChain tool for news fetcher that scrapes news from a URL.
    """
    name: str = "NewsFetcherTool"
    description: str = """
    Scrapes news from a URL.
    Takes a news URL as input.
    """
    def __init__(self):
        pass

    def scrape(self, url: str):
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        downloaded = trafilatura.extract(response.text, url=url)
        return downloaded   

if __name__ == "__main__":
    URL = "https://www.thehindu.com/news/national/jaishankar-hits-out-at-us-tariffs-says-india-will-not-compromise-on-farmers-interests/article69967779.ece"
    scraper = NewsFetcherTool()
    print(scraper.scrape(URL))

