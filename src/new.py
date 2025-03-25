import os
import json
from duckduckgo_search import DDGS
import requests
from bs4 import BeautifulSoup
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from openai import OpenAI

class WebScraperFetcher:
    def fetch_stock_data(self, ticker):
        try:
            results = DDGS().text(
                f"{ticker} stock quote site:finance.yahoo.com", 
                max_results=3
            )
            
            if not results:
                return {"error": f"No results found for '{ticker}'"}
            
            yahoo_url = next(
                (r["href"] for r in results if "/quote/" in r["href"]), 
                None
            )
            if not yahoo_url:
                return {"error": f"Yahoo Finance page not found for '{ticker}'"}
            
            html_content = self.scrape_page(yahoo_url)
            if not html_content:
                return {"error": f"Failed to retrieve data for '{ticker}'"}
            
            stock_data = self.parse_yahoo_data(html_content)
            if not stock_data:
                return {"error": f"Failed to parse data for '{ticker}'"}
            
            return {"ticker": ticker, "data_trends": stock_data}
            
        except Exception as e:
            return {"error": f"Unexpected error: {str(e)}"}

    def scrape_page(self, url):
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        try:
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            return response.text
        except Exception as e:
            print(f"Error fetching page: {e}")
            return None

    def parse_yahoo_data(self, html_content):
        try:
            soup = BeautifulSoup(html_content, "html.parser")
            script = soup.find("script", text=lambda t: t and "root.App.main" in t)
            
            if not script:
                return self.fallback_langchain_parse(html_content)
                
            json_data = json.loads(
                script.string.split("root.App.main =")[1]
                .split("}(this)")[0].strip()[:-1]
            )
            
            main_data = json_data["context"]["dispatcher"]["stores"]["QuoteSummaryStore"]
            
            return [{
                "Close": main_data["price"]["regularMarketPrice"]["raw"],
                "SMA_50": main_data["summaryDetail"]["fiftyDayAverage"]["raw"],
                "SMA_200": main_data["summaryDetail"]["twoHundredDayAverage"]["raw"],
                "RSI": main_data.get("technicalInsights", {})
                          .get("technicalEvents", {})
                          .get("rsi", {}).get("rsi", 0)
            }]
        except Exception as e:
            print(f"Traditional parsing failed: {e}")
            return self.fallback_langchain_parse(html_content)

    def fallback_langchain_parse(self, html_content):
        try:
            client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{
                    "role": "system",
                    "content": """Extract stock data from HTML. Find:
                    - Closing price
                    - 50-day SMA
                    - 200-day SMA
                    - RSI
                    Return JSON with keys: Close, SMA_50, SMA_200, RSI"""
                }, {
                    "role": "user",
                    "content": html_content[:15000]  # Truncate to stay under token limits
                }],
                response_format={"type": "json_object"},
                temperature=0
            )
            
            return [json.loads(response.choices[0].message.content)]
        except Exception as e:
            print(f"LangChain parsing failed: {e}")
            return None

class StockAnalyzer:
    def __init__(self, use_langchain=True):
        self.use_langchain = use_langchain
        if use_langchain:
            self.llm = ChatOpenAI(model="gpt-4o")
            self.prompt_template = PromptTemplate(
                template="""Analyze stock data and provide:
                - Recommendation (Buy/Hold/Sell)
                - Technical analysis
                - Support/resistance levels
                - Risk assessment
                Format as JSON with keys: recommendation, analysis, support_level, resistance_level, risk_level, price_target

                Data: {stock_summary}""",
                input_variables=["stock_summary"]
            )
            self.chain = LLMChain(llm=self.llm, prompt=self.prompt_template)
        else:
            self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    def prepare_summary(self, stock_data):
        if "error" in stock_data:
            return stock_data["error"]
            
        trends = stock_data["data_trends"]
        latest = trends[0]
        return (
            f"Ticker: {stock_data['ticker']}\n"
            f"Latest Close: ${latest['Close']:.2f}\n"
            f"50-Day SMA: ${latest['SMA_50']:.2f}\n"
            f"200-Day SMA: ${latest['SMA_200']:.2f}\n"
            f"RSI: {latest['RSI']:.1f}\n"
            f"Historical Trends: {len(trends)} data points"
        )

    def analyze(self, stock_data):
        summary = self.prepare_summary(stock_data)
        
        if "error" in stock_data:
            return {"error": stock_data["error"]}, summary
            
        try:
            if self.use_langchain:
                response = self.chain.invoke({"stock_summary": summary})
                result = json.loads(response["text"])
            else:
                response = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{
                        "role": "system",
                        "content": "Provide stock analysis in JSON format with keys: recommendation, analysis, support_level, resistance_level, risk_level, price_target"
                    }, {
                        "role": "user",
                        "content": summary
                    }],
                    response_format={"type": "json_object"},
                    temperature=0.3
                )
                result = json.loads(response.choices[0].message.content)
                
            # Validate response structure
            required_keys = {"recommendation", "analysis", "support_level", "resistance_level", "risk_level"}
            if not all(k in result for k in required_keys):
                raise ValueError("Missing required keys in response")
                
            return result, summary
            
        except Exception as e:
            return {"error": f"Analysis failed: {str(e)}"}, summary

# # Usage Example
# if __name__ == "__main__":
#     os.environ["OPENAI_API_KEY"] = "your-api-key-here"
    
#     fetcher = WebScraperFetcher()
#     analyzer = StockAnalyzer(use_langchain=True)
    
#     stock_data = fetcher.fetch_stock_data("AAPL")
#     if "error" not in stock_data:
#         analysis, summary = analyzer.analyze(stock_data)
#         print("Summary:\n", summary)
#         print("\nAnalysis:\n", json.dumps(analysis, indent=2))
#     else:
#         print("Error:", stock_data["error"])