import json
import re
import os
import requests

import yfinance as yf
from duckduckgo_search import DDGS
from bs4 import BeautifulSoup
from langchain_community.document_loaders import BSHTMLLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

class StockData(BaseModel):
    close_price: float = Field(..., description="Current closing price")
    sma_50: float = Field(..., description="50-day simple moving average")
    sma_200: float = Field(..., description="200-day simple moving average")
    rsi: float = Field(..., description="Relative Strength Index")
    volume: float = Field(..., description="Trading volume")

class WebScraperFetcher:
    def __init__(self):
        # Initialize LangChain components
        self.ticker = ""
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0)
        self.parser = JsonOutputParser(pydantic_object=StockData)
        
        self.prompt = ChatPromptTemplate.from_template(
            """Extract financial metrics from HTML. Focus on:
            - Current closing price
            - 50-day and 200-day moving averages
            - Relative Strength Index (RSI)
            - Trading volume
            
            HTML Content:
            {html_content}
            
            {format_instructions}"""
        )
        self.chain = self.prompt | self.llm | self.parser

    def fetch_stock_data(self, ticker):
        self.ticker = ticker
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
            
            # Try traditional parsing first
            stock_data = self.parse_traditional(html_content)
            
            # Fallback to LangChain if needed
            if not stock_data:
                stock_data = self.parse_with_langchain(html_content)
            
            if not stock_data:
                return {"error": f"Failed to parse data for '{ticker}'"}
            
            return {"ticker": ticker, "data_trends": [stock_data]}
            
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

    def parse_traditional(self, html_content):
        """Traditional scraping method for Yahoo Finance"""
        try:
            soup = BeautifulSoup(html_content, "html.parser")
            script = soup.find("script", text=re.compile(r"root\.App\.main"))
            
            if not script:
                return None
                
            json_data = json.loads(
                script.string.split("root.App.main =")[1]
                .split("}(this)")[0].strip()[:-1]
            )
            
            main_data = json_data["context"]["dispatcher"]["stores"]["QuoteSummaryStore"]
            
            return {
                "Close": main_data["price"]["regularMarketPrice"]["raw"],
                "SMA_50": main_data["summaryDetail"]["fiftyDayAverage"]["raw"],
                "SMA_200": main_data["summaryDetail"]["twoHundredDayAverage"]["raw"],
                "RSI": main_data.get("technicalInsights", {})
                          .get("technicalEvents", {})
                          .get("rsi", {}).get("rsi", 0),
                "Volume": main_data["summaryDetail"]["volume"]["raw"]
            }
        except Exception as e:
            print(f"Traditional parsing failed: {e}")
            return None

    def parse_with_langchain(self, html_content):
        """LLM-powered parsing fallback"""
        try:
            # Save temporary HTML file
            temp_path = "temp_yahoo.html"
            with open(temp_path, "w", encoding="utf-8") as f:
                f.write(html_content)
            
            # Load and process HTML
            loader = BSHTMLLoader(temp_path)
            docs = loader.load()
            
            # Trim content to first 10k characters
            content = docs[0].page_content[:10000]
            
            # Run processing chain
            result = self.chain.invoke({
                "html_content": content,
                "format_instructions": self.parser.get_format_instructions()
            })
            
            return {
                "ticker": self.ticker,
                "trends" : {
                    "Close": result["close_price"],
                    "SMA_50": result["sma_50"],
                    "SMA_200": result["sma_200"],
                    "RSI": result["rsi"],
                    "Volume": result["volume"]
                }
            }
            
            
        except Exception as e:
            print(f"LangChain parsing failed: {e}")
            return None
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

class YahooFinanceFetcher:
    def fetch_stock_data(self, ticker, period="1y", interval="1d"):
        try:
            stock = yf.Ticker(ticker)
            stock_data = stock.history(period=period, interval=interval)

            if stock_data.empty:
                return {"error": f"No data found for '{ticker}'."}

            stock_data["SMA_50"] = stock_data["Close"].rolling(window=50).mean()
            stock_data["SMA_200"] = stock_data["Close"].rolling(window=200).mean()
            stock_data["RSI"] = self.compute_rsi(stock_data["Close"])

            stock_trends = stock_data[["Close", "SMA_50", "SMA_200", "RSI"]].dropna().tail(50).to_dict(orient="records")
            return {"ticker": ticker, "data_trends": stock_trends}

        except Exception as e:
            return {"error": f"Error fetching stock data: {str(e)}"}

    def compute_rsi(self, series, period=14):
        delta = series.diff(1)
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=period, min_periods=1).mean()
        avg_loss = loss.rolling(window=period, min_periods=1).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

# class WebScraperFetcher:
#     def fetch_stock_data(self, ticker):
#         try:
#             # Search for the correct Yahoo Finance quote page
#             query = f"{ticker} stock quote site:finance.yahoo.com"
#             results = DDGS().text(query, max_results=3)
            
#             if not results:
#                 return {"error": f"No results found for '{ticker}'"}
            
#             # Find the correct URL from search results
#             yahoo_url = next((r["href"] for r in results if "/quote/" in r["href"]), None)
#             print("Yahoo URL:", yahoo_url)
#             if not yahoo_url:
#                 return {"error": f"Yahoo Finance page not found for '{ticker}'"}
            
#             # Scrape and parse page data
#             page_data = self.scrape_page(yahoo_url)
#             # print("Page data:", page_data)
#             if not page_data:
#                 return {"error": f"Failed to retrieve data for '{ticker}'"}
            
#             stock_data = self.parse_yahoo_data(page_data)
#             print("Stock data:", stock_data)
#             if not stock_data:
#                 return {"error": f"Failed to parse data for '{ticker}'"}
            
#             return {"ticker": ticker, "data_trends": [stock_data]}
            
#         except Exception as e:
#             return {"error": f"Unexpected error: {str(e)}"}

#     def scrape_page(self, url):
#         headers = {
#             "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
#         }
#         try:
#             response = requests.get(url, headers=headers, timeout=10)
#             response.raise_for_status()
#             return response.text
#         except Exception as e:
#             print(f"Error fetching page: {e}")
#             return None

#     def parse_yahoo_data(self, html_content):
#         try:
#             soup = BeautifulSoup(html_content, "html.parser")
            
#             # Locate all script tags
#             scripts = soup.find_all("script")
#             target_script = None
            
#             # Iterate over script tags to find the one containing 'root.App.main'
#             for script in scripts:
#                 if script.string and 'root.App.main' in script.string:
#                     target_script = script
#                     break
            
#             if not target_script:
#                 print("Target script containing 'root.App.main' not found.")
#                 return None
            
#             # Extract JSON data from the script tag
#             script_content = target_script.string
#             json_text_match = re.search(r'root\.App\.main\s*=\s*(\{.*\})\s*;\n', script_content)
#             if not json_text_match:
#                 print("Could not extract JSON text from the script.")
#                 return None
            
#             json_text = json_text_match.group(1)
#             json_data = json.loads(json_text)
            
#             # Navigate through the JSON structure to find relevant data
#             main_data = json_data.get("context", {}).get("dispatcher", {}).get("stores", {}).get("QuoteSummaryStore", {})
            
#             # Extract current price data
#             price_data = main_data.get("price", {}).get("regularMarketPrice", {})
#             rsi_data = main_data.get("technicalInsights", {}).get("technicalEvents", {}).get("rsi", {})
            
#             return {
#                 "Close": price_data.get("raw", 0),
#                 "SMA_50": main_data.get("summaryDetail", {}).get("fiftyDayAverage", {}).get("raw", 0),
#                 "SMA_200": main_data.get("summaryDetail", {}).get("twoHundredDayAverage", {}).get("raw", 0),
#                 "RSI": rsi_data.get("rsi", 0),
#                 "Volume": main_data.get("summaryDetail", {}).get("volume", {}).get("raw", 0)
#             }
            
#         except Exception as e:
#             print(f"Parsing error: {e}")
#             return None