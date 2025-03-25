import yfinance as yf
import requests
from typing import Dict, Any
from langchain.tools import BaseTool

from src.utils.logger import setup_logger


logger = setup_logger()

class StockDataTool(BaseTool):
    name: str = "stock_data_retrieval"
    description: str = "Retrieve comprehensive stock market data and financial metrics"

    def _run(self, ticker: str) -> Dict[str, Any]:
        logger.info(f"Fetching stock data for ticker: {ticker}")
        try:
            stock = yf.Ticker(ticker)
            
            data = {
                "basic_info": {
                    "longName": stock.info.get('longName', 'N/A'),
                    "sector": stock.info.get('sector', 'N/A'),
                    "industry": stock.info.get('industry', 'N/A')
                },
                "financial_metrics": {
                    "marketCap": stock.info.get('marketCap', 0),
                    "trailingPE": stock.info.get('trailingPE', 0),
                    "dividendYield": stock.info.get('dividendYield', 0),
                    "beta": stock.info.get('beta', 0)
                },
                "price_history": stock.history(period="1y").to_dict()
            }
            logger.info(f"Successfully retrieved stock data for {ticker}")
            return data
        except Exception as e:
            logger.error(f"Error fetching stock data for {ticker}: {str(e)}", exc_info=True)
            raise Exception(f"Failed to retrieve stock data for {ticker}: {str(e)}")

class MarketAnalysisTool(BaseTool):
    name: str = "market_analysis"
    description: str = "Perform in-depth market and technical analysis"

    def _run(self, stock_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            analysis = {
                "technical_indicators": self._calculate_technical_indicators(stock_data),
                "trend_analysis": self._analyze_trends(stock_data),
                "risk_assessment": self._assess_risk(stock_data)
            }
            return analysis
        except Exception as e:
            logger.error(f"Error performing market analysis: {str(e)}", exc_info=True)
            raise Exception(f"Failed to perform market analysis: {str(e)}")

    def _calculate_technical_indicators(self, stock_data):
        try:
            price_history = stock_data.get('price_history', {})
            close_prices = price_history.get('Close', {})
            
            if not close_prices:
                raise ValueError("Insufficient price data")

            prices = list(close_prices.values())
            
            # Calculate moving averages
            sma_20 = sum(prices[-20:]) / min(20, len(prices))
            sma_50 = sum(prices[-50:]) / min(50, len(prices))
            
            # Calculate RSI
            def calculate_rsi(prices, periods=14):
                deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
                gain = [x if x > 0 else 0 for x in deltas]
                loss = [-x if x < 0 else 0 for x in deltas]
                
                avg_gain = sum(gain[-periods:]) / periods
                avg_loss = sum(loss[-periods:]) / periods
                
                if avg_loss == 0:
                    return 100
                
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
                return rsi

            # Calculate MACD
            def calculate_macd(prices):
                ema_12 = sum(prices[-12:]) / 12  # Simplified EMA
                ema_26 = sum(prices[-26:]) / 26
                macd = ema_12 - ema_26
                return macd

            return {
                "sma_20": sma_20,
                "sma_50": sma_50,
                "rsi": calculate_rsi(prices),
                "macd": calculate_macd(prices),
                "current_price": prices[-1],
                "price_change": ((prices[-1] / prices[-2]) - 1) * 100 if len(prices) > 1 else 0
            }
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {str(e)}", exc_info=True)
            raise Exception(f"Failed to calculate technical indicators: {str(e)}")

    def _analyze_trends(self, stock_data):
        try:
            price_history = stock_data.get('price_history', {})
            close_prices = price_history.get('Close', {})
            
            if not close_prices:
                raise ValueError("Insufficient price data")

            prices = list(close_prices.values())
            
            # Determine trend direction
            short_term_trend = "bullish" if prices[-1] > prices[-5] else "bearish"
            medium_term_trend = "bullish" if prices[-1] > prices[-20] else "bearish"
            long_term_trend = "bullish" if prices[-1] > prices[-50] else "bearish"
            
            # Calculate volatility
            returns = [(prices[i] / prices[i-1] - 1) for i in range(1, len(prices))]
            volatility = (sum([r**2 for r in returns]) / len(returns))**0.5 * (252**0.5)  # Annualized
            
            # Volume trend (assuming volume data is available)
            volume_data = price_history.get('Volume', {})
            volume_trend = "increasing" if volume_data and list(volume_data.values())[-1] > sum(list(volume_data.values())[-5:]) / 5 else "decreasing"

            return {
                "short_term_trend": short_term_trend,
                "medium_term_trend": medium_term_trend,
                "long_term_trend": long_term_trend,
                "volatility": volatility,
                "volume_trend": volume_trend,
                "support_level": min(prices[-20:]),
                "resistance_level": max(prices[-20:])
            }
        except Exception as e:
            logger.error(f"Error analyzing trends: {str(e)}", exc_info=True)
            raise Exception(f"Failed to analyze trends: {str(e)}")

    def _assess_risk(self, stock_data):
        try:
            # Get basic financial metrics
            financial_metrics = stock_data.get('financial_metrics', {})
            basic_info = stock_data.get('basic_info', {})
            
            # Calculate risk factors
            market_cap_risk = "high" if financial_metrics.get('marketCap', 0) < 2e9 else \
                             "medium" if financial_metrics.get('marketCap', 0) < 10e9 else "low"
            
            beta = financial_metrics.get('beta', 1)
            beta_risk = "high" if beta > 1.5 else "medium" if beta > 1 else "low"
            
            # PE ratio analysis
            pe_ratio = financial_metrics.get('trailingPE', 0)
            valuation_risk = "high" if pe_ratio > 30 or pe_ratio < 0 else \
                            "medium" if pe_ratio > 20 else "low"
            
            # Sector risk assessment
            sector = basic_info.get('sector', 'Unknown')
            sector_risk = {
                'Technology': 'medium',
                'Healthcare': 'low',
                'Financial': 'medium',
                'Energy': 'high',
                'Consumer Cyclical': 'medium',
                'Consumer Defensive': 'low'
            }.get(sector, 'medium')

            return {
                "overall_risk_score": (
                    {"high": 3, "medium": 2, "low": 1}[market_cap_risk] +
                    {"high": 3, "medium": 2, "low": 1}[beta_risk] +
                    {"high": 3, "medium": 2, "low": 1}[valuation_risk]
                ) / 3,  # Normalized to 1-3 scale
                "risk_factors": {
                    "market_cap_risk": market_cap_risk,
                    "beta_risk": beta_risk,
                    "valuation_risk": valuation_risk,
                    "sector_risk": sector_risk
                },
                "beta": beta,
                "sector": sector,
                "market_cap": financial_metrics.get('marketCap', 0)
            }
        except Exception as e:
            logger.error(f"Error assessing risk: {str(e)}", exc_info=True)
            raise Exception(f"Failed to assess risk: {str(e)}")

class StockNameToTickerTool(BaseTool):
    name: str = "stock_name_to_ticker"
    description: str = "Convert company name to stock ticker symbol"

    def _run(self, stock_name: str) -> str:
        logger.info(f"Converting stock name to ticker: {stock_name}")
        try:
            yfinance = "https://query2.finance.yahoo.com/v1/finance/search"
            user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36'
            params = {"q": stock_name, "quotes_count": 1, "country": "United States"}

            res = requests.get(url=yfinance, params=params, headers={'User-Agent': user_agent})
            data = res.json()

            if not data or 'quotes' not in data or not data['quotes']:
                raise ValueError(f"No ticker found for company name: {stock_name}")

            ticker = data['quotes'][0]['symbol']
            logger.info(f"Found ticker {ticker} for company {stock_name}")
            return ticker
        except Exception as e:
            logger.error(f"Error converting stock name to ticker: {str(e)}", exc_info=True)
            raise ValueError(f"Failed to find ticker for {stock_name}: {str(e)}")