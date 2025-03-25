import json
from openai import OpenAI
import os
from langchain_openai import ChatOpenAI
from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate

class LangChainAnalyzer:
    def __init__(self):
        # Initialize the chat model with GPT-4
        self.llm = ChatOpenAI(model="gpt-4o")
        
        # Define the prompt template with proper JSON formatting guidance
        self.prompt_template = PromptTemplate(
            input_variables=["stock_summary"],
            template="""You are a senior financial analyst. Analyze the following stock data and provide:
1. A clear recommendation (Buy/Hold/Sell)
2. Detailed technical analysis
3. Key support/resistance levels
4. Risk assessment

Format your response as JSON with these keys:
- recommendation
- analysis
- support_level
- resistance_level
- risk_level
- price_target

Stock Data:
{stock_summary}"""
        )
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt_template)

    def prepare_stock_summary(self, stock_data):
        if "error" in stock_data:
            return stock_data["error"]

        ticker = stock_data["ticker"]
        trends = stock_data["data_trends"]

        summary = [
            f"Analysis for {ticker} (Last 50 Days):",
            f"Latest Close: ${trends[-1]['Close']:.2f}",
            f"50-Day SMA: ${trends[-1]['SMA_50']:.2f}",
            f"200-Day SMA: ${trends[-1]['SMA_200']:.2f}",
            f"RSI: {trends[-1]['RSI']:.2f}",
            "\nRecent Trends:"
        ]
        
        # Add last 5 days' summary
        for i, day in enumerate(trends[-5:]):
            summary.append(
                f"Day {i+1}: Close ${day['Close']:.2f} | "
                # f"Vol. {day['Volume']:,} | "
                f"50SMA ${day['SMA_50']:.2f} | "
                f"RSI {day['RSI']:.2f}"
            )
            
        return "\n".join(summary)

    def get_stock_insight(self, stock_data):
        summary = self.prepare_stock_summary(stock_data)
        
        try:
            # Using the modern invoke() method with structured input
            response = self.chain.invoke({"stock_summary": summary})
            print(response)
            json_string = response["text"].strip()
            if json_string.startswith("```json"):
                json_string = json_string[7:]
            if json_string.endswith("```"):
                json_string = json_string[:-3]
            recommendation_data = json.loads(json_string)
        except json.JSONDecodeError:
            return {"error": "Failed to parse model response"}, summary
        except Exception as e:
            return {"error": f"Analysis error: {str(e)}"}, summary

        return recommendation_data, summary
    
    

class OpenAIAnalyzer:
    def prepare_stock_summary(self, stock_data):
        if "error" in stock_data:
            return stock_data["error"]

        ticker = stock_data["ticker"]
        trends = stock_data["data_trends"]

        summary = f"Stock data analysis for {ticker} over the last 50 days:\n"
        for i, trend in enumerate(trends[-5:]):  # Last 5 days
            close_price = trend.get('Close')
            sma_50 = trend.get('SMA_50')
            sma_200 = trend.get('SMA_200')
            rsi = trend.get('RSI')

            # Handle None values
            close_price_str = f"${close_price:.2f}" if close_price is not None else "N/A"
            sma_50_str = f"${sma_50:.2f}" if sma_50 is not None else "N/A"
            sma_200_str = f"${sma_200:.2f}" if sma_200 is not None else "N/A"
            rsi_str = f"{rsi:.2f}" if rsi is not None else "N/A"

            summary += (
                f"Day {i+1}: Close price: {close_price_str}, "
                f"50-day SMA: {sma_50_str}, "
                f"200-day SMA: {sma_200_str}, "
                f"RSI: {rsi_str}\n"
            )
        return summary

    def get_stock_insight(self, stock_data):
        summary = self.prepare_stock_summary(stock_data)
        
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        
        PROMPT = """
        You are a financial analyst providing detailed stock recommendations in JSON format.
        Given the following stock data, analyze and provide a recommendation in JSON format:
        {
        "ticker": "<Stock Ticker>",
        "analysis_summary": "<Summary of trends>",
        "recommendation": "<Buy/Sell/Hold>",
        "reasoning": "<Justification>",
        "risk_assessment": "<Potential risks>",
        "confidence_score": "<Confidence level (0-100)>"
        }
        """
        
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": PROMPT},
                {"role": "user", "content": summary},
            ],
            max_tokens=300,
            temperature=0.7,
            response_format={'type': 'json_object'},
        )
        
        try:
            recommendation_data = json.loads(completion.choices[0].message.content.strip())
        except Exception as e:
            recommendation_data = {"error": f"Error parsing JSON: {e}"}
        
        return recommendation_data, summary



