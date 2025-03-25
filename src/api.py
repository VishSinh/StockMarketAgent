import json
from flask import Flask, request, jsonify
from stock_fetcher import YahooFinanceFetcher, WebScraperFetcher
from price_analyzer import OpenAIAnalyzer, LangChainAnalyzer
# from new import WebScraperFetcher, StockAnalyzer


app = Flask(__name__)

# Initialize fetchers and analyzers
yahoo_fetcher = YahooFinanceFetcher()
scraper_fetcher = WebScraperFetcher()
openai_analyzer = OpenAIAnalyzer()
langchain_analyzer = LangChainAnalyzer()

@app.route("/stock", methods=["GET"])
def get_stock_recommendation():
    ticker = request.args.get("ticker", "").upper()

    if not ticker:
        return jsonify({"error": "No ticker provided."}), 400


    stock_data = yahoo_fetcher.fetch_stock_data(ticker)
    if "error" in stock_data:
        return jsonify(stock_data), 400
        
    # print("YahooFinance data:", stock_data)

    recommendation_data, summary = openai_analyzer.get_stock_insight(stock_data)
    if "error" in recommendation_data:
        return jsonify(recommendation_data), 500
    
    openai_response = {
        "ticker": recommendation_data["ticker"],
        "analysis_summary": recommendation_data["analysis_summary"],
        "recommendation": recommendation_data["recommendation"],
        "reasoning": recommendation_data["reasoning"],
        "risk_assessment": recommendation_data["risk_assessment"],
        "confidence_score": recommendation_data["confidence_score"],
        "raw_summary": summary  # Optional raw summary for debugging
    }
    
    recommendation_data, summary = langchain_analyzer.get_stock_insight(stock_data)


    return jsonify({
        "openai": openai_response,
        "langchain": recommendation_data
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
