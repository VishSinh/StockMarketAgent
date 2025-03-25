from flask import Flask, request, jsonify
import requests

from src.agent import StockRecommendationAgent
from src.utils.logger import setup_logger


app = Flask(__name__)
agent = StockRecommendationAgent()
logger = setup_logger()


@app.route('/stock', methods=['GET'])
def get_stock_recommendation():
    ticker = request.args.get('ticker', '').upper()
    
    stock = request.args.get('stock', '')
    
    if not ticker and not stock:
        logger.error("No ticker or stock name provided")
        return jsonify({"error": "Please provide either a ticker or stock name"}), 400

    try:
        recommendation = agent.generate_comprehensive_recommendation(ticker, stock)
        return jsonify(recommendation)
    except ValueError as e:
        logger.error(f"ValueError processing recommendation for {ticker}: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Error processing recommendation for {ticker}: {str(e)}", exc_info=True)
        return jsonify({"error": "An unexpected error occurred"}), 500



