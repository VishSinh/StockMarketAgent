from os import getenv as os_getenv
from langchain.agents import initialize_agent, AgentType
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_core.output_parsers import JsonOutputParser

from src.tools import StockDataTool, MarketAnalysisTool, StockNameToTickerTool
from src.utils.logger import setup_logger

logger = setup_logger()

class StockRecommendationAgent:
    def __init__(self):
        logger.info("Initializing StockRecommendationAgent")
        api_key = os_getenv("OPENAI_API_KEY")
        if not api_key:
            logger.error("OPENAI_API_KEY not found in environment variables")
            raise ValueError("OPENAI_API_KEY not found")
        
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.3,
            api_key=api_key
        )

        # Create memory for contextual understanding
        self.memory = ConversationBufferMemory(
            memory_key="chat_history", 
            return_messages=True
        )

        # Define tools
        self.tools = [
            StockDataTool(),
            MarketAnalysisTool(),
            StockNameToTickerTool()
        ]

        # Create agent with advanced reasoning
        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            memory=self.memory
        )
        logger.info("StockRecommendationAgent initialized successfully")
        
        
    def get_ticker_symbol(self, ticker, symbol):
        try:
            if not ticker and symbol:
                logger.info(f"Converting stock name to ticker: {symbol}")
                ticker = self.tools[2]._run(symbol)
                
            if not ticker:
                raise ValueError("Could not find Ticker for the Stock Name")
            
            return ticker
            
        except Exception as e:
            logger.error(f"Error converting stock name to ticker: {str(e)}", exc_info=True)
            raise ValueError(f"Failed to find ticker for {symbol}: {str(e)}")
       


    def generate_comprehensive_recommendation(self, ticker: str = None, stock: str = None):
        try:
            ticker = self.get_ticker_symbol(ticker, stock)

            # Step 1: Retrieve Stock Data
            logger.info(f"Step 1: Retrieving stock data for {ticker}")
            stock_data = self.tools[0]._run(ticker)
            
            # Step 2: Perform Market Analysis
            logger.info(f"Step 2: Performing market analysis for {ticker}")
            market_analysis = self.tools[1]._run(stock_data)
            
            # Step 3: Generate Recommendation using LLM
            logger.info(f"Step 3: Generating LLM recommendation for {ticker}")
            recommendation_prompt = PromptTemplate(
                input_variables=["stock_data", "market_analysis"],
                template="""
                As a senior financial analyst, provide a comprehensive stock recommendation.

                Stock Data: {stock_data}
                Market Analysis: {market_analysis}

                Generate a JSON response with:
                - recommendation (Buy/Hold/Sell)
                - confidence_score
                - key_insights
                - risk_assessment
                - price_target
                """
            )

            recommendation_chain = LLMChain(
                llm=self.llm, 
                prompt=recommendation_prompt,
                output_parser=JsonOutputParser()
            )

            recommendation = recommendation_chain.run({
                "stock_data": stock_data,
                "market_analysis": market_analysis
            })
            
            logger.info(f"Successfully generated recommendation for {ticker}")
            return recommendation

        except ValueError as e:
            logger.error(f"ValueError generating recommendation for {ticker}: {str(e)}", exc_info=True)
            raise ValueError(f"Failed to generate recommendation: {str(e)}")
        except Exception as e:
            logger.error(f"Error generating recommendation for {ticker}: {str(e)}", exc_info=True)
            raise Exception(f"Comprehensive analysis failed: {str(e)}")
