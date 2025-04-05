from django.shortcuts import render
from django.http import JsonResponse
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta
import time
import re
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
import os
import json
import traceback
import logging

from django.shortcuts import render
from django.http import JsonResponse
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from newspaper import Article
import yfinance as yf
import os
import logging
import traceback
from google import genai
from dotenv import load_dotenv
from activity.views import TopTradedAssetsView

load_dotenv()
logger = logging.getLogger(__name__)



class Bot_Response(APIView):
    """
    API view for stock risk analysis
    """

    def post(self, request):
        """
        Handle POST requests with stock ticker information
        """
        try:
            # Extract ticker from request data
            data = request.data
            ticker = data.get('ticker')

            if not ticker:
                return Response(
                    {"error": "Please provide a stock ticker symbol"},
                    status=status.HTTP_400_BAD_REQUEST
                )

            # Perform stock analysis
            risk_metrics = self.analyze_stock_risk(ticker)

            return Response(risk_metrics, status=status.HTTP_200_OK)

        except Exception as e:
            logger.error(f"Error in API: {str(e)}")
            logger.error(traceback.format_exc())
            return Response(
                {"error": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    def analyze_stock_risk_with_langchain(self, metrics):
        """
        Use LangChain with OpenAI to analyze stock risk metrics
        """
        # Ensure API key is set
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")

        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)

        template = """
        You are a financial analyst specializing in stock risk assessment.
        Based on the following metrics, provide a detailed risk analysis and classify the stock as Low, Medium, or High risk.
        Explain your reasoning clearly and in simple terms.

        Metrics:
        - Daily Return: {Daily_Return}
        - Volatility: {Volatility}
        - RSI: {RSI}
        - MACD: {MACD}
        - Bollinger Bands (Upper): {BB_upper}
        - Bollinger Bands (Lower): {BB_lower}
        - Sharpe Ratio: {Sharpe_Ratio}

        Respond with:
        1. Risk Classification (Low, Medium, High)
        2. Reasoning behind the classification

        The output format must be in JSON format with the following keys:
        - risk_classification
        - reasoning
        """

        prompt = PromptTemplate.from_template(template)
        final_prompt = prompt.format(**metrics)
        response = llm.predict(final_prompt)
        return response

    def _safe_float(self, value):
        """Convert various numeric types to float safely"""
        if pd.isna(value) or value is None:
            return 0.0
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    def analyze_stock_risk(self, ticker):
        """
        Analyze risk metrics for a single stock ticker.

        Parameters:
            ticker (str): The stock ticker symbol (e.g., 'AAPL', 'MSFT').

        Returns:
            dict: A dictionary containing calculated risk metrics for the stock.
        """
        # Download historical data
        today = time.strftime("%Y-%m-%d")

        try:
            df = yf.download(ticker, start='2020-01-01', end=today)

            # Check if we got any data
            if df.empty:
                return {
                    "ticker": ticker,
                    "error": "No data found for this ticker. Please verify the ticker symbol."
                }

            # Handle MultiIndex columns if present (newer yfinance versions)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0] for col in df.columns]

            # Calculate daily returns safely
            df['Daily_Return'] = df['Close'].pct_change().fillna(0)

            # Calculate volatility (21-day rolling standard deviation annualized)
            df['Volatility'] = df['Daily_Return'].rolling(window=21).std().fillna(0) * np.sqrt(252)

            # Calculate moving averages (50-day and 200-day)
            df['MA_50'] = df['Close'].rolling(window=50).mean().fillna(df['Close'])
            df['MA_200'] = df['Close'].rolling(window=200).mean().fillna(df['Close'])

            # Calculate RSI using pandas_ta with error handling
            try:
                df['RSI'] = df.ta.rsi(close=df['Close'], length=14).fillna(50)
            except Exception as e:
                logger.warning(f"Error calculating RSI: {str(e)}. Using default values.")
                df['RSI'] = pd.Series([50] * len(df), index=df.index)

            # Calculate MACD with error handling
            try:
                macd_result = df.ta.macd(close=df['Close'], fast=12, slow=26, signal=9)
                if isinstance(macd_result, pd.DataFrame):
                    df['MACD'] = macd_result.iloc[:, 0].fillna(0)  # First column should be MACD
                else:
                    col_name = 'MACD_12_26_9'
                    df['MACD'] = macd_result[col_name] if col_name in macd_result else pd.Series([0] * len(df),
                                                                                                 index=df.index)
            except Exception as e:
                logger.warning(f"Error calculating MACD: {str(e)}. Using default values.")
                df['MACD'] = pd.Series([0] * len(df), index=df.index)

            # Calculate Bollinger Bands with error handling
            try:
                bb_bands = df.ta.bbands(close=df['Close'], length=20)
                if isinstance(bb_bands, pd.DataFrame):
                    df['BB_upper'] = bb_bands.iloc[:, 0].fillna(df['Close'] * 1.1)  # Default to 10% above price
                    df['BB_middle'] = bb_bands.iloc[:, 1].fillna(df['Close'])
                    df['BB_lower'] = bb_bands.iloc[:, 2].fillna(df['Close'] * 0.9)  # Default to 10% below price
                else:
                    df['BB_upper'] = bb_bands.get('BBU_20_2.0',
                                                  pd.Series([df['Close'].iloc[-1] * 1.1] * len(df), index=df.index))
                    df['BB_middle'] = bb_bands.get('BBM_20_2.0',
                                                   pd.Series([df['Close'].iloc[-1]] * len(df), index=df.index))
                    df['BB_lower'] = bb_bands.get('BBL_20_2.0',
                                                  pd.Series([df['Close'].iloc[-1] * 0.9] * len(df), index=df.index))
            except Exception as e:
                logger.warning(f"Error calculating Bollinger Bands: {str(e)}. Using default values.")
                df['BB_upper'] = df['Close'] * 1.1
                df['BB_middle'] = df['Close']
                df['BB_lower'] = df['Close'] * 0.9

            # Calculate Sharpe Ratio safely
            mean_return = df['Daily_Return'].mean()
            volatility = df['Volatility'].mean()
            sharpe_ratio = mean_return / volatility if volatility and volatility != 0 else 0

            # Get the most recent data row with valid values
            valid_rows = df.dropna(subset=['RSI', 'MACD', 'BB_upper', 'BB_lower']).tail(1)
            if not valid_rows.empty:
                latest_row = valid_rows.iloc[0]
            else:
                latest_row = df.iloc[-1]  # Use last row anyway if no valid rows

            # Extract the latest values for key metrics with safe conversion
            latest_data = {
                'ticker': ticker,
                'latest_close': self._safe_float(latest_row['Close']),
                'daily_return': self._safe_float(mean_return),
                'volatility': self._safe_float(volatility),
                'ma_50': self._safe_float(latest_row['MA_50']),
                'ma_200': self._safe_float(latest_row['MA_200']),
                'rsi': self._safe_float(latest_row['RSI']),
                'macd': self._safe_float(latest_row['MACD']),
                'bb_upper': self._safe_float(latest_row['BB_upper']),
                'bb_middle': self._safe_float(latest_row['BB_middle']),
                'bb_lower': self._safe_float(latest_row['BB_lower']),
                'sharpe_ratio': self._safe_float(sharpe_ratio),
            }

            metrics = {
                "Daily_Return": latest_data['daily_return'],
                "Volatility": latest_data['volatility'],
                "RSI": latest_data['rsi'],
                "MACD": latest_data['macd'],
                "BB_upper": latest_data['bb_upper'],
                "BB_lower": latest_data['bb_lower'],
                "Sharpe_Ratio": latest_data['sharpe_ratio'],
            }

            try:
                result = self.analyze_stock_risk_with_langchain(metrics)

                # First try regex matching for JSON
                match = re.search(r'{\s*"risk_classification":\s*"(.+?)",\s*"reasoning":\s*"(.+?)"\s*}', result,
                                  re.DOTALL)

                if match:
                    risk_classification = match.group(1)
                    reasoning = match.group(2)
                    latest_data['risk_classification'] = risk_classification
                    latest_data['summary'] = reasoning
                else:
                    # Try to find and parse the first JSON object in the response
                    json_match = re.search(r'({[\s\S]*?})', result)
                    if json_match:
                        try:
                            json_result = json.loads(json_match.group(1))
                            latest_data['risk_classification'] = json_result.get('risk_classification', 'Unknown')
                            latest_data['summary'] = json_result.get('reasoning', 'No analysis available')
                        except json.JSONDecodeError:
                            latest_data['risk_classification'] = 'Unknown'
                            latest_data['summary'] = 'Could not parse JSON from LLM response'
                            latest_data['raw_response'] = result
                    else:
                        # If no JSON format detected, use a basic classification detection
                        result_lower = result.lower()
                        if "high risk" in result_lower:
                            latest_data['risk_classification'] = "High"
                        elif "medium risk" in result_lower:
                            latest_data['risk_classification'] = "Medium"
                        elif "low risk" in result_lower:
                            latest_data['risk_classification'] = "Low"
                        else:
                            latest_data['risk_classification'] = "Unknown"

                        # Use the full response as summary
                        latest_data['summary'] = result
                        latest_data['parsing_note'] = "Used text-based classification as JSON wasn't found"

            except Exception as e:
                logger.error(f"Error in LLM analysis: {str(e)}")
                latest_data['risk_classification'] = "Error"
                latest_data['summary'] = f"Error performing risk analysis: {str(e)}"

            return latest_data

        except Exception as e:
            logger.error(f"Error analyzing stock {ticker}: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                "ticker": ticker,
                "error": f"Failed to analyze stock: {str(e)}"
            }




class Trade_News(APIView):
    """
    API view for fetching and summarizing financial news for top traded stock tickers
    """

    def get(self, request):
        """
        Handle GET requests with optional ticker parameters
        If no tickers provided, uses top traded assets from activity app

        Example usage:
        - With specific tickers: /api/chatbot/news/?tickers=AAPL,MSFT,GOOG
        - For top traded assets: /api/chatbot/news/
        """
        try:
            # Get tickers from query params if provided
            ticker_param = request.query_params.get('tickers', '')

            if ticker_param:
                # Use provided tickers
                tickers = ticker_param.split(',')
                source = 'user_specified'
            else:
                # Get top traded assets from the activity app
                try:
                    # Create instance of TopTradedAssetsView to get data
                    top_assets_view = TopTradedAssetsView()
                    top_assets_response = top_assets_view.get(request)
                    top_assets_data = top_assets_response.data

                    # Extract ticker symbols from top assets data
                    tickers = [asset['asset_name'] for asset in top_assets_data]
                    source = 'top_traded'

                    # If no top assets found, use default tickers
                    if not tickers:
                        tickers = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'TSLA']
                        source = 'default'

                except Exception as e:
                    logger.error(f"Error fetching top assets: {str(e)}")
                    # Fall back to default tickers
                    tickers = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'TSLA']
                    source = 'default (after error)'

            # Limit number of tickers to prevent abuse
            tickers = tickers[:5]

            # Fetch news for the tickers
            news_data = self.get_news(tickers)

            return Response({
                "status": "success",
                "source": source,
                "tickers": tickers,
                "data": news_data
            }, status=status.HTTP_200_OK)

        except Exception as e:
            logger.error(f"Error in Trade_News API: {str(e)}")
            logger.error(traceback.format_exc())
            return Response({
                "status": "error",
                "message": str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    def generate_text(self, text):
        """
        Generate a summary of news article using Google's Gemini model
        """
        try:
            # Get API key from environment variable
            GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
            if not GEMINI_API_KEY:
                return "API key not found. Please set the GEMINI_API_KEY environment variable."

            # Configure the Gemini client
            client = genai.Client(api_key=GEMINI_API_KEY)

            # Create prompt for summarization
            prompt = f"You are a finance expert. Please provide a summary of following news: {text} generate summary give complete text only"

            # Generate the summary
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[prompt]
            )

            return response.text

        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            return f"Unable to generate summary: {str(e)}"

    def get_news(self, tickers):
        """
        Fetch news articles for a list of ticker symbols
        """
        articles = {}

        for ticker in tickers:
            try:
                # Search for news related to the ticker
                search_result = yf.Search(ticker, max_results=10, news_count=10, include_research=True)
                news_data = search_result.news

                ticker_articles = []

                # Process each article
                for article_info in news_data[:3]:  # Limit to 3 articles per ticker
                    try:
                        article = Article(article_info['link'])
                        article.download()
                        article.parse()

                        # Only process substantial articles
                        if len(article.text.strip()) >= 500:
                            summary = self.generate_text(article.text[:4000])  # Limit text length for API

                            ticker_articles.append({
                                'title': article_info['title'],
                                'publisher': article_info.get('publisher', 'Unknown'),
                                'link': article_info['link'],
                                'published': article_info.get('providerPublishTime', ''),
                                'summary': summary
                            })

                            if len(ticker_articles) >= 2:  # Limit to 2 articles per ticker
                                break

                    except Exception as e:
                        logger.warning(f"Error processing article for {ticker}: {str(e)}")
                        continue

                if ticker_articles:
                    articles[ticker] = ticker_articles
                else:
                    articles[ticker] = [{"info": "No substantial news articles found for this ticker"}]

            except Exception as e:
                logger.error(f"Error fetching news for {ticker}: {str(e)}")
                articles[ticker] = [{"error": f"Failed to fetch news: {str(e)}"}]

        return articles