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

        the output formate should be in JSON format with the following keys:
        - risk_classification
        - reasoning
        """

        prompt = PromptTemplate.from_template(template)
        final_prompt = prompt.format(**metrics)
        response = llm.predict(final_prompt)
        return response

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
        df = yf.download(ticker, start='2020-01-01', end=today)

        # Ensure column names are consistent
        df.columns = [col if isinstance(col, str) else col[0] for col in df.columns]

        # Calculate daily returns
        df['Daily_Return'] = df['Close'].pct_change()

        # Calculate volatility (21-day rolling standard deviation annualized)
        df['Volatility'] = df['Daily_Return'].rolling(window=21).std() * np.sqrt(252)

        # Calculate moving averages (50-day and 200-day)
        df['MA_50'] = df['Close'].rolling(window=50).mean()
        df['MA_200'] = df['Close'].rolling(window=200).mean()

        # Calculate Relative Strength Index (RSI)
        df['RSI'] = df.ta.rsi(length=14)

        # Calculate MACD (Moving Average Convergence Divergence)
        macd = df.ta.macd(fast=12, slow=26, signal=9)
        df['MACD'] = macd['MACD_12_26_9']

        # Calculate Bollinger Bands
        bb_bands = df.ta.bbands(close=df['Close'], length=20)
        df['BB_upper'] = bb_bands['BBU_20_2.0']
        df['BB_middle'] = bb_bands['BBM_20_2.0']
        df['BB_lower'] = bb_bands['BBL_20_2.0']

        # Calculate Sharpe Ratio (Risk-adjusted return)
        mean_return = df['Daily_Return'].mean()
        volatility = df['Volatility'].mean()
        sharpe_ratio = mean_return / volatility if volatility != 0 else 0

        # Drop rows with NaN values to clean the data
        df = df.dropna()

        # Extract the latest values for key metrics
        latest_data = {
            'ticker': ticker,
            'latest_close': float(df.iloc[-1]['Close']),
            'daily_return': float(mean_return),
            'volatility': float(volatility),
            'ma_50': float(df.iloc[-1]['MA_50']),
            'ma_200': float(df.iloc[-1]['MA_200']),
            'rsi': float(df.iloc[-1]['RSI']),
            'macd': float(df.iloc[-1]['MACD']),
            'bb_upper': float(df.iloc[-1]['BB_upper']),
            'bb_middle': float(df.iloc[-1]['BB_middle']),
            'bb_lower': float(df.iloc[-1]['BB_lower']),
            'sharpe_ratio': float(sharpe_ratio),
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

        result = self.analyze_stock_risk_with_langchain(metrics)
        match = re.search(r'{\s*"risk_classification":\s*"(.+?)",\s*"reasoning":\s*"(.+?)"\s*}', result, re.DOTALL)

        if match:
            risk_classification = match.group(1)
            reasoning = match.group(2)
            latest_data['risk_classification'] = risk_classification
            latest_data['summary'] = reasoning
        else:
            # Try to parse the whole response as JSON
            try:
                json_result = json.loads(result)
                latest_data['risk_classification'] = json_result.get('risk_classification', 'Unknown')
                latest_data['summary'] = json_result.get('reasoning', 'No analysis available')
            except json.JSONDecodeError:
                latest_data['risk_classification'] = 'Unknown'
                latest_data['summary'] = 'Could not parse LLM response'
                latest_data['raw_response'] = result

        return latest_data