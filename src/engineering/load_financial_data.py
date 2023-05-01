from pandas_datareader import data as pdr
import yfinance as yfin
yfin.pdr_override()

def pull_yahoo_daily_close_price(ticker, start_date="2019-01-01", end_date="2023-01-01"):
    """
    pulls the daily adjusted close price for a given ticker or list of tickers
    between the given dates

    #Arguments

    ticker: either a list of strings or a single string of stock tickers

    start_date: start date as string formatted yyyy-mm-dd

    end_date: end date as string formatted yyy-mm-dd
    """
    prices = pdr.get_data_yahoo(ticker,
                                start=start_date,
                                end=end_date)
    
    return prices['Adj Close'].reset_index()