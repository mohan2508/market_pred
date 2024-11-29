import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from datetime import date
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
from statsmodels.tsa.arima.model import ARIMA
from stocknews import StockNews 
import requests 
import json 

st.markdown('## Stock :orange[Web] :red[App]')
st.image("https://wallpaperaccess.com/full/2927403.jpg")
st.markdown('## Stock :red[Future] :green[Prediction]')

stocks=st.text_input('Enter Stock Ticker','SBIN.NS')
Start='2015-01-01'
Today = date.today().strftime("%Y-%m-%d")

data=yf.Ticker(stocks)
#selected_stocks=st.selectbox("Selected Stock dataset for prediction",stocks)
hys_data=data.history(period='10y')
hys_data.reset_index(inplace=True)
col1,col2,col3,col4,col5=st.columns(5, gap="medium")
st.divider()


data,prediction,chart,news,fundamental=st.tabs(['Data','Prediction','Chart','News','Fundamental Analysis'])
with data:
    st.subheader(":orange[Raw] :green[Data]")
    st.write(hys_data.tail(10))
with chart:
    def plot_raw_data():
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=hys_data['Date'],y=hys_data['Open'],name='stock_open'))
        fig.add_trace(go.Scatter(x=hys_data['Date'],y=hys_data['Close'],name='stock_close'))
        st.plotly_chart(fig)  
    #st.subheader(':red[Forecast of Raw Data]')    
    #plot_raw_data() 
    hys_data['Date']=pd.to_datetime(hys_data['Date']).dt.date 
    df_train=hys_data[['Date','Close']].rename(columns={'Date':'ds','Close':'y'})         
    m=Prophet(daily_seasonality=True)
    m.fit(df_train)
    future=m.make_future_dataframe(periods=30)
    forecast=m.predict(future)
    st.subheader(':green[Forecast data for Future 30days]')
    #st.write(forecast.head(30))
    fig2=m.plot_components(forecast)
    st.write(':orange[Forecast data]')
    fig1=plot_plotly(m,forecast)
    st.plotly_chart(fig1)
    st.write(':red[Forecast Trend]')
    st.write(fig2)    
#st.subheader(':red[Forecast of Raw Data]')    
#plot_raw_data()  

hys_data['Date']=pd.to_datetime(hys_data['Date']).dt.date
#Forcasting
with prediction:
    high,low,close=st.tabs(['High','Low','Close'])
    with high:
        df_train=hys_data[['Date','High']].rename(columns={'Date':'ds','High':'y'})
        m=Prophet(daily_seasonality=True)
        m.fit(df_train)
        future=m.make_future_dataframe(periods=30)
        forecast=m.predict(future)
        data1=hys_data['High']
        model=ARIMA(data1,order=(2,1,3))
        result=model.fit()
        forecast_steps=30
        forecast_values=result.predict(start=len(data1),end=len(data1)+forecast_steps-1,dynamic=False)
        st.write(forecast_values.head(15))
    with low:
        df_train=hys_data[['Date','Low']].rename(columns={'Date':'ds','Low':'y'})
        m=Prophet(daily_seasonality=True)
        m.fit(df_train)
        future=m.make_future_dataframe(periods=30)
        forecast=m.predict(future)
        data1=hys_data['Low']
        model=ARIMA(data1,order=(2,1,3))
        result=model.fit()
        forecast_steps=30
        forecast_values=result.predict(start=len(data1),end=len(data1)+forecast_steps-1,dynamic=False)
        st.write(forecast_values.head(15))   
    with close:
        df_train=hys_data[['Date','Close']].rename(columns={'Date':'ds','Close':'y'})
        m=Prophet(daily_seasonality=True)
        m.fit(df_train)
        future=m.make_future_dataframe(periods=30)
        forecast=m.predict(future)
        data1=hys_data['Close']
        model=ARIMA(data1,order=(2,1,3))
        result=model.fit()
        forecast_steps=30
        forecast_values=result.predict(start=len(data1),end=len(data1)+forecast_steps-1,dynamic=False)
        st.write(forecast_values.head(15))  
        
with news:
    st.header(f'News of {stocks}')
    sn=StockNews(stocks,save_news=False)
    df_news=sn.read_rss()
    for i in range(10):
        st.subheader(f'News : {i+1}')
        st.write(df_news['published'][i])
        st.write(df_news['title'][i])
        st.write(df_news['summary'][i])
        title_sentiment=df_news['sentiment_title'][i]
        st.write(f'Title Sentiment {title_sentiment}')
        news_sentiment=df_news['sentiment_summary'][i]
        st.write(f'News Sentiment {news_sentiment}')

with fundamental:
    ticker=yf.Ticker(stocks).info
    df=pd.DataFrame(ticker)
    st.header(f':orange[Fundamental Details of :] {stocks} ')
    st.write(df.head(1))
    
    fundamentals=['marketCap','trailingPE','totalRevenue','totalDebt',
                  'payoutRatio','earningsGrowth']
    df1=df[df.columns[df.columns.isin(fundamentals)]]
    st.divider()
    
    st.subheader(':green[Important Features]')
    col1,col2=st.columns(2)
    col1.write(f":red[Market Cap :] {df1['marketCap'][0]}")
    col1.write(f":orange[Trailing PE :] {df1['trailingPE'][0]}")
    col1.write(f":green[Total Revenue :] {df1['totalRevenue'][0]}")
    col2.write(f":red[Total Debt :] {df1['totalDebt'][0]}")
    col2.write(f":orange[Pay out Ratio:] {df1['payoutRatio'][0]}")
    col2.write(f":green[Earning Growth :] {df1['earningsGrowth'][0]}")
            
            
  
   
        
               
        
        
        
        

    





     
    





   







    



