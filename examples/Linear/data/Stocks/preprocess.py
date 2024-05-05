import pandas as pd


etth1 = pd.read_csv('../ETT/ETTh1.csv')
column_etth1 = etth1['date']
stock1 = pd.read_csv('NVDA.csv')
stock1 = stock1.rename(columns={'time':'date'})
stock2 = pd.read_csv('AMD.csv')
stock2 = stock2.rename(columns={'time':'date'})

column_stock1 = stock1['date']
column_stock2 = stock2['date']
stock1['date'] = pd.to_datetime(stock1['date'])
stock2['date'] = pd.to_datetime(stock2['date'])

stock1['date'] = stock1['date'].dt.strftime('%Y-%m-%d %H:%M:%S')
stock2['date'] = stock2['date'].dt.strftime('%Y-%m-%d %H:%M:%S')

stock1.to_csv('NVDA.csv', index=False)
stock2.to_csv('AMD.csv', index=False)
