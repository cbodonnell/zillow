import csv
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression


def avg_pct_change(x, df_recent):
    price = x['SalePrice']
    recent_price = df_recent.loc[df_recent['RegionName'] == x['RegionName'], 'SalePrice']
    pct_change = (recent_price - price) / price
    return pct_change.values[0]


def score_old(x, df_recent):
    price = x['SalePrice']
    recent_price = df_recent.loc[df_recent['RegionName'] == x['RegionName'], 'SalePrice']
    pct_change = (recent_price - price) / price
    score = (2019 - x['Year']) * pct_change
    return score.values[0]


def score_new(x, df_recent):
    price = x['SalePrice']
    recent_price = df_recent.loc[df_recent['RegionName'] == x['RegionName'], 'SalePrice']
    pct_change = (recent_price - price) / price
    score = pct_change / (2019 - x['Year'])
    return score.values[0] * 10


if __name__ == '__main__':
    results = []
    with open("data/Sale_Prices_Zip.csv") as f:
        reader = csv.reader(f)
        for row in reader:
            results.append(row)


    headers = results[0]
    data = results[1:]

    data_indices = {}
    for i, header in enumerate(headers):
        data_indices[header] = i

    columns = headers[:4] + ['Year', 'Date', 'SalePrice']
    months = headers[4:]

    items = []

    for row in data:
        for month in months:
            sale_price = row[data_indices[month]]
            if sale_price:
                year_month = [int(d) for d in month.split('-')]
                item = [
                    row[0],
                    row[1],
                    row[2],
                    row[3],
                    year_month[0],
                    datetime.date(year_month[0], year_month[1], 1),
                    int(sale_price) if sale_price else 0
                ]
                items.append(item)

    df = pd.DataFrame(items, columns=columns)
    df = df.loc[df['StateName'].isin(['Delaware'])]
    df = df.loc[df['RegionName'].isin(['19720', '19963', '19804', '19901', '19904', '19973', '19977'])]
    df = df.loc[df['Year'] > 2012]
    df.pivot(index='Date', columns='RegionName', values='SalePrice').plot()
    print(df)
    df = df.groupby(['RegionName', 'Year']).mean().reset_index()
    print(df)
    df_recent = df.loc[df['Year'].isin([2019])]
    # df = df.loc[df['Year'].isin([2018, 2016, 2014, 2009])]
    print(df)
    df['AvgPctChange'] = df.apply(lambda x: avg_pct_change(x, df_recent), axis=1)
    df['ScoreOld'] = df.apply(lambda x: score_old(x, df_recent), axis=1)
    df['ScoreNew'] = df.apply(lambda x: score_new(x, df_recent), axis=1)
    # df = df.loc[df['RegionName'].isin(['19808'])]
    df = df.groupby(['RegionName']).mean().reset_index()
    df = df.drop(['Year', 'SalePrice'], axis=1)
    df['AvgRank'] = df['AvgPctChange'].rank(ascending=False)
    df['OldRank'] = df['ScoreOld'].rank(ascending=False)
    df['NewRank'] = df['ScoreNew'].rank(ascending=False)
    df['TotalRank'] = df.apply(lambda x: x['AvgRank'] + x['OldRank'] + x['NewRank'], axis=1)
    print(df)
    df.plot.bar(x='RegionName', y=['AvgRank', 'OldRank', 'NewRank'], rot=0)
    df.plot.bar(x='RegionName', y=['TotalRank'], rot=0)
    plt.show()
