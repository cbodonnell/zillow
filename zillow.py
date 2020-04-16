import csv
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression


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

    columns = headers[:4] + ['Date', 'SalePrice']
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
                    datetime.date(year_month[0], year_month[1], 1).toordinal(),
                    int(sale_price) if sale_price else 0
                ]
                items.append(item)

    df = pd.DataFrame(items, columns=columns)
    print(df)
    # df = df.loc[df['StateName'].isin(['Delaware'])]
    df = df.loc[df['RegionName'].isin(['19808'])]
    print(df)
    # dates = df['Date'].values.reshape(-1, 1)
    # prices = df['SalePrice'].values.reshape(-1, 1)
    #
    # # Numpy
    # print(dates)
    # # print(prices)
    # z = np.poly1d(np.polyfit(dates, prices, 1))
    #
    # # Sklearn
    # linear_regressor = LinearRegression()  # create object for the class
    # linear_regressor.fit(dates, prices)  # perform linear regression
    # print(linear_regressor.intercept_)  # intercept
    # print(linear_regressor.coef_)  # slope
    # Y_pred = linear_regressor.predict(prices)  # make predictions
    #
    # plt.scatter(dates, prices)
    # plt.plot(dates, z, color='red')
    # plt.show()

    df = df.pivot(index='Date', columns='RegionName', values='SalePrice')
    print(df)
    df.plot()
    plt.show()
