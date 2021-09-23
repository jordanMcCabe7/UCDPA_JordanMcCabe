# UCD Professional Academy
# Jordan McCabe 23/09/2021
# used pandas to import and manipulate data
# used matplotlib for data visualisation
# used scipy for linear regression
# used alphavantage for api data

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
import requests
from alpha_vantage.timeseries import TimeSeries

# Imported historical gold data
gold_d = pd.read_csv('/Users/Jordan/PycharmProjects/UCDPA_JordanMcCabe/monthly_csv.csv')

# print the head
print("Gold Dataframe :\n", gold_d.head())

# Import our historical CPI data
cpiDf = pd.read_csv('/Users/Jordan/PycharmProjects/UCDPA_JordanMcCabe/CPIAUCSL.csv')
# print the head
print("CPI Dataframe :\n", cpiDf.head())

# Inspected the gold dataframe
print(gold_d.info())
print(gold_d.shape)
print(gold_d.describe())
print(gold_d.columns)

# Create a Function that checks for null values
def nullcolumns(df): return df[df.isna().any(axis=1)]

print(nullcolumns(gold_d))

# Inspected the CPI dataframe
print(cpiDf.info())
print(cpiDf.shape)
print(cpiDf.describe())
print(cpiDf.columns)

# Change the date format
gold_d["Date"] = pd.to_datetime(gold_d["Date"]).dt.strftime('%d-%m-%Y')
cpiDf["DATE"] = pd.to_datetime(cpiDf["DATE"]).dt.strftime('%d-%m-%Y')

# Summarised statistics for gold_d
goldMinD = gold_d["Date"].min()
goldMaxD = gold_d["Date"].max()
print("GoldMinDt: ", goldMinD)
print("GoldMaxDt: ", goldMaxD)
print("GoldMedP: ", gold_d["Price"].median())
print("GoldMinP: ", gold_d["Price"].min())
print("GoldMinP: ", gold_d["Price"].max())
print("GoldV: ", gold_d["Price"].var())
print("GoldSD: ", gold_d["Price"].std())

# look for Null Values in Gold
nullGold = gold_d.isna().sum().sum()
print("There are: ", nullGold, " Null values in the Gold Dataset")

# look for Null Values in CPI
nullCpi = cpiDf.isna().sum().sum()
print("There are: ", nullCpi, " Null values in the CPI Dataset")

# Insert a new column that shows name of the asset we are using , in this case Gold
gold_d.insert(0, "Name", "Gold")
# Insert a new column that shows name of the asset we are using , in this case CPI
cpiDf.insert(0, "Name", "CPI")

# Double check null values
dbl_nulls = cpiDf.isna().sum().sum()
print("There are: ", dbl_nulls, " Null values in the CPI Dataset")

# Rename CPIAUCSL to Price Index
cpiDf.rename(columns={'CPIAUCSL': 'Price Index'}, inplace=True)
print(cpiDf.head())

# Rename DATE to Date
cpiDf.rename(columns={'DATE': 'Date'}, inplace=True)
print(cpiDf.head())

# Confirm that dates are the same data type
print("The CPI Date Column is of Datatype: ", cpiDf["Date"].dtype)
print("The Gold Date Column is of Datatype: ", gold_d["Date"].dtype)

# As previously seen, the gold data doesnt have as many date values as the cpi data
# We will check the minimum and maximum date range for this
goldMinD = gold_d["Date"].min()
goldMaxD = gold_d["Date"].max()
cpiMinD = cpiDf["Date"].min()
cpiMaxD = cpiDf["Date"].max()

# also checked the minimum and maximum price
goldMaxPrice = gold_d["Price"].max()
cpiMaxPrice = cpiDf["Price Index"].max()
print("Gold Minimum Date: ", goldMinD, " CPI Minimum Date: ", cpiMinD, " Gold Max Price: ", goldMaxPrice)
print("Gold Maximum Date: ", goldMaxD, " CPI Maximum Date: ", cpiMaxD, " CPI Max Price", cpiMaxPrice)

# Create Scatter Plot for gold
gold_d.plot(x="Date", y="Price", kind="scatter", title="Gold Outliers")
plt.show()

# Create Scatter Plot for cpi
cpiDf.plot(x="Date", y="Price Index", kind="scatter", title="CPI Outliers")
plt.show()

# merge the two dataframes to peform analysis
goldCpi = gold_d.merge(cpiDf, on="Date")

# Check for nulls in the Join
gcpi_null = goldCpi.isna().sum().sum()
print("There are: ", gcpi_null, " Null values in the GoldCPI Dataset")


#  I wish to look at price volatility over the last 5 years
goldCpi["Date"] = pd.to_datetime(goldCpi["Date"])
rangeMask = (goldCpi["Date"] > "2016-01-01") & (goldCpi["Date"] <= "2020-01-01")
print(goldCpi.loc[rangeMask])

# Plot the goldCpi
date = goldCpi["Date"]
goldPrice = goldCpi["Price"]
cpiPrice = goldCpi["Price Index"]

# Plot Historical Gold Data
plt.plot(date, goldPrice,  linestyle="--", label="Gold")
plt.xlabel("Date")

# Plot Histrical CPI Data
plt.plot(date, cpiPrice,  linestyle="solid", label="CPI")
plt.ylabel("Price $")
plt.title("Historical CPI & Gold Prices")
plt.legend()
plt.show()

# Calculate the monthly % change
goldCpi.insert(5, "GOLD Change", "")
goldCpi.insert(6, "CPI Change", "")
print(goldCpi.head())

goldCpi["GOLD Change"] = \
    (goldCpi["Price"].diff() * 100) / goldCpi["Price"].shift()
goldCpi["CPI Change"] = \
    (goldCpi["Price Index"].diff() * 100) / goldCpi["Price Index"].shift()

with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(goldCpi)

# Replace NaN with 0 as it is the first value and will be zero
goldCpi["GOLD Change"] = goldCpi["GOLD Change"].fillna(0)
goldCpi["CPI Change"] = goldCpi["CPI Change"].fillna(0)

# Get the Maximum Volatility for CPI
cpiVolatility = goldCpi["CPI Change"].max()

# Get the Maximum volatility for Gold
goldVolatility = goldCpi["GOLD Change"].max()

# Get the Mean Volatility for CPI
cpiVolatilityMean = goldCpi["CPI Change"].mean()

# Get the Mean volatility for Gold
goldVolatilityMean = goldCpi["GOLD Change"].mean()

# 5 years volatility
data_5yrs = [goldCpi["Date"], goldCpi["GOLD Change"], goldCpi["CPI Change"]]
header = ["Date", "Gold Vol 5 years", "CPI Vol 5 years"]
volatility_5yrs = pd.concat(data_5yrs, axis=1, keys=header)
volatility_5yrs["Date"] = pd.to_datetime(goldCpi["Date"])
dateMask5 = (volatility_5yrs["Date"] >= "2016-01-01") & (volatility_5yrs["Date"] <= "2021-01-01")
volatility_5yrs = volatility_5yrs.loc[dateMask5]
print(volatility_5yrs)

# plotting for volatility
date5 = volatility_5yrs["Date"]
gold5 = volatility_5yrs["Gold Vol 5 years"]
cpi5 = volatility_5yrs["CPI Vol 5 years"]

# 5 Years plot
plt.plot(date5, gold5,  linestyle="--", label="Gold")
plt.xlabel("Date")
plt.plot(date5, cpi5, linestyle="solid", label="CPI")
plt.ylabel("Volatility of price")
plt.title("Gold & CPI Price Volatility 5 years")
plt.legend()
plt.show()

# plot gold against cpi
plt.plot(goldCpi["GOLD Change"], goldCpi["CPI Change"], '.')
plt.xlabel('GOLD Change')
plt.ylabel('CPI Change')
slope, intercept, r_value, p_value, std_err = linregress(goldCpi["GOLD Change"], goldCpi["CPI Change"])
x = np.linspace(goldCpi["GOLD Change"].min(), goldCpi["GOLD Change"].max())
plt.plot(x, slope * x + intercept, 'k')
print('beta = ', slope)
print('Corr = ', r_value)
plt.show()

# print s&p 500 data from api to demonstrate use of api
# this was done using alpha-vantage api which can be used to access stock data at varying frequencies
api_key = 'Q9QYBP879YCQ4KJ6'
ts = TimeSeries(key=api_key, output_format='pandas')
data, meta_data = ts.get_monthly_adjusted(symbol='VOO')
print(data.info())
print(data.head())