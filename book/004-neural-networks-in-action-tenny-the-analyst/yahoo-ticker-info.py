import yfinance as yf

# Define the ticker symbols
ticker = "AAPL"

stock = yf.Ticker(ticker)

# Fetch the information
info = stock.info

# Convert the keys to a list and sort them
sorted_keys = sorted(list(info.keys()))

print(f"{info.get('longName')}")

# Print the sorted keys
for key in sorted_keys:
    print(f"{key}: {info.get(key)}")

# Apple Inc.
# 52WeekChange: 0.39208603
# SandP52WeekChange: 0.20686829
# address1: One Apple Park Way
# ask: 180.81
# askSize: 900
# auditRisk: 4
# averageDailyVolume10Day: 51248110
# averageVolume: 54028051
# averageVolume10days: 51248110
# beta: 1.29
# bid: 180.75
# bidSize: 1200
# boardRisk: 1
# bookValue: 3.997
# city: Cupertino
# companyOfficers: [{'maxAge': 1, 'name': 'Mr. Timothy D. Cook', 'age': 62, 'title': 'CEO & Director', 'yearBorn': 1961, 'fiscalYear': 2022, 'totalPay': 16425933, 'exercisedValue': 0, 'unexercisedValue': 0}, {'maxAge': 1, 'name': 'Mr. Luca  Maestri', 'age': 60, 'title': 'CFO & Senior VP', 'yearBorn': 1963, 'fiscalYear': 2022, 'totalPay': 5019783, 'exercisedValue': 0, 'unexercisedValue': 0}, {'maxAge': 1, 'name': 'Mr. Jeffrey E. Williams', 'age': 59, 'title': 'Chief Operating Officer', 'yearBorn': 1964, 'fiscalYear': 2022, 'totalPay': 5018337, 'exercisedValue': 0, 'unexercisedValue': 0}, {'maxAge': 1, 'name': 'Ms. Katherine L. Adams', 'age': 59, 'title': 'Senior VP, General Counsel & Secretary', 'yearBorn': 1964, 'fiscalYear': 2022, 'totalPay': 5015208, 'exercisedValue': 0, 'unexercisedValue': 0}, {'maxAge': 1, 'name': "Ms. Deirdre  O'Brien", 'age': 56, 'title': 'Senior Vice President of Retail', 'yearBorn': 1967, 'fiscalYear': 2022, 'totalPay': 5019783, 'exercisedValue': 0, 'unexercisedValue': 0}, {'maxAge': 1, 'name': 'Mr. Chris  Kondo', 'title': 'Senior Director of Corporate Accounting', 'fiscalYear': 2022, 'exercisedValue': 0, 'unexercisedValue': 0}, {'maxAge': 1, 'name': 'Mr. James  Wilson', 'title': 'Chief Technology Officer', 'fiscalYear': 2022, 'exercisedValue': 0, 'unexercisedValue': 0}, {'maxAge': 1, 'name': 'Suhasini  Chandramouli', 'title': 'Director of Investor Relations', 'fiscalYear': 2022, 'exercisedValue': 0, 'unexercisedValue': 0}, {'maxAge': 1, 'name': 'Mr. Greg  Joswiak', 'title': 'Senior Vice President of Worldwide Marketing', 'fiscalYear': 2022, 'exercisedValue': 0, 'unexercisedValue': 0}, {'maxAge': 1, 'name': 'Mr. Adrian  Perica', 'age': 49, 'title': 'Head of Corporate Development', 'yearBorn': 1974, 'fiscalYear': 2022, 'exercisedValue': 0, 'unexercisedValue': 0}]
# compensationAsOfEpochDate: 1672444800
# compensationRisk: 6
# country: United States
# currency: USD
# currentPrice: 181.18
# currentRatio: 0.988
# dateShortInterest: 1702598400
# dayHigh: 182.76
# dayLow: 180.17
# debtToEquity: 199.418
# dividendRate: 0.96
# dividendYield: 0.0053
# earningsGrowth: 0.135
# earningsQuarterlyGrowth: 0.108
# ebitda: 125820002304
# ebitdaMargins: 0.32827
# enterpriseToEbitda: 22.892
# enterpriseToRevenue: 7.515
# enterpriseValue: 2880222527488
# exDividendDate: 1699574400
# exchange: NMS
# fiftyDayAverage: 187.3978
# fiftyTwoWeekHigh: 199.62
# fiftyTwoWeekLow: 128.12
# financialCurrency: USD
# firstTradeDateEpochUtc: 345479400
# fiveYearAvgDividendYield: 0.8
# floatShares: 15535488445
# forwardEps: 7.15
# forwardPE: 25.339859
# freeCashflow: 82179997696
# fullTimeEmployees: 161000
# gmtOffSetMilliseconds: -18000000
# governanceEpochDate: 1704067200
# grossMargins: 0.44131002
# grossProfits: 170782000000
# heldPercentInsiders: 0.00074
# heldPercentInstitutions: 0.61495996
# impliedSharesOutstanding: 15752800256
# industry: Consumer Electronics
# industryDisp: Consumer Electronics
# industryKey: consumer-electronics
# lastDividendDate: 1699574400
# lastDividendValue: 0.24
# lastFiscalYearEnd: 1696032000
# lastSplitDate: 1598832000
# lastSplitFactor: 4:1
# longBusinessSummary: Apple Inc. designs, manufactures, and markets smartphones, personal computers, tablets, wearables, and accessories worldwide. The company offers iPhone, a line of smartphones; Mac, a line of personal computers; iPad, a line of multi-purpose tablets; and wearables, home, and accessories comprising AirPods, Apple TV, Apple Watch, Beats products, and HomePod. It also provides AppleCare support and cloud services; and operates various platforms, including the App Store that allow customers to discover and download applications and digital content, such as books, music, video, games, and podcasts. In addition, the company offers various services, such as Apple Arcade, a game subscription service; Apple Fitness+, a personalized fitness service; Apple Music, which offers users a curated listening experience with on-demand radio stations; Apple News+, a subscription news and magazine service; Apple TV+, which offers exclusive original content; Apple Card, a co-branded credit card; and Apple Pay, a cashless payment service, as well as licenses its intellectual property. The company serves consumers, and small and mid-sized businesses; and the education, enterprise, and government markets. It distributes third-party applications for its products through the App Store. The company also sells its products through its retail and online stores, and direct sales force; and third-party cellular network carriers, wholesalers, retailers, and resellers. Apple Inc. was founded in 1976 and is headquartered in Cupertino, California.
# longName: Apple Inc.
# marketCap: 2817856110592
# maxAge: 86400
# messageBoardId: finmb_24937
# mostRecentQuarter: 1696032000
# netIncomeToCommon: 96995000320
# nextFiscalYearEnd: 1727654400
# numberOfAnalystOpinions: 39
# open: 181.99
# operatingCashflow: 110543003648
# operatingMargins: 0.30134
# overallRisk: 1
# payoutRatio: 0.1533
# pegRatio: 4.5
# phone: 408 996 1010
# previousClose: 181.91
# priceHint: 2
# priceToBook: 45.328995
# priceToSalesTrailing12Months: 7.3518558
# profitMargins: 0.25305998
# quickRatio: 0.843
# quoteType: EQUITY
# recommendationKey: buy
# recommendationMean: 2.2
# regularMarketDayHigh: 182.76
# regularMarketDayLow: 180.17
# regularMarketOpen: 181.99
# regularMarketPreviousClose: 181.91
# regularMarketVolume: 62379661
# returnOnAssets: 0.20256001
# returnOnEquity: 1.7195
# revenueGrowth: -0.007
# revenuePerShare: 24.344
# sector: Technology
# sectorDisp: Technology
# sectorKey: technology
# shareHolderRightsRisk: 1
# sharesOutstanding: 15552799744
# sharesPercentSharesOut: 0.0077
# sharesShort: 120233720
# sharesShortPreviousMonthDate: 1700006400
# sharesShortPriorMonth: 105837123
# shortName: Apple Inc.
# shortPercentOfFloat: 0.0077
# shortRatio: 2.29
# state: CA
# symbol: AAPL
# targetHighPrice: 250.0
# targetLowPrice: 159.0
# targetMeanPrice: 199.57
# targetMedianPrice: 200.0
# timeZoneFullName: America/New_York
# timeZoneShortName: EST
# totalCash: 61554999296
# totalCashPerShare: 3.958
# totalDebt: 123930001408
# totalRevenue: 383285002240
# trailingAnnualDividendRate: 0.94
# trailingAnnualDividendYield: 0.00516739
# trailingEps: 6.12
# trailingPE: 29.604574
# trailingPegRatio: 2.1717
# twoHundredDayAverage: 180.01515
# underlyingSymbol: AAPL
# uuid: 8b10e4ae-9eeb-3684-921a-9ab27e4d87aa
# volume: 62379661
# website: https://www.apple.com
# zip: 95014