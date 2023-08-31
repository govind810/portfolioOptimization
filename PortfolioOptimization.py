pip install yfinance
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
msft = yf.Ticker("MSFT")
tsla = yf.Ticker("TSLA")
gs = yf.Ticker("GS")
intc = yf.Ticker("INTC")
xom = yf.Ticker("XOM")
portfolio = pd.DataFrame()
portfolio["msft"] = msft.history(period = '30d')["Close"]
portfolio["tsla"] = tsla.history(period = "30d")["Close"]
portfolio["gs"] = gs.history(period = "30d")["Close"]
portfolio["intc"] = intc.history(period = "30d")["Close"]
portfolio["xom"] = xom.history(period = "30d")["Close"]
print(portfolio)
returns_portfolio = portfolio.pct_change()
returns_portfolio.head()
variance_matrix = returns_portfolio.cov() * 252
variance_matrix
port_returns = []
port_volatility = []
port_weights =[]
num_assets = 5
num_portfolios = 10000
individual_returns = returns_portfolio.mean()
individual_returns
for port in range(num_portfolios):
  weights = np.random.rand(num_assets)
  weights /= np.sum(weights)
  port_weights.append(weights)
  returns = np.dot(weights, individual_returns)
  port_returns.append(returns)
  var = variance_matrix.mul(weights, axis = 0).mul(weights, axis=1).sum().sum()
  sd = np.sqrt(var) * np.sqrt(250)
  port_volatility.append(sd)
data = {"Returns": port_returns, "Volatility": port_volatility}
for counter, symbol in enumerate(portfolio.columns.tolist()):
  data[symbol+" weight"] = [w[counter] for w in port_weights]
portfolio_V1 = pd.DataFrame(data)
portfolio_V1.head()
portfolio_V1.plot.scatter(x = "Volatility", y = "Returns", marker="o", color="y", s=15, alpha=0.5, grid=True, figsize=[8,8])
plt.xlabel("Risk (Volatility)")
plt.ylabel("Expected Returns")
min_vol_port = portfolio_V1.iloc[portfolio_V1["Volatility"].idxmin()]
min_vol_port
rf = 0.01
highest_sharpe_port = portfolio_V1.iloc[((portfolio_V1["Returns"] - rf)/portfolio_V1["Volatility"]).idxmax()]
highest_sharpe_port
plt.subplots(figsize=[8,8])
plt.scatter(portfolio_V1["Volatility"], portfolio_V1["Returns"], marker="o", s=10, alpha=0.3, color="y")
plt.scatter(min_vol_port[1], min_vol_port[0], marker="*", s=500, alpha=1, color="b")
plt.scatter(highest_sharpe_port[1], highest_sharpe_port[0], marker="*", s=500, alpha=1, color="g")
plt.xlabel("Risk (Volatility)")
plt.ylabel("Expected Returns")
