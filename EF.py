import numpy as np
import pandas as pd
import scipy.optimize as sco


class PortfolioOptimization:

    def __init__(self, table):
        self.table = table

    def portfolio_annualised_performance(self, weights, mean_returns, cov_matrix):
        returns = np.sum(mean_returns * weights) * 252
        std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
        return std, returns

    def random_portfolios(self, num_portfolios, mean_returns, cov_matrix, risk_free_rate):
        noa = len(mean_returns)
        results = np.zeros((3, num_portfolios))
        weights_record = []
        for i in range(num_portfolios):
            weights = np.random.random(noa)
            weights /= np.sum(weights)
            weights_record.append(weights)
            portfolio_std_dev, portfolio_return = self.portfolio_annualised_performance(weights, mean_returns,
                                                                                     cov_matrix)
            results[0, i] = portfolio_std_dev
            results[1, i] = portfolio_return
            results[2, i] = (portfolio_return - risk_free_rate) / portfolio_std_dev
        return results, weights_record

    def neg_sharpe_ratio(self, weights, mean_returns, cov_matrix, risk_free_rate):
        p_var, p_ret = self.portfolio_annualised_performance(weights, mean_returns, cov_matrix)
        return -(p_ret - risk_free_rate) / p_var

    def max_sharpe_ratio(self, mean_returns, cov_matrix, risk_free_rate):
        num_assets = len(mean_returns)
        args = (mean_returns, cov_matrix, risk_free_rate)
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        bounds = tuple((0.0, 1.0) for asset in range(num_assets))
        result = sco.minimize(self.neg_sharpe_ratio, num_assets * [1.0 / num_assets], args=args,
                              method='SLSQP', bounds=bounds, constraints=constraints)
        return result

    def portfolio_volatility(self, weights, mean_returns, cov_matrix):
        return self.portfolio_annualised_performance(weights, mean_returns, cov_matrix)[0]

    def min_variance(self, mean_returns, cov_matrix):
        num_assets = len(mean_returns)
        args = (mean_returns, cov_matrix)
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        bounds = tuple((0.0, 1.0) for asset in range(num_assets))

        result = sco.minimize(self.portfolio_volatility, num_assets * [1.0 / num_assets], args=args,
                              method='SLSQP', bounds=bounds, constraints=constraints)

        return result

    def efficient_return(self, mean_returns, cov_matrix, target):
        num_assets = len(mean_returns)
        args = (mean_returns, cov_matrix)

        def portfolio_return(weights):
            return self.portfolio_annualised_performance(weights, mean_returns, cov_matrix)[1]

        constraints = [{'type': 'eq', 'fun': lambda x: portfolio_return(x) - target},
                       {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        bounds = tuple((0, 1) for asset in range(num_assets))
        result = sco.minimize(self.portfolio_volatility, num_assets * [1.0 / num_assets], args=args,
                              method='SLSQP', bounds=bounds, constraints=constraints)
        return result

    def efficient_frontier(self, mean_returns, cov_matrix, returns_range):
        efficients = []
        for ret in returns_range:
            efficients.append(self.efficient_return(mean_returns, cov_matrix, ret))
        return efficients

    def display_calculated_ef_with_random(self, mean_returns, cov_matrix, num_portfolios, risk_free_rate):
        results, _ = self.random_portfolios(num_portfolios, mean_returns, cov_matrix, risk_free_rate)

        max_sharpe = self.max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate)
        sdp, rp = self.portfolio_annualised_performance(max_sharpe['x'], mean_returns, cov_matrix)
        max_sharpe_allocation = pd.DataFrame(max_sharpe.x, index=self.table.columns, columns=['allocation'])
        max_sharpe_allocation.allocation = [round(i * 100, 2) for i in max_sharpe_allocation.allocation]
        max_sharpe_allocation = max_sharpe_allocation.T

        min_vol = self.min_variance(mean_returns, cov_matrix)
        sdp_min, rp_min = self.portfolio_annualised_performance(min_vol['x'], mean_returns, cov_matrix)
        min_vol_allocation = pd.DataFrame(min_vol.x, index=self.table.columns, columns=['allocation'])
        min_vol_allocation.allocation = [round(i * 100, 2) for i in min_vol_allocation.allocation]
        min_vol_allocation = min_vol_allocation.T

        max_sharpe_allocation = max_sharpe_allocation.loc[:, (max_sharpe_allocation != 0).any()]
        min_vol_allocation = min_vol_allocation.loc[:, (min_vol_allocation != 0).any()]

        print("-" * 80)
        print("Maximum Sharpe Ratio Portfolio Allocation\n")
        print("Annualised Return:", round(rp, 2))
        print("Annualised Volatility:", round(sdp, 2))
        print("\n")
        print(max_sharpe_allocation)
        print("-" * 80)
        print("Minimum Volatility Portfolio Allocation\n")
        print("Annualised Return:", round(rp_min, 2))
        print("Annualised Volatility:", round(sdp_min, 2))
        print("\n")
        print(min_vol_allocation)


