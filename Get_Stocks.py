import requests
import pandas as pd
import datetime as dt
import yfinance as yf
import numpy as np
import collections
import random



def down_stocks():
    """Escrapea info de internet"""
    r = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    c = r.content
    componentes = pd.read_html(c)[0]
    start = dt.datetime(2015, 1, 1)
    end= dt.datetime.now() - dt.timedelta(days=3)
    stocks = componentes['Symbol'].tolist()
    data = yf.download(stocks, start, end)['Adj Close']
    return data

def quick_stocks():
    """Escrapea info de internet"""
    n_stocks = int(input("Ingrese numero de acciones: "))
    r = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    c = r.content
    componentes = pd.read_html(c)[0]
    start = dt.datetime(2015, 1, 1)
    end= dt.datetime.now() - dt.timedelta(days=3)
    stocks = componentes['Symbol'].tolist()
    random_stocks = random.sample(stocks, n_stocks)
    data = yf.download(random_stocks, start, end)['Adj Close']
    return data


class StockFilter:
    def __init__(self):
        pass
    
    def filtro_nulo(self, data):
        """Elimina nulos"""
        data.dropna(axis=1, inplace=True)
        return data
    
    def filtro_sharpe(self, data):
        """Elimina acciones con sharpe menor a 0"""
        #calculate returns
        returns_df = np.log1p(data.pct_change())
        # Calculate the Sharpe ratio for each stock
        sharpe_ratios = (returns_df.mean()*252) / (returns_df.std()* np.sqrt(252))
        # Filter stocks with Sharpe ratio less than 0
        sharpe_low = sharpe_ratios[sharpe_ratios < 0].index.tolist()
        data = data.drop(columns=sharpe_low)

        return data, sharpe_ratios
        
    def filtro_betas(self, data, betas_file, beta_max):
        """Uso csv de betas para eliminar de data aquellas acciones sin beta o con beta mayor a x"""
        betas_data = pd.read_csv(betas_file)
        betas_filtered = betas_data.dropna(subset=['Beta'])
        betas_filtered = betas_filtered[betas_filtered['Beta'] <= beta_max]
        stock_list = betas_filtered['Stock'].tolist()      
        data = data[data.columns.intersection(stock_list)]
        return data


    def filtro_tiempo(self, data, n_cotizacion):
        """Filtro por un número de ruedas"""
        columnas = data.columns
        min_values = 252 * n_cotizacion
        n_values = len(data)
        for columna in columnas:
            if n_values - data[columna].count() >= min_values:
                data.drop(columna, axis=1, inplace=True)
        return data

    def filtro_precio(self, data, precio_max, precio_min):
        """Elimina stocks en un rango determinado"""
        stocks_drop = []
        for column in data.columns:
            last_value = data[column].iloc[-1]
            if last_value > precio_max or last_value < precio_min:
                stocks_drop.append(column)
        data.drop(columns=stocks_drop, inplace=True)
        return data
    


class StockFilterParticular:
    def __init__(self):
        pass

    def filtro_sector(self, data, sharpe_ratios):
        """Divido acciones por sector y selecciono las que tienen mejor sharpe"""

        data_dict = {'Sector': {}, 'CurrentRatio': {}, 'P/E': {}, 'DebtToEquity': {}, 'EbitdaMargins': {}, 
                     'EnterpriseToEbitda': {}, 'ROA': {}, 'ROE': {}, 'RevenueGrowth': {}, 'EarningsGrowth': {}, 
                     'Beta': {}, 'GrossMargins':{}}

        stock_names = list(data.columns)

       

        for symbol in stock_names:
            # Get the stock info
            # Create a Ticker object
            ticker = yf.Ticker(symbol)
            stock_info = ticker.info
            
            sector = stock_info.get('sector')  # Replace 'sector' with the actual attribute name
            current_ratio = stock_info.get('currentRatio')
            price_earnings = stock_info.get('trailingPE')
            debt_to_equity = stock_info.get('debtToEquity')
            ebitda_margins = stock_info.get('ebitdaMargins')
            enterprise_to_ebitda = stock_info.get('enterpriseToEbitda')
            return_on_assets = stock_info.get('returnOnAssets')
            return_on_equity = stock_info.get('returnOnEquity')
            revenue_growth = stock_info.get('revenueGrowth')
            earnings_growth = stock_info.get('earningsGrowth')
            beta = stock_info.get('beta')
            gross_Margin = stock_info.get('grossMargins')

##Posibles para eliminar total revenue,
##Posibles incorporar price to book, beta 

            # Check if price_earnings is not None
            if price_earnings is not None:
                # Exception handling for trailingPE
                try:
                    price_earnings = float(price_earnings)
                except ValueError:
                    price_earnings = stock_info.get('forwardPE')


            data_dict['Sector'][symbol] = sector
            data_dict['CurrentRatio'][symbol] = current_ratio
            data_dict['P/E'][symbol] = price_earnings
            data_dict['DebtToEquity'][symbol] = debt_to_equity
            data_dict['EbitdaMargins'][symbol] = ebitda_margins
            data_dict['EnterpriseToEbitda'][symbol] = enterprise_to_ebitda
            data_dict['ROA'][symbol] = return_on_assets
            data_dict['ROE'][symbol] = return_on_equity
            data_dict['RevenueGrowth'][symbol] = revenue_growth
            data_dict['EarningsGrowth'][symbol] = earnings_growth
            data_dict['Beta'][symbol] = beta
            data_dict['GrossMargins'] = gross_Margin
            # data_dict['TotalRevenue'][symbol] = total_revenue

        # Create DataFrame df from the dictionary
        df = pd.DataFrame(data_dict)
            # Append sharpe_ratios as a new column in df_b
        df['Sharpe Ratio'] = sharpe_ratios
        # Transpose df_1 to have stock names as columns
        df = df.transpose()
        # Create a list to store the top stocks in each sector
        top_sharpe = []
        top_currentRatio = []
        top_trailingPE = []
        top_ebitdaMargins = []
        top_debtToEquity = [] # bajo es mejor
        top_enterpriseToEbitda = [] # bajo es mejor
        top_returnOnAssets = []
        top_returnOnEquity = []
        top_revenueGrowth = []
        top_earningsGrowth = []
        top_beta = [] # Se elegirán aquellos por debajo de una std. dev
        top_grossMargin = []
        # top_totalRevenue = []
        
        df_b = df.transpose()

        # Iterate over unique sectors
        for sector in df_b['Sector'].unique():
            # Get the stocks in the current sector
            sector_stocks = df_b[df_b['Sector'] == sector]
           
            # Calculate the mean Sharpe ratio, current ratio, quick ratio, EBITDA margins, debt-to-equity, enterprise-to-EBITDA for the sector
            mean_sharpe_ratio = sector_stocks['Sharpe Ratio'].mean()
            mean_current_ratio = sector_stocks['CurrentRatio'].mean()
            mean_price_earnings = sector_stocks['P/E'].mean()
            std_price_earnings = sector_stocks['P/E'].std()  
            mean_ebitda_margins = sector_stocks['EbitdaMargins'].mean()
            mean_debt_to_equity = sector_stocks['DebtToEquity'].mean()
            mean_enterprise_to_ebitda = sector_stocks['EnterpriseToEbitda'].mean()
            mean_return_on_assets = sector_stocks['ROA'].mean()
            mean_return_on_equity = sector_stocks['ROE'].mean()
            mean_revenue_growth = sector_stocks['RevenueGrowth'].mean()
            mean_earnings_growth = sector_stocks['EarningsGrowth'].mean()
            mean_beta = sector_stocks['Beta'].mean()
            std_beta = sector_stocks['Beta'].std()
            mean_gross_margin = sector_stocks['GrossMargins'].mean()
            # mean_total_revenue = sector_stocks['TotalRevenue'].mean()

            # Select the stocks in the sector where Sharpe ratio, current ratio, quick ratio, EBITDA margins are higher than the mean
            top_sharpe_stocks = sector_stocks[sector_stocks['Sharpe Ratio'] > mean_sharpe_ratio]
            top_currentRatio_stocks = sector_stocks[sector_stocks['CurrentRatio'] > mean_current_ratio]
            top01_trailingPE_stocks = sector_stocks[sector_stocks['P/E'] < mean_price_earnings + std_price_earnings]
            top02_trailingPE_stocks = sector_stocks[sector_stocks['P/E'] > mean_price_earnings - std_price_earnings]
            top_ebitdaMargins_stocks = sector_stocks[sector_stocks['EbitdaMargins'] > mean_ebitda_margins]
            top_returnOnAssets_stocks = sector_stocks[sector_stocks['ROA'] > mean_return_on_assets]
            top_returnOnEquity_stocks = sector_stocks[sector_stocks['ROE'] > mean_return_on_equity]
            top_revenueGrowth_stocks = sector_stocks[sector_stocks['RevenueGrowth'] > mean_revenue_growth]
            top_earningsGrowth_stocks = sector_stocks[sector_stocks['EarningsGrowth'] > mean_earnings_growth]
            top_grossMargin_stocks = sector_stocks[sector_stocks['GrossMargins'] > mean_gross_margin]
            
            # top_totalRevenue_stocks = sector_stocks[sector_stocks['TotalRevenue'] > mean_total_revenue]

            # Select the stocks in the sector where debt-to-equity and enterprise-to-EBITDA, Beta are lower and equal than the mean
            top_debtToEquity_stocks = sector_stocks[sector_stocks['DebtToEquity'] < mean_debt_to_equity]
            top_enterpriseToEbitda_stocks = sector_stocks[sector_stocks['EnterpriseToEbitda'] < mean_enterprise_to_ebitda]
            top_beta_stocks = sector_stocks[sector_stocks['Beta'] < mean_beta + std_beta]

            # Add the top stocks to the list
            top_sharpe.extend(top_sharpe_stocks.index.tolist())
            top_currentRatio.extend(top_currentRatio_stocks.index.tolist())
            top_trailingPE.extend(top01_trailingPE_stocks.index.tolist())
            top_trailingPE.extend(top02_trailingPE_stocks.index.tolist())
            top_ebitdaMargins.extend(top_ebitdaMargins_stocks.index.tolist())
            top_debtToEquity.extend(top_debtToEquity_stocks.index.tolist())
            top_enterpriseToEbitda.extend(top_enterpriseToEbitda_stocks.index.tolist())
            top_returnOnAssets.extend(top_returnOnAssets_stocks.index.tolist())
            top_returnOnEquity.extend(top_returnOnEquity_stocks.index.tolist())
            top_revenueGrowth.extend(top_revenueGrowth_stocks.index.tolist())
            top_earningsGrowth.extend(top_earningsGrowth_stocks.index.tolist())
            top_beta.extend(top_beta_stocks.index.tolist())
            top_grossMargin.extend(top_grossMargin_stocks.index.tolist())
            # top_totalRevenue.extend(top_totalRevenue_stocks.index.tolist())

        list_of_lists = [top_sharpe, top_currentRatio, top_trailingPE, top_ebitdaMargins, 
                         top_debtToEquity, top_enterpriseToEbitda, top_returnOnAssets, top_returnOnEquity, top_revenueGrowth, top_earningsGrowth, top_beta,top_grossMargin]
        # Define the minimum required occurrence count ("y")
        min_occurrence = 6

        # Flatten the list of lists into a single list
        flattened_list = [stock for sublist in list_of_lists for stock in sublist]

        # Count the occurrences of each stock
        stock_counts = collections.Counter(flattened_list)

        # Create the new list with stocks that occur in "y" or more number of lists
        new_list = [stock for stock, count in stock_counts.items() if count >= min_occurrence]

        columns_to_drop = [col for col in data.columns if col not in new_list]

        data = data.drop(columns=columns_to_drop)
        
        return data, stock_counts


    def top_sector(self, data, stock_counts):
        """Tomo lista de acciones y numero de filtros que pasaron y le append a que sector corresponde cada stock
        despues elijo por cada sector las acciones que estan en el quantil 0.8 de la muestra basado en numero de filtros pasados
        y drop de dataframe principal las acciones que no pasen esa resticcion"""
        #Tomo acciones y filtros pasados como dataframe
        df_count = pd.DataFrame.from_dict(stock_counts, orient='index', columns=['Number']).reset_index()
        df_count.columns = ['Stock', 'Number']
        #creo columna sector y descargo dato para agregar sector correspondiente
        df_count['Sector'] = ''
        for index, row in df_count.iterrows():
            symbol = row['Stock']
            ticker = yf.Ticker(symbol)
            stock_info = ticker.info
            sector = stock_info.get('sector')
            # Update the 'Sector' column for the current row
            df_count.loc[index, 'Sector'] = sector
        #Creo lista con acciones que esten quantil 0.8 de cada sector en haber pasado filtros
        new_list = []
        sector_quantile = df_count.groupby('Sector')['Number'].quantile(0.8)
        sector_quantile_df = sector_quantile.to_frame()
        sector_quantile_df.reset_index(inplace=True)
        sector_quantile_df.columns = ['Sector', 'Quantile_Number']
        merged_df = df_count.merge(sector_quantile_df, on='Sector')
        for index, row in merged_df.iterrows():
            if row['Number'] > row['Quantile_Number']:
                new_list.append(row['Stock'])
        #Drop de dataframe de acciones principal todas las acciones que no hayan calificado
        columns_to_drop = [col for col in data.columns if col not in new_list]
        data = data.drop(columns=columns_to_drop)
        #Creo otro df con acciones, sector, num filtros pasados y quantil.
        data_sectors =  merged_df[merged_df['Stock'].isin(new_list)]
        return data, data_sectors


