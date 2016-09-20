import requests
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
engine = create_engine('postgresql://ryandunlap:tiger@localhost:5432/kivadb')

class world_bank(object):
    """Class calls world bank and okfn API to get country level data and loads to local postgresql
       database. This data will be used later for building up the model
       
       Example Call: 
       
       wb = world_bank()
       wb.source_world_bank()
       wb.country_mapping_to_db()
       
       """
    
    def __init__(self, country_list = [], dbase_method='replace',start_year='2005',end_year='2016'):
        self.indicators = ['NY.GDP.PCAP.KD.ZG','NY.GDP.MKTP.KD.ZG','ST.INT.ARVL','NY.GNP.PCAP.PP.CD']
        self.tables = ['gdp_growth_pcap','gdp_growth','tourism','gni_pc']
        self.dbase_method = dbase_method
        self.start_year = start_year
        self.end_year = end_year
        if len(country_list)==0:
            self.country_list = self._country_info_to_db()
        else:
            self.country_list = country_list
        
    def country_mapping_to_db(self):
    
        """Function to get currency to country mapping info"""

        country_info = requests.get('http://data.okfn.org/data/core/country-codes/r/country-codes.json').json()

        country_dic = {}

        for _ in country_info:
            country_dic[_['ISO3166-1-Alpha-3']] = {'ISO2':_['ISO3166-1-Alpha-2'],'Currency_Code':_['ISO4217-currency_alphabetic_code'],
                                  'Country_Long_Name':_['ISO4217-currency_country_name'],'Country_Short_Name':_['official_name_en']}

        df_country_mapping = pd.DataFrame(country_dic)
        df_country_mapping = df_country_mapping.T.reset_index()
        df_country_mapping.rename(columns={'index':'ISO3'},inplace=True)
        df_country_mapping.Currency_Code = df_country_mapping.Currency_Code.map(lambda x: x if x!= 'SSP' else 'SDG')
        df_country_mapping.to_sql('country_mapping',engine,if_exists='replace',index=False)
    
    def _country_info_to_db(self):
    
        """Function to get country information from world bank api"""

        country_info = requests.get('http://api.worldbank.org/countries?format=json&per_page=500').json()

        country_dic = {}

        for _ in country_info[1]:
            country_dic[_['id']] = {'longitude':_['longitude'] , 'latitude':_['latitude'], 
                                          'income_level':_['incomeLevel']['value'],'iso2Code':_['iso2Code'],
                                          'region':_['region']['value']}

        df_c = pd.DataFrame(country_dic).T
        df_c = df_c.reset_index()
        df_c.rename(columns={'index':'iso3code'},inplace=True)
        df_c = df_c.loc[~(df_c.region=='Aggregates'),:]
        df_c.to_sql('country_info',engine,if_exists='replace',index=False)
        
        return df_c.iso2Code.unique()
       
    def _get_world_bank_data(self,country_list,indicator):
        
        """Function returns dictionary with data from world bank indicator. 
           Function takes list of iso2 codes and indicator code
           Example URL with indicator description http://api.worldbank.org/indicators/NY.GDP.MKTP.CD
        """
        
        data_dic = {}

        for code in country_list:
            data_dic[code] = {}
            source = 'http://api.worldbank.org/countries/' + code + '/indicators/' + indicator + '?date=' + self.start_year + ':' + self.end_year + '&per_page=5000&format=json'

            try:
                data = requests.get(source).json()[1:][0]
                for _ in data:
                    data_dic[code][_['date']] = _['value']
            except:
                print 'Can not find data for country iso code %s' %(code)

        return data_dic


    def _world_bank_to_db(self,country_list, indicator, table_name):
        """Function writes world bank data to database. Takes list of countries, api indicator names and db table name"""
        
        years_list = [str(y) for y in range(int(self.start_year),int(self.end_year)+1)]
        needed_data = pd.DataFrame([[c,y] for c in country_list for y in years_list])
        needed_data.rename(columns={0:'iso2code',1:'year'},inplace=True)
        
        wb_data = self._get_world_bank_data(country_list,indicator)
        wb_df = pd.DataFrame(wb_data)
        wb_df = wb_df.T.reset_index().rename(columns={'index':'iso2code'})
        wb_df = pd.melt(wb_df,id_vars='iso2code',var_name='year')
        
        merged_data = needed_data.merge(wb_df,how='left',on=['iso2code','year'])
        merged_data.sort_values(by=['iso2code','year'],inplace=True)
        merged_data.value = merged_data.groupby('iso2code')['value'].ffill()
        
        merged_data.to_sql(table_name,engine,if_exists=self.dbase_method,index=False)
        
    def source_world_bank(self):
        for indicator,table in zip(self.indicators,self.tables):
            self._world_bank_to_db(self.country_list,indicator,table)       