import requests
import pandas as pd
from pandas.tseries.offsets import BMonthBegin
from sqlalchemy import create_engine
engine = create_engine('postgresql://ryandunlap:tiger@localhost:5432/kivadb')

class fx_rates(object):
    
    def __init__(self, request_type='single'):
        self.request_type = request_type
        if self.request_type == 'initiate':
            self._initial_fx_load()
    
    def _get_open_fx(self, date_list):
        fx_dic = {}
        for date in date_list:
            fx_dic[date] = {}
            url = 'https://openexchangerates.org/api/historical/' + date + '.json?app_id=c218abdcf9594e8b8a0e4b038f09a809'
            fx_dic[date] = requests.get(url).json()['rates']
        return fx_dic

    def _load_fx_to_db(self, fx_dic,table_name='fx_rates',db_method='append'):
        fx_df = pd.DataFrame(fx_dic).reset_index().rename(columns={'index':'curr_code'})
        fx_df = pd.melt(fx_df,id_vars='curr_code',var_name='date',value_name='fx_rate')
        fx_df.sort_values(by=['curr_code','date'],inplace=True)
        fx_df.fx_rate = fx_df.groupby('curr_code')['fx_rate'].ffill()
        fx_df.fx_rate = fx_df.groupby('curr_code')['fx_rate'].bfill()
        fx_df.to_sql(table_name,engine,if_exists=db_method,index=False)

    def _initial_fx_load(self, start_date = '2011-06-01', end_date = '2016-09-01'):
        self.db_method = 'replace'
        date = pd.to_datetime(start_date)
        stop_date = pd.to_datetime(end_date) - BMonthBegin() + BMonthBegin()

        date_list = [date.strftime('%Y-%m-%d')]

        while date != stop_date:
            date = date + BMonthBegin()
            date_list.append(date.strftime('%Y-%m-%d'))
        date_list.append('2012-01-03')
        date_list.append('2013-01-02')
        date_list.append('2014-01-02')
        date_list.append('2015-01-02')
        date_list.append('2016-01-04')
        self._get_open_fx(date_list)
        
        fx_dic = _get_open_fx(date_list)
        _load_fx_to_db(fx_dic,db_method='replace')
        return True      