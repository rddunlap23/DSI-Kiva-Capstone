import pandas as pd
from sqlalchemy import create_engine
import requests
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
import nltk
import numpy as np
from nltk.corpus import stopwords

from langid.langid import LanguageIdentifier, model
identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)

class model_data(object):
    
    def __init__(self):
        self.engine = create_engine('postgresql://ryandunlap:tiger@localhost:5432/kivadb')
        self.df = self.load_data()
        self.sample_df = pd.DataFrame()
        
    def load_data(self):
    #Builds up df linking all data together

        my_sql = """
        
        SELECT t.*, m."ISO3", i."income_level", i."latitude", i."longitude", i."region", 
               g."value" as "GDP_Growth", p."value" as "GDP_PCAP_Growth", gni."value" as "GNI_PCAP", tr."value" as "Tourism",
               fx."fx_rate" as "FX_Rate", COALESCE(ls."count",0) as "No_Loans"

        FROM flat_loans t
        LEFT JOIN country_mapping m ON m."ISO2" = t."country_code"
        LEFT JOIN country_info i ON i."iso2Code" = t."country_code"
        LEFT JOIN gdp_growth g on (g."iso2code" = t."country_code" and g."year" = t."year")
        LEFT JOIN gdp_growth_pcap p on (p."iso2code" = t."country_code" and p."year" = t."year")
        LEFT JOIN gni_pc gni on (gni."iso2code" = t."country_code" and gni."year" = t."year")
        LEFT JOIN tourism tr on (tr."iso2code" = t."country_code" and tr."year" = t."year")
        LEFT JOIN fx_rates fx on (fx."curr_code" = t."disbursal_currency" and fx."date" = t."fx_date")
        LEFT JOIN loan_supply ls on (ls."posted_date" = t."posted_date");
        """

        my_df = pd.read_sql(my_sql,self.engine)
        return my_df 
    
    def _transform_dummies(self, df):
        df = pd.get_dummies(df,columns=['year','month','activity','sector','country','income_level','region'],drop_first=True)
        return df

    def _drop_cols(self, df):
        cols = ['funded_amount','repayment_interval','country_code','geo_pairs','funded_date','id','planned_expiration_date','posted_date','status','fx_date','end_date','ISO3','disbursal_currency'] 
        df = df.drop(cols, axis=1)
        return df

    def get_stratified_sample(self, col="target", target_sample_size = 2600, target_ratio = 0.05):
        group_df = self.df.groupby(col)
        new_data = pd.DataFrame()
        
        for group_name, g_df in group_df:
            if group_name == 0:
                sample_size = int(target_sample_size/target_ratio)
                sample = g_df.sample(sample_size)
                new_data = pd.concat([new_data, sample])
            else:
                sample = g_df.sample(target_sample_size)
                new_data = pd.concat([new_data, sample])
        
        df = pd.DataFrame(new_data)
        df = self._transform_dummies(df)
        df = self._drop_cols(df)
        return df