import psycopg2
import pandas as pd
import pymongo
from pymongo import MongoClient
import time
from sqlalchemy import create_engine
from pandas.tseries.offsets import BMonthBegin
import datetime
engine = create_engine('postgresql://ryandunlap:tiger@localhost:5432/kivadb')

class initial_load(object):
    
    def __init__(self,verbose=True):
        self.sql = """CREATE VIEW loan_supply as
                      SELECT d."posted_date", d."end_date", count(l."id")
                      FROM 
                      (SELECT DISTINCT "posted_date"::timestamp::date, "posted_date"::timestamp::date + INTERVAL '30 days' as "end_date"
                      FROM flat_loans) d
                      JOIN flat_loans l on (l."posted_date" >= d."posted_date" and l."posted_date" <= d."end_date")
                      GROUP BY d."posted_date", d."end_date";"""
        self.verbose = verbose

    def _get_loans_from_mongo(self):
        client = MongoClient()
        db = client.kivadb
        loans = db.loans
        cursor = loans.find()
        df = pd.DataFrame(list(cursor))

        return df
    
    def flatten_loans_for_db(self):
        """This is series of transformations that clean up and extract features from loans."""  
        df = self._get_loans_from_mongo()
        if self.verbose:
            print 'Done loading loans from mongo database'
       
        #Clean up dates and filters out loans with no expiration date (These are pre 2012 loans before Kiva had policy)
        df.loc[:,'planned_expiration_date'] = pd.to_datetime(df.planned_expiration_date,errors='coerce')
        df.loc[:,'posted_date'] = pd.to_datetime(df.posted_date,errors='coerce')
        df.loc[:,'posted_date'] = pd.DatetimeIndex(df.posted_date).normalize()
        df = df.loc[~(df.planned_expiration_date.isnull()),:]
        df.reset_index()
        df.loc[:,'planned_expiration_date'] = pd.to_datetime(df.planned_expiration_date,errors='coerce')
        df.loc[:,'funded_date'] = pd.to_datetime(df.funded_date,errors='coerce')
        
        if self.verbose:
            print 'Date section 1 done (1/7)'

        df['year'] = df.posted_date.dt.strftime('%Y')
        df['month'] = df.posted_date.dt.strftime('%m')
        df['fx_date'] = df.posted_date - pd.offsets.BMonthBegin()
        df.loc[:,'fx_date'] = df.fx_date.dt.strftime('%Y-%m-%d')
        if self.verbose:
            print 'Date section 2 done (2/7)'

        df['end_date'] = df.funded_date
        end_mask = df.end_date.isnull()
        df.loc[end_mask,'end_date'] = df.loc[end_mask,'planned_expiration_date']
        #df['days_to_expiration'] = (df.planned_expiration_date - df.posted_date).astype('timedelta64[D]')
        if self.verbose:
            print 'Date section 3 done (3/7)'

        #Flattens out borrowers creating columns for gender and number of borrowers
        df['gender'] = df.borrowers.map(lambda x: 1 if x[0]['gender'] == 'F' else 0)
        df['num_borrowers'] = df.borrowers.map(lambda x: len(x))
        df['anonymous'] = df.name.map(lambda x: 1 if x in ['Anonymous','Anonymous Group'] else 0)
        if self.verbose:
            print 'Borrow section done (4/7)'

        #Identifies descriptions in English
        df['english'] = df.description.map(lambda x: 1 if 'en' in x['languages'] else 0)
        df['english_desc'] = df.description.map(lambda x: x['texts']['en'] if 'en' in x['languages'] else 'NA')
        if self.verbose:
            print 'Description section done (5/7)'

        #Pulls out location related information. Have to replace south sedan QS with SS to line up with world bank
        df['country'] = df.location.map(lambda x: x['country'])
        df['country_code'] = df.location.map(lambda x: x['country_code'] if x['country_code'] != 'QS' else 'SS')
        df['geo_pairs'] = df.location.map(lambda x: x['geo']['pairs'])
        if self.verbose:
            print 'Locations done (6/7)'

        #Strips out terms of loan
        df['repayment_term'] = df.terms.map(lambda x: x['repayment_term'])
        df['repayment_interval'] = df.terms.map(lambda x: x['repayment_interval'])
        df['disbursal_currency'] = df.terms.map(lambda x: x['disbursal_currency'] if x['disbursal_currency'] != 'SSP' else 'SDG')
        df['currency_risk'] = df.terms.map(lambda x: x['loss_liability']['currency_exchange'])
        if self.verbose:
            print 'Terms of loan done (7/7)'

        #Flags if there is an image
        df['img_included'] = df.image.map(lambda x: 0 if x=='None' else 1)

        df['target'] = df.status.map(lambda x: 1 if x == 'expired' else 0)
        cols_to_drop = ['_id','author','arrears_amount','basket_amount','borrowers','currency_exchange_loss_amount','date',
                    'delinquent','description','journal_totals','location','name','paid_amount','paid_date','payments',
                    'text','terms','video','bonus_credit_eligibility','image','partner_id','tags','themes',
                    'translator','imgid']

        cols_to_use = [c for c in df.columns if c not in cols_to_drop]

        df = df.loc[:,cols_to_use]

        self.df = df
    
    def loans_to_db(self):
        self.df.to_sql('flat_loans',engine,if_exists='replace',index=False, chunksize=2500)
        
        conn = psycopg2.connect("dbname='kivadb' user='ryandunlap' host='localhost' password='printer'")
        c = conn.cursor()
        c.execute(self.sql)
        conn.commit()
        conn.close()   