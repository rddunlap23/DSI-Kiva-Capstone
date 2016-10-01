import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
import numpy as np
import matplotlib

#%matplotlib inline
    
class plot_code(object):
    def __init__(self,df):
        self.df = df
    
    def plot_loans(self):
        #Plotting breakdown of funded loans
        def sum_in_millions(series):
            return np.sum(series/1000000)

        top_countries = self.df[self.df.status=='funded'].groupby('country')[['loan_amount']].sum().reset_index().sort_values(by='loan_amount',ascending=False).head(25)['country'].unique()

        for column in [('sector','Sector'),('region','Region'),('income_level','Income Level'),('country','Country'),('gender','Gender')]:
            if column[0] == 'country':
                row_mask = ((self.df.country.isin(top_countries)) & (self.df.status == 'funded'))
            else: 
                row_mask = (self.df.status=='funded')

            # Plot funded loans by sector

            height_dic = {'country':16,'gender':2,'region':8,'income_level':4,'sector':16}
            w = 16
            h = height_dic[column[0]]

            self.df.loc[row_mask,:].groupby(column[0])[['loan_amount']].agg(['count',sum_in_millions])\
                                 .sort_values([('loan_amount','sum_in_millions')], ascending=False).reset_index()\
                                 .plot(kind='barh',figsize=(w,h),rot=5, x=[(column[0],'')],y=[('loan_amount','sum_in_millions')])

            plt.xlabel("$ Loaned in Millions", fontsize=14)
            plt.ylabel(column[1], fontsize=14)
            plt.legend().set_visible(False)
            plt.title('Amount Loaned By ' + column[1], fontsize=16)
            if column[0] =='gender':
                plt.yticks(range(2), ('Female', 'Male'))


            path = './assets/'+column[0]+'_loaned.png'
            plt.savefig(path, bbox_inches='tight')

        plt.show()
            
    def plot_loans_by_gender(self):
        #This section plots bar graphs breaking down expired loans
        top_countries = self.df.groupby('country')[['target']].mean().reset_index().sort_values(by='target',ascending=False).head(25)['country'].unique()

        for column in [('sector','Sector'),('region','Region'),('income_level','Income Level'),('country','Country')]:
            if column[0] == 'country':
                row_mask = self.df.country.isin(top_countries)
            else: 
                row_mask = self.df.index
            self.df.loc[row_mask,:].groupby([column[0],'gender'])[['target']].mean().sort_values(by='target').unstack().sort_values([('target',0)], ascending=False).plot(kind='barh',figsize=(16,10),rot=15, color=['steelblue','darkred' ])

            plt.xlabel("% Loans Not Funded", fontsize=14)
            plt.ylabel(column[1], fontsize=14)
            plt.xticks()
            plt.title('Expired Loans by ' + column[1] + '/Gender', fontsize=16)
            plt.legend(['Male','Female'])

            path = './assets/' + column[0] + '.png'
            plt.savefig(path, bbox_inches='tight')

        plt.show()

    def plot_avg_loan_by_expired(self):
    #This section plots bar graphs breaking down expired loans

        top_countries = self.df.groupby('country')[['loan_amount']].mean().reset_index().sort_values(by='loan_amount',ascending=False).head(27)['country'].unique()
        #Removing these four counties as only had 5 loans between them and for very large amount skewing scale for plotting
        top_countries = [c for c in top_countries if c not in ('Papua New Guinea','Mauritania','Botswana','Afghanistan')]
        for column in [('sector','Sector'),('region','Region'),('income_level','Income Level'),('country','Country')]:
            if column[0] == 'country':
                row_mask = self.df.country.isin(top_countries)
            else: 
                row_mask = self.df.index
            self.df.loc[row_mask,:].groupby([column[0],'target'])[['loan_amount']].mean().sort_values(by='loan_amount').unstack().sort_values([('loan_amount',0)], ascending=False).plot(kind='barh',figsize=(16,10),rot=15, color=['black','red' ])

            plt.xlabel("Average Loan Amount ($)", fontsize=14)
            plt.ylabel(column[1], fontsize=14)
            plt.xticks( )
            plt.title('Average Loan Amount by ' + column[1] + '/Funded Status', fontsize=16)
            plt.legend(['Funded','Un-Funded'])
            plt.gca().get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
            path = './assets/' + column[0] + 'avg_loaned.png'
            plt.savefig(path, bbox_inches='tight')

        plt.show()  
    
    def plot_point_map(self):
        #Point map showing $$$ of each country loaned to
        map_df = self.df.groupby(by=['latitude','longitude','country'])['loan_amount'].agg({'No. Loans':'count','Dollar Amount':sum}).reset_index()
        map_df = map_df[map_df.latitude != ""]
        map_df['Dollar Amount'] = map_df['Dollar Amount'].map('${:,.0f}'.format)
        map_df['No. Loans'] = map_df['No. Loans'].map('{:,.2f}'.format)

        point_map = folium.Map(location=[-1.2797499999999999,36.812600000000003], zoom_start=2)
        points = zip(pd.to_numeric(map_df.latitude),pd.to_numeric(map_df.longitude),map_df.country,map_df['No. Loans'],map_df['Dollar Amount'])
        for point in points:
            popup_str = point[2] + ' ' + point[4]
            folium.Marker([point[0], point[1]], popup = popup_str, icon=folium.Icon(color='blue',icon='info-sign')).add_to(point_map)

        point_map.save('point_map.html')
        return point_map
        
    def plot_loan_hist(self):
        fig = plt.figure(figsize=(16,10))
        ax = fig.gca()

        sns.distplot(self.df[(self.df.target==0) & (self.df.loan_amount <10000)].loan_amount, color = 'blue', ax=ax)
        sns.distplot(self.df[(self.df.target==1) & (self.df.loan_amount <10000)].loan_amount, color = 'red', ax=ax)

        path = './assets/loan_amount_dist.png'
        plt.savefig(path, bbox_inches='tight')

        plt.show()

    def gender_loans_count(self):
        fig = plt.figure(figsize=(16,10))
        ax = fig.gca()

        self.df[self.df.status=='funded'].groupby('gender')[['loan_amount']].count().reset_index().sort_values(by='loan_amount',ascending=False).plot(kind='barh',figsize=(16,2),rot=5, x='gender',y='loan_amount',ax=ax)
        plt.yticks(range(2), ('Female', 'Male'))
        plt.legend().set_visible(False)

        plt.xlabel("Number of Loans Made", fontsize=14)
        plt.ylabel("Gender", fontsize=14)

        ax.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

        path = './assets/gender_loan_count.png'
        plt.savefig(path, bbox_inches='tight')

        plt.show()