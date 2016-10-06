import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
import numpy as np
import matplotlib
from scipy import stats

#%matplotlib inline
    
class plot_code(object):
    def __init__(self,df):
        #Load dataframe
        self.df = df
    
    def plot_loans(self):
        #Plotting breakdown of funded loans
        def sum_in_millions(series):
            return np.sum(series/1000000)

        top_countries = self.df[self.df.status=='funded'].groupby('country')[['loan_amount']].count().reset_index().sort_values(by='loan_amount',ascending=False).head(25)['country'].unique()

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
        top_countries = self.df.groupby('country')[['loan_amount']].count().reset_index().sort_values(by='loan_amount',ascending=False).head(25)['country'].unique()
        #top_countries = self.df.groupby('country')[['target']].mean().reset_index().sort_values(by='target',ascending=False).head(25)['country'].unique()

        for column in [('sector','Sector'),('region','Region'),('income_level','Income Level'),('country','Country')]:
            if column[0] == 'country':
                row_mask = self.df.country.isin(top_countries)
            else: 
                row_mask = self.df.index

            height_dic = {'country':16,'gender':2,'region':8,'income_level':4,'sector':16}
            w = 16
            h = height_dic[column[0]]

            self.df.loc[row_mask,:].groupby([column[0],'gender'])[['target']].mean().sort_values(by='target').unstack().sort_values([('target',0)], ascending=False).plot(kind='barh',figsize=(w,h),rot=15, color=['steelblue','darkred' ])

            plt.xlabel("% Loans Not Funded", fontsize=14)
            plt.ylabel(column[1], fontsize=14)
            plt.xticks()
            plt.title('Expired Loans by ' + column[1] + '/Gender', fontsize=16)
            plt.legend(['Male','Female'])
            vals = plt.gca().get_xticks()
            plt.gca().set_xticklabels(['{:3.2f}%'.format(x*100) for x in vals])


            path = './assets/' + column[0] + '.png'
            plt.savefig(path, bbox_inches='tight')

        plt.show()

    def plot_avg_loan_by_expired(self):
        top_countries = self.df.groupby('country')[['loan_amount']].count().reset_index().sort_values(by='loan_amount',ascending=False).head(25)['country'].unique()

        #top_countries = self.df.groupby('country')[['loan_amount']].mean().reset_index().sort_values(by='loan_amount',ascending=False).head(29)['country'].unique()
        #Removing these four counties as only had 5 loans between them and for very large amount skewing scale for plotting
        top_countries = [c for c in top_countries if c not in ('Papua New Guinea','Mauritania','Botswana','Afghanistan')]
        for column in [('gender','Gender'),('sector','Sector'),('region','Region'),('income_level','Income Level'),('country','Country')]:
            if column[0] == 'country':
                row_mask = self.df.country.isin(top_countries)
            else: 
                row_mask = self.df.index
            
            height_dic = {'country':16,'gender':2,'region':8,'income_level':4,'sector':16}
            w = 16
            h = height_dic[column[0]]

            self.df.loc[row_mask,:].groupby([column[0],'target'])[['loan_amount']].mean().sort_values(by='loan_amount').unstack().sort_values([('loan_amount',0)], ascending=False).plot(kind='barh',figsize=(w,h),rot=15, color=['black','darkred' ])

            plt.xlabel("Average Loan Amount ($)", fontsize=14)
            plt.ylabel(column[1], fontsize=14)
            plt.xticks( )
            plt.title('Average Loan Amount by ' + column[1] + '/Funded Status', fontsize=16)
            plt.legend(['Funded','Un-Funded'])
            plt.gca().get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
            if column[0] =='gender':
                plt.yticks(range(2), ('Male', 'Female'))


            path = './assets/' + column[0] + 'avg_loaned.png'
            plt.savefig(path, bbox_inches='tight')

        plt.show()  

    def plot_expired(self):
        top_countries = self.df.groupby('country')[['loan_amount']].count().reset_index().sort_values(by='loan_amount',ascending=False).head(25)['country'].unique()

        #top_countries = self.df.groupby('country')[['loan_amount']].mean().reset_index().sort_values(by='loan_amount',ascending=False).head(29)['country'].unique()
        #Removing these four counties as only had 5 loans between them and for very large amount skewing scale for plotting
        top_countries = [c for c in top_countries if c not in ('Papua New Guinea','Mauritania','Botswana','Afghanistan')]
        for column in [('gender','Gender'),('sector','Sector'),('region','Region'),('income_level','Income Level'),('country','Country')]:
            if column[0] == 'country':
                row_mask = self.df.country.isin(top_countries)
            else: 
                row_mask = self.df.index
            
            height_dic = {'country':16,'gender':2,'region':8,'income_level':4,'sector':16}
            w = 16
            h = height_dic[column[0]]

            self.df.loc[row_mask,:].groupby([column[0]])[['target']].mean().reset_index().sort_values(by='target',ascending=False).plot(kind='barh',x=column[0],figsize=(w,h),rot=15, color=['darkred' ])
            #self.df.loc[row_mask,:].groupby([column[0]])[['target']].mean().sort_values(by='target').unstack().sort_values([('loan_amount',0)], ascending=False).plot(kind='barh',figsize=(w,h),rot=15, color=['black','red' ])

            plt.xlabel("% Loans Not Funded", fontsize=14)
            plt.ylabel(column[1], fontsize=14)
            plt.xticks( )
            plt.title('Expired Loans by ' + column[1], fontsize=16)
            plt.legend().set_visible(False)
            vals = plt.gca().get_xticks()
            plt.gca().set_xticklabels(['{:3.2f}%'.format(x*100) for x in vals])
            
            if column[0] =='gender':
                plt.yticks(range(2), ('Male', 'Female'))


            path = './assets/' + column[0] + '_expired_loans.png'
            plt.savefig(path, bbox_inches='tight')

        plt.show()
    
    def plot_point_map(self):
        #Point map showing $$$ of each country loaned to
        map_df = self.df.groupby(by=['latitude','longitude','country'])['loan_amount'].agg({'No. Loans':'count','Dollar Amount':sum}).reset_index()
        map_df = map_df[map_df.latitude != np.nan]
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

        sns.distplot(self.df[(self.df.target==0) & (self.df.loan_amount <10000)].loan_amount, color = 'steelblue', ax=ax)
        sns.distplot(self.df[(self.df.target==1) & (self.df.loan_amount <10000)].loan_amount, color = 'darkred', ax=ax)

        plt.title('Distribution of Loan Amounts', fontsize=16)
        plt.xlabel("Loan Amount ($Dollars)", fontsize=14)
        plt.gca().get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

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

    def plot_avg_loans_by_gender(self):
        top_countries = self.df.groupby('country')[['loan_amount']].count().reset_index().sort_values(by='loan_amount',ascending=False).head(25)['country'].unique()
        for column in [('sector','Sector'),('region','Region'),('income_level','Income Level'),('country','Country')]:
            if column[0] == 'country':
                row_mask = ((self.df.country.isin(top_countries)) & (self.df.status == 'funded'))
                #row_mask = self.index
            else: 
                row_mask = (self.df.status=='funded')
            
            height_dic = {'country':16,'gender':2,'region':8,'income_level':4,'sector':16}
            w = 16
            h = height_dic[column[0]]

            
            self.df.loc[row_mask,:].groupby([column[0],'gender'])[['loan_amount']].mean().sort_values(by='loan_amount').unstack().sort_values([('loan_amount',0)], ascending=False).plot(kind='barh',figsize=(w,h),rot=15, color=['steelblue','darkred' ])

            plt.xlabel("Avg $ Loaned", fontsize=14)
            plt.ylabel(column[1], fontsize=14)
            plt.xticks()
            plt.title('Averages $ Loaned By ' + column[1] + '/Gender', fontsize=16)
            plt.legend(['Male','Female'])
            plt.gca().get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

            path = './assets/' + column[0] + 'avg_by_gender.png'
            plt.savefig(path, bbox_inches='tight')

            plt.show()

    def correlation_heat_map(self):
        df = self.df.select_dtypes(exclude=['object','datetime'])

        corrs = df.corr()

        # Set the default matplotlib figure size:
        fig, ax = plt.subplots(figsize=(16,12))

        mask = np.zeros_like(corrs, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True

        ax = sns.heatmap(corrs, mask=mask)

        ax.set_xticklabels(ax.xaxis.get_ticklabels(), fontsize=14, rotation=70)
        ax.set_yticklabels(ax.yaxis.get_ticklabels(), fontsize=14, rotation=0)

        path = './assets/correlation_matrix.png'
        plt.savefig(path, bbox_inches='tight')

        plt.show()

    def t_test(self, column):
        print_dic = {'No_Loans':('supply of unfunded loans','supply of funded loans'),
                     'loan_amount':('loan amount of unfunded loans','loan amount of funded loans')
                    }
        
        funded = self.df[self.df.target==0][column]
        unfunded = self.df[self.df.target==1][column]

        two_sample_diff_var = stats.ttest_ind(funded, unfunded, equal_var = False)    

        print "The mean %s is %.3f and the mean %s is %.3f. The t-statistic is %.3f and the p-value is %.6f." \
        % (print_dic[column][0], np.mean(unfunded), print_dic[column][1],np.mean(funded), two_sample_diff_var[0],two_sample_diff_var[1])

    def plot_hists(self, column):
        fig = plt.figure(figsize=(16,8))
        ax = fig.gca()

        sns.distplot(self.df[self.df.target==0][column], color = 'steelblue', ax=ax)
        sns.distplot(self.df[self.df.target==1][column], color = 'darkred', ax=ax)

        plt.title('Distribution of Loan Supply', fontsize=16)
        plt.xlabel("    Number of Competing Loans", fontsize=14)
        plt.gca().get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

        path = './assets/' + column + '_dist.png'
        plt.savefig(path, bbox_inches='tight')

        plt.show()

    def plot_time_periods(self):
        self.df.groupby('month')['target'].mean().reset_index().plot(figsize=(16,8),kind='bar',x='month')
        #ax = plt.gca()
        plt.title('% Unfunded Loans By Month')
        
        self.df.groupby('year')['target'].mean().reset_index().plot(figsize=(16,8),kind='bar',x='year')
        #ax = plt.gca()
        plt.title('% Unfunded Loans By Year')

        self.df.groupby(['month','year'])['target'].mean().unstack().plot(figsize=(16,8),kind='bar')
        plt.xticks(range(13), ('Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'))
        plt.xlabel('Month')
        plt.Axes.set
        #ax = plt.gca()
        plt.title('% Unfunded Loans By Year/Month')

        plt.show()  
