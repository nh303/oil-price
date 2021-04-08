
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.linear_model import ElasticNetCV as en 
from statsmodels.tsa.stattools import adfuller as adf
import os






df=pd.read_csv(r'C:\Users\...\data\brent crude nokjpy.csv')
df.set_index(pd.to_datetime(df[list(df.columns)[0]]),inplace=True)
del df[list(df.columns)[0]]



ax=plt.figure(figsize=(10,5)).add_subplot(111)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.scatter(df['brent'][df.index<'2017-04-25'],df['nok'][df.index<'2017-04-25'],s=1,c='#5f0f4e')

plt.title('NOK Brent Correlation')
plt.xlabel('Brent in JPY')
plt.ylabel('NOKJPY')
plt.show()


def dual_axis_plot(xaxis,data1,data2,fst_color='r',
                    sec_color='b',fig_size=(10,5),
                   x_label='',y_label1='',y_label2='',
                   legend1='',legend2='',grid=False,title=''):
    
    fig=plt.figure(figsize=fig_size)
    ax=fig.add_subplot(111)
    

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label1, color=fst_color)
    ax.plot(xaxis, data1, color=fst_color,label=legend1)
    ax.tick_params(axis='y',labelcolor=fst_color)
    ax.yaxis.labelpad=15

    plt.legend(loc=3)
    ax2 = ax.twinx()

    ax2.set_ylabel(y_label2, color=sec_color,rotation=270)
    ax2.plot(xaxis, data2, color=sec_color,label=legend2)
    ax2.tick_params(axis='y',labelcolor=sec_color)
    ax2.yaxis.labelpad=15

    fig.tight_layout()
    plt.legend(loc=4)
    plt.grid(grid)
    plt.title(title)
    plt.show()
    

dual_axis_plot(df.index,df['nok'],df['interest rate'],
               fst_color='#34262b',sec_color='#cb2800',
               fig_size=(10,5),x_label='Date',
               y_label1='NOKJPY',y_label2='Norges Bank Interest Rate %',
               legend1='NOKJPY',legend2='Interest Rate',
               grid=False,title='NOK vs Interest Rate')


dual_axis_plot(df.index,df['nok'],df['brent'],
               fst_color='#4f2d20',sec_color='#3feee6',
               fig_size=(10,5),x_label='Date',
               y_label1='NOKJPY',y_label2='Brent in JPY',
               legend1='NOKJPY',legend2='Brent',
               grid=False,title='NOK vs Brent')
               

ind=df['gdp yoy'].dropna().index
dual_axis_plot(df.loc[ind].index,
               df['nok'].loc[ind],
               df['gdp yoy'].dropna(),
               fst_color='#116466',sec_color='#ff652f',
               fig_size=(10,5),x_label='Date',
               y_label1='NOKJPY',y_label2='Norway GDP YoY %',
               legend1='NOKJPY',legend2='GDP',
               grid=False,title='NOK vs GDP')



x0=pd.concat([df['usd'],df['gbp'],df['eur'],df['brent']],axis=1)
x1=sm.add_constant(x0)
x=x1[x1.index<'2017-04-25']
y=df['nok'][df.index<'2017-04-25']

model=sm.OLS(y,x).fit()
print(model.summary(),'\n')



m=en(alphas=[0.0001, 0.0005, 0.001, 0.01, 0.1, 1, 10],
     l1_ratio=[.01, .1, .5, .9, .99],  max_iter=5000).fit(x0[x0.index<'2017-04-25'], y)  
print(m.intercept_,m.coef_)



df['sk_fit']=(df['usd']*m.coef_[0]+df['gbp']*m.coef_[1]+
                 df['eur']*m.coef_[2]+df['brent']*m.coef_[3]+m.intercept_)


df['sk_residual']=df['nok']-df['sk_fit']



upper=np.std(df['sk_residual'][df.index<'2017-04-25'])
lower=-upper

signals=pd.concat([df[i] for i in ['nok', 'usd', 'eur', 'gbp', 'brent', 'sk_fit','sk_residual']], \
                  axis=1)[df.index>='2017-04-25']
signals['fitted']=signals['sk_fit']
del signals['sk_fit']

signals['upper']=signals['fitted']+upper
signals['lower']=signals['fitted']+lower
signals['stop profit']=signals['fitted']+2*upper
signals['stop loss']=signals['fitted']+2*lower
signals['signals']=0




index=list(signals.columns).index('signals')

for j in range(len(signals)):
    
    if signals['nok'].iloc[j]>signals['upper'].iloc[j]:
        signals.iloc[j,index]=-1  
          
    if signals['nok'].iloc[j]<signals['lower'].iloc[j]:
        signals.iloc[j,index]=1 
       
    signals['cumsum']=signals['signals'].cumsum()

    if signals['cumsum'].iloc[j]>1 or signals['cumsum'].iloc[j]<-1:
        signals.iloc[j,index]=0
  
    if signals['nok'].iloc[j]>signals['stop profit'].iloc[j]:         
        signals['cumsum']=signals['signals'].cumsum()
        signals.iloc[j,index]=-signals['cumsum'].iloc[j]+1
        signals['cumsum']=signals['signals'].cumsum()
        break

    if signals['nok'].iloc[j]<signals['stop loss'].iloc[j]:
        signals['cumsum']=signals['signals'].cumsum()
        signals.iloc[j,index]=-signals['cumsum'].iloc[j]-1
        signals['cumsum']=signals['signals'].cumsum()
        break


ax=plt.figure(figsize=(10,5)).add_subplot(111)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

signals['nok'].plot(label='NOKJPY',c='#594f4f',alpha=0.5)
ax.plot(signals.loc[signals['signals']>0].index,
         signals['nok'][signals['signals']>0],
         lw=0,marker='^',c='#83af9b',label='LONG', markersize=10)
ax.plot(signals.loc[signals['signals']<0].index,
         signals['nok'][signals['signals']<0],
         lw=0,marker='v',c='#fe4365',label='SHORT', markersize=10)
ax.plot(pd.to_datetime('2017-12-20'),
         signals['nok'].loc['2017-12-20'],
         lw=0,marker='*',c='#f9d423', markersize=15, alpha=0.8,
         label='Potential Exit Point of Momentum Trading')

plt.axvline('2017/11/15',linestyle=':',c='k',label='Exit')
plt.legend()
plt.title('NOKJPY Positions')
plt.ylabel('NOKJPY')
plt.xlabel('Date')
plt.show()



ax=plt.figure(figsize=(10,5)).add_subplot(111)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

signals['fitted'].plot(lw=2.5,label='Fitted',c='w',alpha=0.6)
signals['nok'].plot(lw=2,label='Actual',c='#04060f',alpha=0.8)
ax.fill_between(signals.index,signals['upper'],
                signals['lower'],alpha=0.2,label='1 Sigma',color='#2a3457')
ax.fill_between(signals.index,signals['stop profit'],
                signals['stop loss'],alpha=0.1,label='2 Sigma',color='#720017')

plt.legend(loc='best')
plt.title('Fitted vs Actual')
plt.ylabel('NOKJPY')
plt.xlabel('Date')
plt.show()



ax=plt.figure(figsize=(10,5)).add_subplot(111)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

(df['nok']/df['nok'][0]*100).plot(c='#ff8c94',label='Norwegian Krone',alpha=0.9)
(df['usd']/df['usd'][0]*100).plot(c='#9de0ad',label='US Dollar',alpha=0.9)
(df['eur']/df['eur'][0]*100).plot(c='#45ada8',label='Euro',alpha=0.9)
(df['gbp']/df['gbp'][0]*100).plot(c='#f8b195',label='UK Sterling',alpha=0.9)
(df['brent']/df['brent'][0]*100).plot(c='#6c5b7c',label='Brent Crude',alpha=0.5)

plt.legend(loc='best')
plt.ylabel('Normalized Price by 100')
plt.xlabel('Date')
plt.title('Trend')
plt.show()


x2=df['eur'][df.index<'2017-04-25']
x3=sm.add_constant(x2)

model=sm.OLS(y,x3).fit()
ero=model.resid

print(adf(ero))
print(model.summary())


capital0=2000
positions=100
portfolio=pd.DataFrame(index=signals.index)
portfolio['holding']=signals['nok']*signals['cumsum']*positions
portfolio['cash']=capital0-(signals['nok']*signals['signals']*positions).cumsum()
portfolio['total asset']=portfolio['holding']+portfolio['cash']
portfolio['signals']=signals['signals']




portfolio=portfolio[portfolio.index>'2017-10-01']
portfolio=portfolio[portfolio.index<'2018-01-01']



ax=plt.figure(figsize=(10,5)).add_subplot(111)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

portfolio['total asset'].plot(c='#594f4f',alpha=0.5,label='Total Asset')
ax.plot(portfolio.loc[portfolio['signals']>0].index,portfolio['total asset'][portfolio['signals']>0],
         lw=0,marker='^',c='#2a3457',label='LONG',markersize=10,alpha=0.5)
ax.plot(portfolio.loc[portfolio['signals']<0].index,portfolio['total asset'][portfolio['signals']<0],
         lw=0,marker='v',c='#720017',label='The Big Short',markersize=15,alpha=0.5)
ax.fill_between(portfolio['2017-11-20':'2017-12-20'].index,
                 (portfolio['total asset']+np.std(portfolio['total asset']))['2017-11-20':'2017-12-20'],
                 (portfolio['total asset']-np.std(portfolio['total asset']))['2017-11-20':'2017-12-20'],
                 alpha=0.2, color='#547980')

plt.text(pd.to_datetime('2017-12-20'),
          (portfolio['total asset']+np.std(portfolio['total asset'])).loc['2017-12-20'],
          'What if we use MACD here?')
plt.axvline('2017/11/15',linestyle=':',label='Exit',c='#ff847c')
plt.legend()
plt.title('Portfolio Performance')
plt.ylabel('Asset Value')
plt.xlabel('Date')
plt.show()



import oil_money_trading_backtest as om


signals=om.signal_generation(dataset,'brent','nok',om.oil_money)
p=om.portfolio(signals,'nok')
om.plot(signals,'nok')
om.profit(p,'nok')


dic={}
for holdingt in range(5,20):
    for stopp in np.arange(0.3,1.1,0.05):
        signals=om.signal_generation(dataset,'brent','nok',om.oil_money, \
                                     holding_threshold=holdingt, \
                                     stop=stopp)
        
        p=om.portfolio(signals,'nok')
        dic[holdingt,stopp]=p['asset'].iloc[-1]/p['asset'].iloc[0]-1
     
profile=pd.DataFrame({'params':list(dic.keys()),'return':list(dic.values())})




ax=plt.figure(figsize=(10,5)).add_subplot(111)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
profile['return'].apply(lambda x:x*100).hist(histtype='bar', \
                                            color='#f09e8c', \
                                            width=0.45,bins=20)
plt.title('Distribution of Return on NOK Trading')
plt.grid(False)
plt.ylabel('Frequency')
plt.xlabel('Return (%)')
plt.show()


#
matrix=pd.DataFrame(columns= \
                    [round(i,2) for i in np.arange(0.3,1.1,0.05)])

matrix['index']=np.arange(5,20)
matrix.set_index('index',inplace=True)

for i,j in profile['params']:
    matrix.at[i,round(j,2)]= \
    profile['return'][profile['params']==(i,j)].item()*100

for i in matrix.columns:
    matrix[i]=matrix[i].apply(float)



fig=plt.figure(figsize=(10,5))
ax=fig.add_subplot(111)
sns.heatmap(matrix,cmap='gist_heat_r',square=True, \
            xticklabels=3,yticklabels=3)
ax.collections[0].colorbar.set_label('Return(%) \n', \
                                     rotation=270)
plt.xlabel('\nStop Loss/Profit (points)')
plt.ylabel('Position Holding Period (days)\n')
plt.title('Profit Heatmap\n',fontsize=10)
plt.style.use('default')


