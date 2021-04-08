import pandas as pd
import os
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split




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
    ax2=ax.twinx()

    ax2.set_ylabel(y_label2, color=sec_color,rotation=270)
    ax2.plot(xaxis, data2, color=sec_color,label=legend2)
    ax2.tick_params(axis='y',labelcolor=sec_color)
    ax2.yaxis.labelpad=15

    fig.tight_layout()
    plt.legend(loc=4)
    plt.grid(grid)
    plt.title(title)
    plt.show()
    
df=pd.read_csv(r'C:\Users\home\no_need\Downloads\quant-trading-master\quant-trading-master\Oil Money project\data\vas crude copaud.csv')
df.set_index('date',inplace=True)
df.index=pd.to_datetime(df.index)




D={}

for i in df.columns:
    if i!='cop':
            x=sm.add_constant(df[i])
            y=df['cop']
            m=sm.OLS(y,x).fit()
            D[i]=m.rsquared
            
D=dict(sorted(D.items(),key=lambda x:x[1],reverse=True))


colorlist=[]

for i in D:
    if i =='wti':
        colorlist.append('#447294')
    elif i=='brent':
        colorlist.append('#8fbcdb')
    elif i=='vasconia':
        colorlist.append('#f4d6bc')
    else:
        colorlist.append('#cdc8c8')
        
ax=plt.figure(figsize=(10,5)).add_subplot(111)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

width=0.7

for i in D:
    plt.bar(list(D.keys()).index(i)+width,            
            D[i],width=width,label=i,
            color=colorlist[list(D.keys()).index(i)])
 
plt.title('Regressions on COP')
plt.ylabel('R Squared\n')
plt.xlabel('\nRegressors')
plt.xticks(np.arange(len(D))+width,
           [i.upper() for i in D.keys()],fontsize=8)
plt.show()


ax=plt.figure(figsize=(10,5)).add_subplot(111)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

(df['vasconia']/df['vasconia'].iloc[0]).plot(c='#6f6ff4',
                                             label='Vasconia',alpha=0.5)
(df['brent']/df['brent'].iloc[0]).plot(c='#e264c0',
                                   label='Brent',alpha=0.5)
(df['wti']/df['wti'].iloc[0]).plot(c='#fb6630',
                                   label='WTI',alpha=0.5)
plt.legend(loc=0)
plt.xlabel('Date')
plt.ylabel('Normalized Value by 100')
plt.title('Crude Oil Blends')
plt.show()


dual_axis_plot(df.index,df['cop'],df['gold'],
               x_label='Date',y_label1='Colombian Peso',
               y_label2='Gold LBMA',
               legend1='COP',
               legend2='Gold',
               title='COP VS Gold',
               fst_color='#96CEB4',sec_color='#FFA633')


dual_axis_plot(df.index,df['cop'],df['usd'],
               x_label='Date',y_label1='Colombian Peso',
               y_label2='US Dollar',
               legend1='COP',
               legend2='USD',
               title='COP VS USD',
               fst_color='#9DE0AD',sec_color='#5C4E5F')


dual_axis_plot(df.index,df['cop'],df['brl'],
               x_label='Date',y_label1='Colombian Peso',
               y_label2='Brazilian Real',
               legend1='COP',
               legend2='BRL',
               title='COP VS BRL',
               fst_color='#a4c100',sec_color='#f7db4f')


dual_axis_plot(df.index,df['usd'],df['mxn'],
               x_label='Date',y_label1='US Dollar',
               y_label2='Mexican Peso',
               legend1='USD',
               legend2='MXN',
               title='USD VS MXN',
               fst_color='#F4A688',sec_color='#A2836E')


dual_axis_plot(df.index,df['cop'],df['mxn'],
               x_label='Date',y_label1='Colombian Peso',
               y_label2='Mexican Peso',
               legend1='COP',
               legend2='MXN',
               title='COP VS MXN',
               fst_color='#F26B38',sec_color='#B2AD7F')

dual_axis_plot(df.index,df['cop'],df['vasconia'],
               x_label='Date',y_label1='Colombian Peso',
               y_label2='Vasconia Crude',
               legend1='COP',
               legend2='Vasconia',
               title='COP VS Vasconia',
               fst_color='#346830',sec_color='#BBAB9B')


m=sm.OLS(df['cop'][:'2016'],sm.add_constant(df['vasconia'][:'2016'])).fit()
before=m.rsquared
m=sm.OLS(df['cop']['2017':],sm.add_constant(df['vasconia']['2017':])).fit()
after=m.rsquared

ax=plt.figure(figsize=(10,5)).add_subplot(111)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.bar(['Before 2017',
         'After 2017'],
        [before,after],color=['#82b74b', '#5DD39E'])
plt.ylabel('R Squared')
plt.title('Before/After Regression')
plt.show()


x_train,x_test,y_train,y_test=train_test_split(
        sm.add_constant(df['vasconia'][:'2016']),
        df['cop'][:'2016'],test_size=0.5,shuffle=False)
    
m=sm.OLS(y_test,x_test).fit()
    
forecast=m.predict(x_test)
    
ax=plt.figure(figsize=(10,5)).add_subplot(111)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
forecast.plot(label='Fitted',c='#FEFBD8')
y_test.plot(label='Actual',c='#ffd604')
ax.fill_between(y_test.index,
                    forecast+np.std(m.resid),
                    forecast-np.std(m.resid),
                    color='#F4A688', 
                    alpha=0.6, 
                    label='1 Sigma')
    
ax.fill_between(y_test.index,
                    forecast+2*np.std(m.resid),
                    forecast-2*np.std(m.resid),
                    color='#8c7544', 
                    alpha=0.8, 
                    label='2 Sigma')
    
plt.legend(loc=0)
plt.title(f'Colombian Peso Positions\nR Squared {round(m.rsquared*100,2)}%\n')
plt.xlabel('\nDate')
plt.ylabel('COPAUD')
plt.show()

x_train,x_test,y_train,y_test=train_test_split(
        sm.add_constant(df['vasconia']['2017':]),
        df['cop']['2017':],test_size=0.5,shuffle=False)
    
m=sm.OLS(y_test,x_test).fit()
    
forecast=m.predict(x_test)
    
ax=plt.figure(figsize=(10,5)).add_subplot(111)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
forecast.plot(label='Fitted',c='#FEFBD8')
y_test.plot(label='Actual',c='#ffd604')
ax.fill_between(y_test.index,
                    forecast+np.std(m.resid),
                    forecast-np.std(m.resid),
                    color='#F4A688', 
                    alpha=0.6, 
                    label='1 Sigma')
    
ax.fill_between(y_test.index,
                    forecast+2*np.std(m.resid),
                    forecast-2*np.std(m.resid),
                    color='#8c7544', \
                    alpha=0.8, \
                    label='2 Sigma')
    
plt.legend(loc=0)
plt.title(f'Colombian Peso Positions\nR Squared {round(m.rsquared*100,2)}%\n')
plt.xlabel('\nDate')
plt.ylabel('COPAUD')
plt.show()


dataset=df['2016':]
dataset.reset_index(inplace=True)


import oil_money_trading_backtest as om

signals=om.signal_generation(dataset,'vasconia','cop',om.oil_money,stop=0.001)
p=om.portfolio(signals,'cop')
om.plot(signals,'cop')
om.profit(p,'cop')


dic={}
for holdingt in range(5,20):
    for stopp in np.arange(0.001,0.005,0.0005):
        signals=om.signal_generation(dataset,'vasconia','cop',om.oil_money,
                                     holding_threshold=holdingt,
                                     stop=round(stopp,4))
        
        p=om.portfolio(signals,'cop')
        dic[holdingt,round(stopp,4)]=p['asset'].iloc[-1]/p['asset'].iloc[0]-1
     
profile=pd.DataFrame({'params':list(dic.keys()),'return':list(dic.values())})



ax=plt.figure(figsize=(10,5)).add_subplot(111)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
profile['return'].apply(lambda x:x*100).hist(histtype='bar',
                                             color='#b2660e',
                                             width=0.45,bins=20)
plt.title('Distribution of Return on COP Trading')
plt.grid(False)
plt.ylabel('Frequency')
plt.xlabel('Return (%)')
plt.show()


matrix=pd.DataFrame(columns=[round(i,4) for i in np.arange(0.001,0.005,0.0005)])

matrix['index']=np.arange(5,20)
matrix.set_index('index',inplace=True)

for i,j in profile['params']:
    matrix.at[i,round(j,4)]=     profile['return'][profile['params']==(i,j)].item()*100

for i in matrix.columns:
    matrix[i]=matrix[i].apply(float)



fig=plt.figure(figsize=(10,5))
ax=fig.add_subplot(111)
sns.heatmap(matrix,cmap=plt.cm.viridis,
            xticklabels=3,yticklabels=3)
ax.collections[0].colorbar.set_label('Return(%)\n',
                                      rotation=270)
plt.xlabel('\nStop Loss/Profit (points)')
plt.ylabel('Position Holding Period (days)\n')
plt.title('Profit Heatmap\n',fontsize=10)
plt.style.use('default')



