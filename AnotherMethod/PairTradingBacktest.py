import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import fix_yahoo_finance as yf
from sklearn.model_selection import train_test_split

def cointegration(data1, data2):

    df1, test1, df2, test2 = train_test_split(data1, data2, test_size=0.7, shuffle=False)

    train = pd.DataFrame()
    train['asset1'] = df1['Close']
    train['asset2'] = df2['Close']

    x = sm.add_constant(train['asset1'])
    y = train['asset2']
    model = sm.OLS(y, x).fit()
    resid = model.resid
    signals = pd.DataFrame()
    signals['asset1'] = test1['Close']  
    signals['asset2'] = test2['Close']

    signals['fitted'] = np.mat(sm.add_constant(signals['asset2']))* np.mat(model.params).reshape(2, 1)

    signals['residual'] = signals['asset1'] - signals['fitted']
    signals['z'] = (signals['residual'] - np.mean(signals['residual']))/np.std(signals['residual'])
    signals['z upper limit'] = signals['z']*0 + np.mean(signals['z']) + np.std(signals['z'])
    signals['z lower limit'] = signals['z']*0 + np.mean(signals['z']) - np.std(signals['z'])

    return signals




def signal_gen(df1, df2, method):

    signals = method(df1, df2)
    signals['signals1'] = 0
    signals['signals1'] = np.select([signals['z']>signals['z upper limit'], \
                                    signals['z']<signals['z lower limit']], \
                                    [1, -1],default=0)
    signals['positions1'] = signals['signals1'].diff()
    signals['signals2'] = -signals['signals1']
    signals['positions2'] = signals['signals2'].diff()
    return signals

def plot(new, ticket1, ticket2):

    '''
    signal_data = open('signal_data.txt','w')
    new_str = ''+str(new)
    signal_data.write(new_str)
    signal_data.close()
    #print(new)
    '''

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(211)

    new['z'].plot(label='z statistic', c='#e8175d')
    ax.fill_between(new.index,new['z upper limit'], \
                    new['z lower limit'],label='+- 1 sigma', \
                    alpha = 0.5, color='#f7bd4f')
    plt.legend(loc='best') 
    plt.title('Cointegration Normalized Residual')
    plt.xlabel('Date')
    plt.ylabel('value')


    fig = plt.figure(figsize=(10, 10))
    bx = fig.add_subplot(212)

    new['asset1'].plot(label='{}'.format(ticket1))
    new['asset2'].plot(label='{}'.format(ticket2))

    bx.plot(new.loc[new['positions1'] == 1].index, \
            new['asset1'][new['positions1'] == 1], \
            lw=0, marker='^', markersize=8, \
            label='LONG {}'.format(ticket1), c='g', alpha=0.7)
    bx.plot(new.loc[new['positions1'] == -1].index, \
            new['asset1'][new['positions1'] == -1], \
            lw=0, marker='v', markersize=8, \
            label='SHORT {}'.format(ticket1), c='r', alpha=0.7)
    bx.plot(new.loc[new['positions2'] == 1].index, \
            new['asset2'][new['positions2'] == 1], \
            lw=0, marker=2, markersize=12, \
            label='LONG {}'.format(ticket2), c='g', alpha=0.9)
    bx.plot(new.loc[new['positions2'] == -1].index, \
            new['asset2'][new['positions2'] == -1], \
            lw=0, marker=3, markersize=12, \
            label='SHORT {}'.format(ticket2), c='r', alpha=0.9)

    bx.legend(loc='best')
    plt.title('Pair Trading')
    plt.xlabel('Date')
    plt.ylabel('price')
    plt.grid(True)
    plt.show()



def main():

    start_date = '2013-01-01'
    end_date = '2014-12-31'
    ticket1 = 'NVDA'
    ticket2 = 'AMD'
    #dataNVDA = open("dataNVIDIA.txt","w")
    #dataAMD = open("dataAMD.txt","w")
    data_frame_1 = yf.download(ticket1, start = start_date, end = end_date)
    data_frame_2 = yf.download(ticket2, start = start_date, end = end_date)
    #data1 = "The data from NVIDIA is "+str(data_frame_1)
    #data2 = "The data from AMD is "+str(data_frame_2)
    #dataNVDA.write(data1)
    #dataAMD.write(data2)
    #dataNVDA.close()
    #dataAMD.close()
    signal = signal_gen(data_frame_1, data_frame_2, cointegration)
    plot(signal,ticket1,ticket2)




if __name__ == '__main__' :
    main()
