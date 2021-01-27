# Time Series Preprocessing

from preprocess import truncate

x_train,y_train,x_val,y_val,x_test,y_test=truncate(data,lookback=168,forecast=48,len_test=10000,len_val=10000,target='full')
print(x_train.shape,y_train.shape,x_val.shape,y_val.shape,x_test.shape,y_test.shape)

# -> (47800, 168, 10) (47800, 48, 3) (10000, 168, 10) (10000, 48, 3) (10000, 168, 10) (10000, 48, 3)


# Get predictions
cp_path='a_path'
model_path=cp_path+"model_demo.h5"
model=models.load_model(model_path)
yp_val=model.predict(x_val)
print(yp_val.shape)

# -> (10000, 48, 3)


# Display predictions for the three pollutants
import matplotlib.pyplot as plt

def display_val(fore,start,end):
    """
    fore : Number of hours to forecast (ex : 24 to get the 24 hours forecasts compared to the real avlues at 24 hours)
    start/end : beginning/end of the selected values (between 0 and 9999)
    """
    fore+=-1
    for i,col in enumerate(df.columns[:3]):
         std,mean=df[col].std(),df[col].mean()
         plt.figure(figsize=(25,8))
         plt.plot(y_val[start:end,fore,i]*std+mean,label=col +' truth')
         plt.plot(yp_val[start:end,fore,i]*std+mean,label=col+' val')
         plt.legend(loc='upper left')
         plt.show()
 

#Dislay average metrics for each forecast hour
from sklearn.metrics import mean_absolute_error,r2_score

def errors():
    poll=['PM10','O3','NO2']
    df_mae=pd.DataFrame(data=None,index=['moy']+[str(i+1)for i in range(len(yp_val[0]))],columns=poll)
    for i in range(3):
        mae=r2_score(y_val[:,:,i], yp_val[:,:,i])
        df_mae.at['moy',poll[i]]=mae
        for j in range(len(yp_val[0])):
            mae=r2_score(y_val[:,j,i], yp_val[:,j,i])
            df_mae.at[str(j+1),poll[i]]=mae

    plt.figure(figsize=(15,8))
    plt.plot(df_mae['PM10'][1:],label='R2 PM10',color='blue')
    plt.plot(df_mae['O3'][1:],label='R2 O3',color='green')
    plt.plot(df_mae['NO2'][1:],label='R2 NO2',color='red')
    plt.plot([df_mae.index[1],df_mae.index[-1]],[df_mae['PM10'][0],df_mae['PM10'][0]],label='Avg R2 PM10',color='blue',linestyle='dotted')
    plt.plot([df_mae.index[1],df_mae.index[-1]],[df_mae['O3'][0],df_mae['O3'][0]],label='Avg R2 O3',color='green',linestyle='dotted')
    plt.plot([df_mae.index[1],df_mae.index[-1]],[df_mae['NO2'][0],df_mae['NO2'][0]],label='Avg R2 NO2',color='red',linestyle='dotted')
    plt.legend(loc='upper right',fontsize='x-large')
    plt.ylabel("R2",fontsize='x-large')
    plt.xlabel("Hours of forecast",fontsize='x-large')
    plt.show()
