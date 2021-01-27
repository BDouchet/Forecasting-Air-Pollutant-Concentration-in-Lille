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


#Metrics
from sklearn.metrics import mean_absolute_error,r2_score

if yp_val.shape[-1]==3: # if the predictions contain the three pollutants
    poll=['PM10','O3','NO2']
    df_met=pd.DataFrame(data=None,
                        index=['mae','R2','mae6','mae12','mae24','mae48','R6','R12','R24','R48'],columns=poll)
    for i in range(3):
        mae=mean_absolute_error(y_val[:,:,i],yp_val[:,:,i])
        df_met.at['mae',poll[i]]=mae
        R2=r2_score(y_val[:,:,i],yp_val[:,:,i])
        df_met.at['R2',poll[i]]=R2
        for j in [6,12,24,48]:
            mae=mean_absolute_error(y_val[:,j-1,i],yp_val[:,j-1,i])
            R2=r2_score(y_val[:,j-1,i],yp_val[:,j-1,i])
            df_met.at['mae'+str(j),poll[i]]=mae
            df_met.at['R'+str(j),poll[i]]=R2   
    
if yp_val.shape[-1]==1: # if the predictions contain only one pollutant
    df_met=pd.DataFrame(data=None,
                        index=['mae','R2','mae6','mae12','mae24','mae48','R6','R12','R24','R48'],
                        columns=['poll'])
    mae=mean_absolute_error(y_val[:,:],yp_val[:,:,0])
    df_met.at['mae','poll']=mae
    R2=r2_score(y_val[:,:],yp_val[:,:,0])
    df_met.at['R2','poll']=R2
    for j in [6,12,24,48]:
        mae=mean_absolute_error(y_val[:,j-1],yp_val[:,j-1,0])
        r2=r2_score(y_val[:,j-1],yp_val[:,j-1,0])
        df_met.at['mae'+str(j),'poll']=mae
        df_met.at['R'+str(j),'poll']=r2

print(df_met)
