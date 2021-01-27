import numpy as np

# Time Series Preprocessing

def truncate(data,lookback=168,forecast=48,len_test=10000,len_val=10000,target='full'):
  """
  data : standardized dataframe
  lookback : number of previous time steps to consider
  forecast : number of future time steps to consider
  len_test/len_val : size of the test/validation dataste
  target : Pollutants to predict between {'full','PM10','O3','NO2'}
  """
    x_train,y_train,x_val,y_val,x_test,y_test=[],[],[],[],[],[]
    n=len(data)   
    for i in range(0,n-forecast-lookback+1):
        if i<n-len_test-len_val-lookback-forecast+1:
            x_train.append(data[i:i+lookback])
            y_train.append(data[i+lookback:i+lookback+forecast,:3])
        elif i<n-len_test-lookback-forecast+1:
            x_val.append(data[i:i+lookback])
            y_val.append(data[i+lookback:i+lookback+forecast,:3])
        else:
            x_test.append(data[i:i+lookback])
            y_test.append(data[i+lookback:i+lookback+forecast,:3])
    if target=='full':
        return np.array(x_train),np.array(y_train),np.array(x_val),np.array(y_val),np.array(x_test),np.array(y_test)
    elif target == 'PM10':
        return np.array(x_train),np.array(y_train)[:,:,0],np.array(x_val),np.array(y_val)[:,:,0],np.array(x_test),np.array(y_test)[:,:,0]
    elif target =='O3':
        return np.array(x_train),np.array(y_train)[:,:,1],np.array(x_val),np.array(y_val)[:,:,1],np.array(x_test),np.array(y_test)[:,:,1]
    elif target=='NO2':
        return np.array(x_train),np.array(y_train)[:,:,2],np.array(x_val),np.array(y_val)[:,:,2],np.array(x_test),np.array(y_test)[:,:,2]
