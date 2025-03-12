---
layout: post
permalink: /class-raining/
---

<h1>Binary Classification predicting is raining or not</h1>
<p> This project purpose is to predicting status of raining (yes/no) using many numerical variable such as pressure, maxtemp, etc.</p>


```python
import pandas as pd
data=pd.read_csv("C:/Users/User/Documents/rainfall_train.csv")
data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>day</th>
      <th>pressure</th>
      <th>maxtemp</th>
      <th>temparature</th>
      <th>mintemp</th>
      <th>dewpoint</th>
      <th>humidity</th>
      <th>cloud</th>
      <th>sunshine</th>
      <th>winddirection</th>
      <th>windspeed</th>
      <th>rainfall</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>1017.4</td>
      <td>21.2</td>
      <td>20.6</td>
      <td>19.9</td>
      <td>19.4</td>
      <td>87.0</td>
      <td>88.0</td>
      <td>1.1</td>
      <td>60.0</td>
      <td>17.2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2</td>
      <td>1019.5</td>
      <td>16.2</td>
      <td>16.9</td>
      <td>15.8</td>
      <td>15.4</td>
      <td>95.0</td>
      <td>91.0</td>
      <td>0.0</td>
      <td>50.0</td>
      <td>21.9</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>3</td>
      <td>1024.1</td>
      <td>19.4</td>
      <td>16.1</td>
      <td>14.6</td>
      <td>9.3</td>
      <td>75.0</td>
      <td>47.0</td>
      <td>8.3</td>
      <td>70.0</td>
      <td>18.1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>4</td>
      <td>1013.4</td>
      <td>18.1</td>
      <td>17.8</td>
      <td>16.9</td>
      <td>16.8</td>
      <td>95.0</td>
      <td>95.0</td>
      <td>0.0</td>
      <td>60.0</td>
      <td>35.6</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>5</td>
      <td>1021.8</td>
      <td>21.3</td>
      <td>18.4</td>
      <td>15.2</td>
      <td>9.6</td>
      <td>52.0</td>
      <td>45.0</td>
      <td>3.6</td>
      <td>40.0</td>
      <td>24.8</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2185</th>
      <td>2185</td>
      <td>361</td>
      <td>1014.6</td>
      <td>23.2</td>
      <td>20.6</td>
      <td>19.1</td>
      <td>19.9</td>
      <td>97.0</td>
      <td>88.0</td>
      <td>0.1</td>
      <td>40.0</td>
      <td>22.1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2186</th>
      <td>2186</td>
      <td>362</td>
      <td>1012.4</td>
      <td>17.2</td>
      <td>17.3</td>
      <td>16.3</td>
      <td>15.3</td>
      <td>91.0</td>
      <td>88.0</td>
      <td>0.0</td>
      <td>50.0</td>
      <td>35.3</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2187</th>
      <td>2187</td>
      <td>363</td>
      <td>1013.3</td>
      <td>19.0</td>
      <td>16.3</td>
      <td>14.3</td>
      <td>12.6</td>
      <td>79.0</td>
      <td>79.0</td>
      <td>5.0</td>
      <td>40.0</td>
      <td>32.9</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2188</th>
      <td>2188</td>
      <td>364</td>
      <td>1022.3</td>
      <td>16.4</td>
      <td>15.2</td>
      <td>13.8</td>
      <td>14.7</td>
      <td>92.0</td>
      <td>93.0</td>
      <td>0.1</td>
      <td>40.0</td>
      <td>18.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2189</th>
      <td>2189</td>
      <td>365</td>
      <td>1013.8</td>
      <td>21.2</td>
      <td>19.1</td>
      <td>18.0</td>
      <td>18.0</td>
      <td>89.0</td>
      <td>88.0</td>
      <td>1.0</td>
      <td>70.0</td>
      <td>48.0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>2190 rows × 13 columns</p>
</div>



<h1> Preprocessing Data</h1>
<p> we must doing prepocessing data so the data is clean. The preprocessing that we used are detecting missing value and get type of dataset</p>


<h2>Missing value</h2>


```python
data.isnull().sum()
```




    id               0
    day              0
    pressure         0
    maxtemp          0
    temparature      0
    mintemp          0
    dewpoint         0
    humidity         0
    cloud            0
    sunshine         0
    winddirection    0
    windspeed        0
    rainfall         0
    dtype: int64



there's no missing value in dataset

<h2>Dimension of the data</h2>


```python
print(f"Len of the rainfall dataset is {len(data)}")
print('------------------------')
print(f"variabel of  dataset are {data.columns.tolist()}")

```

    Len of the rainfall dataset is 2190
    ------------------------
    variabel of  dataset are ['id', 'day', 'pressure', 'maxtemp', 'temparature', 'mintemp', 'dewpoint', 'humidity', 'cloud', 'sunshine', 'winddirection', 'windspeed', 'rainfall']
    

- type of data


```python
data.dtypes
```




    id                 int64
    day                int64
    pressure         float64
    maxtemp          float64
    temparature      float64
    mintemp          float64
    dewpoint         float64
    humidity         float64
    cloud            float64
    sunshine         float64
    winddirection    float64
    windspeed        float64
    rainfall           int64
    dtype: object



all the variabel is numerical and variabel y has transformed to numerical class

<h1>EXPLANATORY DATA ANALYSIS</h1>
<p text-indent=10px> After we checking and cleaning the dataset, we must doing explanatory data analysis to give us insight and summary od dataset, since we cant use all 365, we choose another option for example we want to convert data quarterly first and visualize based on that</p>

for example


```python
days=data['day']
#get date 2024 and matching with day
date_per=pd.to_datetime("2024-01-01") + pd.to_timedelta(days,unit='D')
matching_data=pd.DataFrame({
    'day':days,
    'date':date_per
})
#convert to period Q and acces string using. str and show only last two words
matching_data['quarter']=date_per.dt.to_period('Q').astype(str).str[-2:]
matching_data.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>day</th>
      <th>date</th>
      <th>quarter</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2024-01-02</td>
      <td>Q1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>2024-01-03</td>
      <td>Q1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>2024-01-04</td>
      <td>Q1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>2024-01-05</td>
      <td>Q1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>2024-01-06</td>
      <td>Q1</td>
    </tr>
  </tbody>
</table>
</div>



now we implement it to dataset


```python
data['date']=pd.to_datetime("2024-01-01") + pd.to_timedelta(days,unit='D')
data['quarter']=data['date'].dt.to_period('Q').astype(str).str[-2:]
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>day</th>
      <th>pressure</th>
      <th>maxtemp</th>
      <th>temparature</th>
      <th>mintemp</th>
      <th>dewpoint</th>
      <th>humidity</th>
      <th>cloud</th>
      <th>sunshine</th>
      <th>winddirection</th>
      <th>windspeed</th>
      <th>rainfall</th>
      <th>date</th>
      <th>quarter</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>1017.4</td>
      <td>21.2</td>
      <td>20.6</td>
      <td>19.9</td>
      <td>19.4</td>
      <td>87.0</td>
      <td>88.0</td>
      <td>1.1</td>
      <td>60.0</td>
      <td>17.2</td>
      <td>1</td>
      <td>2024-01-02</td>
      <td>Q1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2</td>
      <td>1019.5</td>
      <td>16.2</td>
      <td>16.9</td>
      <td>15.8</td>
      <td>15.4</td>
      <td>95.0</td>
      <td>91.0</td>
      <td>0.0</td>
      <td>50.0</td>
      <td>21.9</td>
      <td>1</td>
      <td>2024-01-03</td>
      <td>Q1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>3</td>
      <td>1024.1</td>
      <td>19.4</td>
      <td>16.1</td>
      <td>14.6</td>
      <td>9.3</td>
      <td>75.0</td>
      <td>47.0</td>
      <td>8.3</td>
      <td>70.0</td>
      <td>18.1</td>
      <td>1</td>
      <td>2024-01-04</td>
      <td>Q1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>4</td>
      <td>1013.4</td>
      <td>18.1</td>
      <td>17.8</td>
      <td>16.9</td>
      <td>16.8</td>
      <td>95.0</td>
      <td>95.0</td>
      <td>0.0</td>
      <td>60.0</td>
      <td>35.6</td>
      <td>1</td>
      <td>2024-01-05</td>
      <td>Q1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>5</td>
      <td>1021.8</td>
      <td>21.3</td>
      <td>18.4</td>
      <td>15.2</td>
      <td>9.6</td>
      <td>52.0</td>
      <td>45.0</td>
      <td>3.6</td>
      <td>40.0</td>
      <td>24.8</td>
      <td>0</td>
      <td>2024-01-06</td>
      <td>Q1</td>
    </tr>
  </tbody>
</table>
</div>




```python
group_pressure=data.groupby('quarter')['pressure'].mean()
group_pressure.values
```




    array([1017.7046595 , 1009.29333333, 1009.52581818, 1018.05028463])




```python
#defining column x variable
kol_x=[i for i in data.columns.tolist() if i!='id' and i!='day' and i!='date' and i!='rainfall' and i!='quarter']
#make line plot for all x variable
from matplotlib import pyplot as plt
nrow=2
ncol=5
fig,ax=plt.subplots(nrow,ncol,figsize=(11,8))
for (i,j),k in zip([(i,j) for i in range(nrow) for j in range(ncol)], kol_x):
    group_data=data.groupby('quarter')[k].mean()
    xtic=[i for i in range(4)]
    plotting=ax[i][j].plot(group_data.index,group_data.values,marker='o')
    ax[i][j].set_xticks(xtic)
    ax[i][j].spines['top'].set_visible(False)
    ax[i][j].spines['right'].set_visible(False)
    ax[i][j].spines['left'].set_visible(False)
    ax[i][j].set_yticks([])
    xval=plotting[0].get_xdata()
    yval=plotting[0].get_ydata()
    for l,m in zip(xval,yval):
        ytex=round(m,2)
        ax[i][j].text(l,m,f"{ytex}",fontsize=8)
    ax[i][j].set_title(f"{k}",loc='left',fontweight='bold',fontsize=8)
    
plt.tight_layout()


```


    
![png]({{ site.baseurl }}/assets/image-bin/data_binary_17_0.png)
    


- based on quareter, variabel mintemp, temperature maxtemp, wind direction have same pattern,
-  presure, humadity, cloud, and windspeed are increased when we enter Q4


```python
fig,ax=plt.subplots(1,4,figsize=(10,10))
qua=data['quarter'].unique().tolist()
for i,j in zip(ax,qua):
    tabel_freq=data[data['quarter']==j]['rainfall'].value_counts()
    i.pie(tabel_freq.values,labels=tabel_freq.index,wedgeprops=dict(width=0.3))
    i.set_title(f"distribution of rainfall in \n{j}", fontsize=10)
plt.tight_layout()
```


    
![png]({{ site.baseurl }}/assets/image-bin/data_binary_19_0.png)
    


- the rainfall 'yess' is dominating in every quarter in a year

**Check correlatio in every variable x**


```python
#check corelation
corr=data[kol_x].corr()
corr
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pressure</th>
      <th>maxtemp</th>
      <th>temparature</th>
      <th>mintemp</th>
      <th>dewpoint</th>
      <th>humidity</th>
      <th>cloud</th>
      <th>sunshine</th>
      <th>winddirection</th>
      <th>windspeed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>pressure</th>
      <td>1.000000</td>
      <td>-0.800499</td>
      <td>-0.816531</td>
      <td>-0.814453</td>
      <td>-0.817008</td>
      <td>-0.119949</td>
      <td>0.098600</td>
      <td>-0.257163</td>
      <td>-0.643293</td>
      <td>0.266012</td>
    </tr>
    <tr>
      <th>maxtemp</th>
      <td>-0.800499</td>
      <td>1.000000</td>
      <td>0.982932</td>
      <td>0.965529</td>
      <td>0.906703</td>
      <td>-0.072615</td>
      <td>-0.289047</td>
      <td>0.452387</td>
      <td>0.662235</td>
      <td>-0.354168</td>
    </tr>
    <tr>
      <th>temparature</th>
      <td>-0.816531</td>
      <td>0.982932</td>
      <td>1.000000</td>
      <td>0.987150</td>
      <td>0.933617</td>
      <td>-0.025016</td>
      <td>-0.249355</td>
      <td>0.414019</td>
      <td>0.668963</td>
      <td>-0.342262</td>
    </tr>
    <tr>
      <th>mintemp</th>
      <td>-0.814453</td>
      <td>0.965529</td>
      <td>0.987150</td>
      <td>1.000000</td>
      <td>0.941342</td>
      <td>0.009891</td>
      <td>-0.219399</td>
      <td>0.379497</td>
      <td>0.663828</td>
      <td>-0.328871</td>
    </tr>
    <tr>
      <th>dewpoint</th>
      <td>-0.817008</td>
      <td>0.906703</td>
      <td>0.933617</td>
      <td>0.941342</td>
      <td>1.000000</td>
      <td>0.153390</td>
      <td>-0.088446</td>
      <td>0.249676</td>
      <td>0.643073</td>
      <td>-0.312179</td>
    </tr>
    <tr>
      <th>humidity</th>
      <td>-0.119949</td>
      <td>-0.072615</td>
      <td>-0.025016</td>
      <td>0.009891</td>
      <td>0.153390</td>
      <td>1.000000</td>
      <td>0.584854</td>
      <td>-0.541592</td>
      <td>-0.012430</td>
      <td>0.062285</td>
    </tr>
    <tr>
      <th>cloud</th>
      <td>0.098600</td>
      <td>-0.289047</td>
      <td>-0.249355</td>
      <td>-0.219399</td>
      <td>-0.088446</td>
      <td>0.584854</td>
      <td>1.000000</td>
      <td>-0.805128</td>
      <td>-0.127087</td>
      <td>0.184698</td>
    </tr>
    <tr>
      <th>sunshine</th>
      <td>-0.257163</td>
      <td>0.452387</td>
      <td>0.414019</td>
      <td>0.379497</td>
      <td>0.249676</td>
      <td>-0.541592</td>
      <td>-0.805128</td>
      <td>1.000000</td>
      <td>0.272235</td>
      <td>-0.241752</td>
    </tr>
    <tr>
      <th>winddirection</th>
      <td>-0.643293</td>
      <td>0.662235</td>
      <td>0.668963</td>
      <td>0.663828</td>
      <td>0.643073</td>
      <td>-0.012430</td>
      <td>-0.127087</td>
      <td>0.272235</td>
      <td>1.000000</td>
      <td>-0.192417</td>
    </tr>
    <tr>
      <th>windspeed</th>
      <td>0.266012</td>
      <td>-0.354168</td>
      <td>-0.342262</td>
      <td>-0.328871</td>
      <td>-0.312179</td>
      <td>0.062285</td>
      <td>0.184698</td>
      <td>-0.241752</td>
      <td>-0.192417</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



- Based on table correlation, there's strong positive correlation between temperature (mintemp and maxtemp too) and dew point, pressure and dewpoint, temperature have strong positive correlation
- there's negative strong corelation between cloud and sunshine

plot just strong corellation (positive and negative)


```python
#temperature vs dew point
rainfall_type=data['rainfall'].unique().tolist()
for i,j in zip(rainfall_type,['blue','yellow']):
    ambil=data[data['rainfall']==i]
    plt.scatter(ambil['temparature'],ambil['dewpoint'],color=j,label=i)
    plt.xlabel('temparature')
    plt.ylabel('dewpoint')
    plt.title('temparature vs dewpoint')
    plt.legend()
```


    
![png]({{ site.baseurl }}/assets/image-bin/data_binary_25_0.png)
    



```python
#presuere vs dew point
fig,ax=plt.subplots(1,2,figsize=(5,3))
for i,j in zip(range(2),['temparature','dewpoint']):
    rainfall_type=data['rainfall'].unique().tolist()
    for k,l in zip(rainfall_type,['blue','yellow']):
        ambil=data[data['rainfall']==k]
        ax[i].scatter(ambil['pressure'],ambil[j],color=l,label=i)
        ax[i].set_xlabel('presurre')
        ax[i].set_ylabel(j)
        ax[i].set_title(f'presurre vs {j}')
        ax[i].legend()
plt.tight_layout()

```


    
![png]({{ site.baseurl }}/assets/image-bin/data_binary_26_0.png)
    



```python
#cloud and shunshine
rainfall_type=data['rainfall'].unique().tolist()
for i,j in zip(rainfall_type,['blue','yellow']):
    ambil=data[data['rainfall']==i]
    plt.scatter(ambil['cloud'],ambil['sunshine'],color=j,label=i)
    plt.xlabel('cloud')
    plt.ylabel('sunshine')
    plt.title('cloud vs sunshine')
    plt.legend()
```


    
![png]({{ site.baseurl }}/assets/image-bin/data_binary_27_0.png)
    



```python
data['rainfall'].unique().tolist()
```




    [1, 0]



**correlation every variable x vs y**


```python
fig,ax=plt.subplots(2,5,figsize=(8,5))
for (i,j),k in zip([(i,j) for i in range (2) for j in range (5)],kol_x):
    kum=[]
    for l in data['rainfall'].unique().tolist():
        ambil=data[data['rainfall']==l]
        get_data=ambil[k]
        kum.append(get_data)
    ax[i][j].boxplot(kum,labels=data['rainfall'].unique().tolist())
    ax[i][j].set_title(k)
plt.tight_layout()
```

    C:\Users\User\AppData\Local\Temp\ipykernel_8436\1658684302.py:8: MatplotlibDeprecationWarning: The 'labels' parameter of boxplot() has been renamed 'tick_labels' since Matplotlib 3.9; support for the old name will be dropped in 3.11.
      ax[i][j].boxplot(kum,labels=data['rainfall'].unique().tolist())
    C:\Users\User\AppData\Local\Temp\ipykernel_8436\1658684302.py:8: MatplotlibDeprecationWarning: The 'labels' parameter of boxplot() has been renamed 'tick_labels' since Matplotlib 3.9; support for the old name will be dropped in 3.11.
      ax[i][j].boxplot(kum,labels=data['rainfall'].unique().tolist())
    C:\Users\User\AppData\Local\Temp\ipykernel_8436\1658684302.py:8: MatplotlibDeprecationWarning: The 'labels' parameter of boxplot() has been renamed 'tick_labels' since Matplotlib 3.9; support for the old name will be dropped in 3.11.
      ax[i][j].boxplot(kum,labels=data['rainfall'].unique().tolist())
    C:\Users\User\AppData\Local\Temp\ipykernel_8436\1658684302.py:8: MatplotlibDeprecationWarning: The 'labels' parameter of boxplot() has been renamed 'tick_labels' since Matplotlib 3.9; support for the old name will be dropped in 3.11.
      ax[i][j].boxplot(kum,labels=data['rainfall'].unique().tolist())
    C:\Users\User\AppData\Local\Temp\ipykernel_8436\1658684302.py:8: MatplotlibDeprecationWarning: The 'labels' parameter of boxplot() has been renamed 'tick_labels' since Matplotlib 3.9; support for the old name will be dropped in 3.11.
      ax[i][j].boxplot(kum,labels=data['rainfall'].unique().tolist())
    C:\Users\User\AppData\Local\Temp\ipykernel_8436\1658684302.py:8: MatplotlibDeprecationWarning: The 'labels' parameter of boxplot() has been renamed 'tick_labels' since Matplotlib 3.9; support for the old name will be dropped in 3.11.
      ax[i][j].boxplot(kum,labels=data['rainfall'].unique().tolist())
    C:\Users\User\AppData\Local\Temp\ipykernel_8436\1658684302.py:8: MatplotlibDeprecationWarning: The 'labels' parameter of boxplot() has been renamed 'tick_labels' since Matplotlib 3.9; support for the old name will be dropped in 3.11.
      ax[i][j].boxplot(kum,labels=data['rainfall'].unique().tolist())
    C:\Users\User\AppData\Local\Temp\ipykernel_8436\1658684302.py:8: MatplotlibDeprecationWarning: The 'labels' parameter of boxplot() has been renamed 'tick_labels' since Matplotlib 3.9; support for the old name will be dropped in 3.11.
      ax[i][j].boxplot(kum,labels=data['rainfall'].unique().tolist())
    C:\Users\User\AppData\Local\Temp\ipykernel_8436\1658684302.py:8: MatplotlibDeprecationWarning: The 'labels' parameter of boxplot() has been renamed 'tick_labels' since Matplotlib 3.9; support for the old name will be dropped in 3.11.
      ax[i][j].boxplot(kum,labels=data['rainfall'].unique().tolist())
    C:\Users\User\AppData\Local\Temp\ipykernel_8436\1658684302.py:8: MatplotlibDeprecationWarning: The 'labels' parameter of boxplot() has been renamed 'tick_labels' since Matplotlib 3.9; support for the old name will be dropped in 3.11.
      ax[i][j].boxplot(kum,labels=data['rainfall'].unique().tolist())
    


    
![png]({{ site.baseurl }}/assets/image-bin/data_binary_30_1.png)
    


- cloud, sunshine, humidity, and windspeed distinguish between classm so there're relationship with y
- other than cloud, sunshine, humidity, and windspeed  are not separable, so there're no relationship y

<h1>CLASSIFICATION</h1>
<p>We make model classification to predicting rainfall or not using machine learning model, before that we check distribution of class and relationship between variable dependent and independent</p>


```python
print(f"class label in this dataset is {data['rainfall'].unique().tolist()}")
print('---------------------')
data['rainfall'].value_counts()
```

    class label in this dataset is [1, 0]
    ---------------------
    




    rainfall
    1    1650
    0     540
    Name: count, dtype: int64



- between class 1 and 0 has not equal or the dataset is imbalance.
- because there're some variable that has relationship with y we want to make two model:
    - logistic regression for cloud, sunshine, humidity, and windspeed
    - XGBOOST for all variable

**remove variabel that not important** : remove days and id


```python
data_new=data.drop(['id','day'],axis=1)
```

**logistic regression**


```python
from sklearn.model_selection import train_test_split,StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
data_cl=data_new.copy()
kol_log=['cloud','sunshine','windspeed','humidity']
X=data_cl[kol_log]
y=data_cl['rainfall']
#split teh dara into validation set, stratify keep distribution of class is same
xtrain,xval,ytrain,yval=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
xtrain.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cloud</th>
      <th>sunshine</th>
      <th>windspeed</th>
      <th>humidity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>919</th>
      <td>79.0</td>
      <td>0.3</td>
      <td>24.4</td>
      <td>82.0</td>
    </tr>
    <tr>
      <th>118</th>
      <td>64.0</td>
      <td>6.3</td>
      <td>29.6</td>
      <td>73.0</td>
    </tr>
    <tr>
      <th>784</th>
      <td>79.0</td>
      <td>0.2</td>
      <td>23.3</td>
      <td>79.0</td>
    </tr>
    <tr>
      <th>719</th>
      <td>81.0</td>
      <td>0.6</td>
      <td>38.3</td>
      <td>83.0</td>
    </tr>
    <tr>
      <th>327</th>
      <td>60.0</td>
      <td>6.8</td>
      <td>35.3</td>
      <td>65.0</td>
    </tr>
  </tbody>
</table>
</div>



using : KFOLD : evaluate model performance divide subset data into K fold. K-1 fold for training and remaining fold for validation data and repeated for K times. Using stratifiedkfold when class label is imbalanced


```python
#make kf
kf=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
#since imbalance class we use roc_auc score
roc_train=[]
roc_val=[]
for tr_index,val_index in kf.split(xtrain,ytrain):
    xtrain_fold,xval_fold=xtrain.iloc[tr_index],xtrain.iloc[val_index]
    ytrain_fold,yval_fold=ytrain.iloc[tr_index],ytrain.iloc[val_index]
     #scaler
    scaler=StandardScaler()
    tr_clean=scaler.fit_transform(xtrain_fold)
    val_clean=scaler.transform(xval_fold)
    #make model
    model_log=LogisticRegression(class_weight='balanced',solver='liblinear',random_state=42)
    model_log.fit(tr_clean,ytrain_fold)
    #extract only columns 1 column 1 is positive class
    tr_pred=model_log.predict_proba(tr_clean)[:,1]
    val_pred=model_log.predict_proba(val_clean)[:,1]
    #score
    score_tr=roc_auc_score(ytrain_fold,tr_pred)
    score_val=roc_auc_score(yval_fold,val_pred)
    roc_train.append(score_tr)
    roc_val.append(score_val)
print('roc auc score training :')
print('-----------------------------------------')
print(roc_train)
print('-----------------------------------------')
print('roc auc score validation :')
print('-----------------------------------------')
print(roc_val)
```

    roc auc score training :
    -----------------------------------------
    [np.float64(0.8910902503293807), np.float64(0.8854111769872639), np.float64(0.8888596952180765), np.float64(0.8953516377649325), np.float64(0.881505627080049)]
    -----------------------------------------
    roc auc score validation :
    -----------------------------------------
    [np.float64(0.8763932427725531), np.float64(0.8982061999303378), np.float64(0.8841613812544045), np.float64(0.8601127554615927), np.float64(0.913715644820296)]
    

- for imbalance data, using class_weight ='balanced' to predict minority class is suits and liblinear solver is works well for binary classification
- ROC-AUC score : good for imbalanced data, its good for model know to separate between class
- based on training and validation roc-auc the model is not overfitting and works well because there' tiny gap between them. overfitting happen when training set is larger than validation set


```python
#perform logistic regression in validation
standar=StandardScaler()
val_new=standar.fit_transform(xval)
pred=model_log.predict_proba(val_new)[:,1]
rocscore=roc_auc_score(yval,pred)
rocscore
```




    np.float64(0.8781144781144782)



- thee roc auc score above is considered godd

**Perform logistic regression in test set**


```python
import pandas as pd
data_test=pd.read_csv("C:/Users/User/Documents/rainfall_test.csv")
data_test.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>day</th>
      <th>pressure</th>
      <th>maxtemp</th>
      <th>temparature</th>
      <th>mintemp</th>
      <th>dewpoint</th>
      <th>humidity</th>
      <th>cloud</th>
      <th>sunshine</th>
      <th>winddirection</th>
      <th>windspeed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>725</th>
      <td>2915</td>
      <td>361</td>
      <td>1020.8</td>
      <td>18.2</td>
      <td>17.6</td>
      <td>16.1</td>
      <td>13.7</td>
      <td>96.0</td>
      <td>95.0</td>
      <td>0.0</td>
      <td>20.0</td>
      <td>34.3</td>
    </tr>
    <tr>
      <th>726</th>
      <td>2916</td>
      <td>362</td>
      <td>1011.7</td>
      <td>23.2</td>
      <td>18.1</td>
      <td>16.0</td>
      <td>16.0</td>
      <td>78.0</td>
      <td>80.0</td>
      <td>1.6</td>
      <td>40.0</td>
      <td>25.2</td>
    </tr>
    <tr>
      <th>727</th>
      <td>2917</td>
      <td>363</td>
      <td>1022.7</td>
      <td>21.0</td>
      <td>18.5</td>
      <td>17.0</td>
      <td>15.5</td>
      <td>92.0</td>
      <td>96.0</td>
      <td>0.0</td>
      <td>50.0</td>
      <td>21.9</td>
    </tr>
    <tr>
      <th>728</th>
      <td>2918</td>
      <td>364</td>
      <td>1014.4</td>
      <td>21.0</td>
      <td>20.0</td>
      <td>19.7</td>
      <td>19.8</td>
      <td>94.0</td>
      <td>93.0</td>
      <td>0.0</td>
      <td>50.0</td>
      <td>39.5</td>
    </tr>
    <tr>
      <th>729</th>
      <td>2919</td>
      <td>365</td>
      <td>1020.9</td>
      <td>22.2</td>
      <td>18.8</td>
      <td>17.0</td>
      <td>13.3</td>
      <td>79.0</td>
      <td>89.0</td>
      <td>0.2</td>
      <td>60.0</td>
      <td>50.6</td>
    </tr>
  </tbody>
</table>
</div>




```python
X_test=data_test[kol_log]
xtest_str=standar.transform(X_test)
predict_test=model_log.predict_proba(xtest_str)[:,1]
#threshold 0.5 if more than 0,5 we conver to 1
threshold=0.5
def binary_class(x):
    if x>=0.5:
        return 1
    else:
        return 0
binary=list(map(binary_class,predict_test))
datakum=pd.DataFrame({
    'id':data_test['id'],
    'class':binary
})
datakum
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2190</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2191</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2192</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2193</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2194</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>725</th>
      <td>2915</td>
      <td>1</td>
    </tr>
    <tr>
      <th>726</th>
      <td>2916</td>
      <td>1</td>
    </tr>
    <tr>
      <th>727</th>
      <td>2917</td>
      <td>1</td>
    </tr>
    <tr>
      <th>728</th>
      <td>2918</td>
      <td>1</td>
    </tr>
    <tr>
      <th>729</th>
      <td>2919</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>730 rows × 2 columns</p>
</div>




```python
datakum['class'].value_counts()
```




    class
    1    662
    0     68
    Name: count, dtype: int64



**using xgboost in all x variable** : using tree rather than probability, learn from mistake and improve based on them (like add more weighting for rare class)


```python
from sklearn.model_selection import train_test_split, StratifiedKFold
import xgboost
from sklearn.metrics import roc_auc_score
data_xg=data_new.copy()
X=data_xg[kol_x]
y=data_xg['rainfall']
#split teh dara into validation set, stratify keep distribution of class is same
xtrain,xval,ytrain,yval=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
ytrain.shape
```




    (1752,)




```python
#make kf
kf=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
tab_count=y.value_counts()
scale=tab_count.loc[1]/tab_count.loc[0]
#since imbalance class we use roc_auc score
roc_train=[]
roc_val=[]
for tr_index,val_index in kf.split(xtrain,ytrain):
    xtrain_fold,xval_fold=xtrain.iloc[tr_index],xtrain.iloc[val_index]
    ytrain_fold,yval_fold=ytrain.iloc[tr_index],ytrain.iloc[val_index]
    #make model
    model_xg=xgboost.XGBClassifier(scale_pos_weight=scale,
                                    eval_metric='auc',
                                    max_depth=2,
                                    reg_alpha=0.6,
                                    reg_lambda=9,
                                    n_estimators=200,
                                   gamma=5)
    model_xg.fit(xtrain_fold,ytrain_fold)
    #extract only columns 1 column 1 is positive class
    tr_pred=model_xg.predict_proba(xtrain_fold)[:,1]
    val_pred=model_xg.predict_proba(xval_fold)[:,1]
    #score
    score_tr=roc_auc_score(ytrain_fold,tr_pred)
    score_val=roc_auc_score(yval_fold,val_pred)
    roc_train.append(score_tr)
    roc_val.append(score_val)
print('roc auc score training :')
print('-----------------------------------------')
print(roc_train)
print('-----------------------------------------')
print('roc auc score validation :')
print('-----------------------------------------')
print(roc_val)
```

    roc auc score training :
    -----------------------------------------
    [np.float64(0.9155097167325427), np.float64(0.9055198726394378), np.float64(0.9141842923454194), np.float64(0.9202999102294622), np.float64(0.9091839639166229)]
    -----------------------------------------
    roc auc score validation :
    -----------------------------------------
    [np.float64(0.8738244514106583), np.float64(0.9029954719609892), np.float64(0.893256694855532), np.float64(0.8677325581395351), np.float64(0.9094212473572939)]
    

 model_log=xgboost.XGBClassifier(scale_pos_weight=scale,
                                    eval_metric='auc',
                                    max_depth=2,
                                    reg_alpha=0.6,
                                    reg_lambda=9,
                                    n_estimators=200,
                                   gamma=5)

- scale_pos_weigt : for imbalanced data, the scale is class 1/class 0
- using auc-roc for evaluate model
- max deep, how deep tree, lower value - more generalize higher value tree more memorize data training, so model can overfitting
- reg-alpha = regularization L1 higher value model more generalize (its like keep important feature)
- reg lambda =regularization L2, same as reg-alpha (adjust tree grow, higher value more controllable, reduce overfitting)
- n_estimator=number of tree
- gamma = its like decision of split or not, higher value, tree will careful to split data, and low model tree easily split, so the model will complex


```python
#perform logistic regression in validation
pred=model_xg.predict_proba(xval)[:,1]
rocscore=roc_auc_score(yval,pred)
rocscore
```




    np.float64(0.887317620650954)



- the result xgboost is higher than using logistic regression

**XGBoost on test set**


```python
import pandas as pd
data_test=pd.read_csv("C:/Users/User/Documents/rainfall_test.csv")
data_test.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>day</th>
      <th>pressure</th>
      <th>maxtemp</th>
      <th>temparature</th>
      <th>mintemp</th>
      <th>dewpoint</th>
      <th>humidity</th>
      <th>cloud</th>
      <th>sunshine</th>
      <th>winddirection</th>
      <th>windspeed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>725</th>
      <td>2915</td>
      <td>361</td>
      <td>1020.8</td>
      <td>18.2</td>
      <td>17.6</td>
      <td>16.1</td>
      <td>13.7</td>
      <td>96.0</td>
      <td>95.0</td>
      <td>0.0</td>
      <td>20.0</td>
      <td>34.3</td>
    </tr>
    <tr>
      <th>726</th>
      <td>2916</td>
      <td>362</td>
      <td>1011.7</td>
      <td>23.2</td>
      <td>18.1</td>
      <td>16.0</td>
      <td>16.0</td>
      <td>78.0</td>
      <td>80.0</td>
      <td>1.6</td>
      <td>40.0</td>
      <td>25.2</td>
    </tr>
    <tr>
      <th>727</th>
      <td>2917</td>
      <td>363</td>
      <td>1022.7</td>
      <td>21.0</td>
      <td>18.5</td>
      <td>17.0</td>
      <td>15.5</td>
      <td>92.0</td>
      <td>96.0</td>
      <td>0.0</td>
      <td>50.0</td>
      <td>21.9</td>
    </tr>
    <tr>
      <th>728</th>
      <td>2918</td>
      <td>364</td>
      <td>1014.4</td>
      <td>21.0</td>
      <td>20.0</td>
      <td>19.7</td>
      <td>19.8</td>
      <td>94.0</td>
      <td>93.0</td>
      <td>0.0</td>
      <td>50.0</td>
      <td>39.5</td>
    </tr>
    <tr>
      <th>729</th>
      <td>2919</td>
      <td>365</td>
      <td>1020.9</td>
      <td>22.2</td>
      <td>18.8</td>
      <td>17.0</td>
      <td>13.3</td>
      <td>79.0</td>
      <td>89.0</td>
      <td>0.2</td>
      <td>60.0</td>
      <td>50.6</td>
    </tr>
  </tbody>
</table>
</div>




```python
X_test=data_test[kol_x]
predict_test=model_xg.predict_proba(X_test)[:,1]
#threshold 0.5 if more than 0,5 we conver to 1
threshold=0.5
def binary_class(x):
    if x>=0.5:
        return 1
    else:
        return 0
binary=list(map(binary_class,predict_test))
datakum=pd.DataFrame({
    'id':data_test['id'],
    'class':binary
})
datakum
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2190</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2191</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2192</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2193</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2194</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>725</th>
      <td>2915</td>
      <td>1</td>
    </tr>
    <tr>
      <th>726</th>
      <td>2916</td>
      <td>1</td>
    </tr>
    <tr>
      <th>727</th>
      <td>2917</td>
      <td>1</td>
    </tr>
    <tr>
      <th>728</th>
      <td>2918</td>
      <td>1</td>
    </tr>
    <tr>
      <th>729</th>
      <td>2919</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>730 rows × 2 columns</p>
</div>




```python
datakum['class'].value_counts()
```




    class
    1    662
    0     68
    Name: count, dtype: int64


