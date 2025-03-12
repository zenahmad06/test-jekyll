# DESCRIPTIVE STATISTIK

    Aadalah angka-angka yang digunakan untuk membuat mendeskripsikan atau meringkas data, beberapa descriptive statistik biasanya digunakan sepersti menghitung rata-rata dan persebaran data
    
    Kita latihan menggunakan dataset:


```python
import pandas as pd
data=pd.read_csv("C:/Users/User/Documents/CardioGoodFitness.csv")
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
      <th>Product</th>
      <th>Age</th>
      <th>Gender</th>
      <th>Education</th>
      <th>MaritalStatus</th>
      <th>Usage</th>
      <th>Fitness</th>
      <th>Income</th>
      <th>Miles</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>TM195</td>
      <td>18</td>
      <td>Male</td>
      <td>14</td>
      <td>Single</td>
      <td>3</td>
      <td>4</td>
      <td>29562</td>
      <td>112</td>
    </tr>
    <tr>
      <th>1</th>
      <td>TM195</td>
      <td>19</td>
      <td>Male</td>
      <td>15</td>
      <td>Single</td>
      <td>2</td>
      <td>3</td>
      <td>31836</td>
      <td>75</td>
    </tr>
    <tr>
      <th>2</th>
      <td>TM195</td>
      <td>19</td>
      <td>Female</td>
      <td>14</td>
      <td>Partnered</td>
      <td>4</td>
      <td>3</td>
      <td>30699</td>
      <td>66</td>
    </tr>
    <tr>
      <th>3</th>
      <td>TM195</td>
      <td>19</td>
      <td>Male</td>
      <td>12</td>
      <td>Single</td>
      <td>3</td>
      <td>3</td>
      <td>32973</td>
      <td>85</td>
    </tr>
    <tr>
      <th>4</th>
      <td>TM195</td>
      <td>20</td>
      <td>Male</td>
      <td>13</td>
      <td>Partnered</td>
      <td>4</td>
      <td>2</td>
      <td>35247</td>
      <td>47</td>
    </tr>
  </tbody>
</table>
</div>



## Central Tendency 

    Central tendency berarti nilai tengah yang menjelaskan distribusi dari data (probability distribusi). Paling banyak digunakan untuk mengetahui central tendency adalah mean, namun jika bentuk dari persebaran data skewed maka median lebih cocok digunakan
    
    Kita ingin mengetahui central tendency dataset di atas misal variabel/kolom Age:

1. Mean

- salah satu metode yang paling banyak digunakan
- nilai mean dipengaruhi juga oleh outlier data
- rumusnya :


![mean.JPG]({{ site.baseurl }}/assets/image-desc/mean.JPG)

2. median

- nilai tengah dari data yang membagi 2 bagian
- pertama-tama data diurutkan terlebih dahulu dari terendah ke tertinggi
- cara penentuannya 
    - jika jumlah bilangan adalaah ganjil maka:

![medianganjil.JPG]({{ site.baseurl }}/assets/image-desc/medianganjil.JPG)

- Misal jumlah bilangan ada 7, maka posisi median ada di (7+1)/2 ada di posisi 4, setelah itu lihat posisi berdasarkan urutan data ascending
    - jika jumlah bilangan adalaah genap maka:

![mediangenap.JPG]({{ site.baseurl }}/assets/image-desc/mediangenap.JPG)

3. Mode

- Nilai yang sering banyak muncul
- Biasanya digunakan pada variabel tipe nominal, namun mencari mode pada urutan angka juga bisa


```python
subset_age=data['Age']
#mean, round 2 hanya ada di belakang koma 2 angka saj
mean=subset_age.mean().round(2)
median=subset_age.median().round(2)
mode=subset_age.mode()
print("mean variabel Age:",mean)
print("median variabel Age:",median)
print("mode variabel Age:",mode.tolist())
```

    mean variabel Age: 28.79
    median variabel Age: 26.0
    mode variabel Age: [25]
    

# Dispertion of variability

    Mengukur perseberan dari data. Metode yang bisa digunakan untuk menjelaskan dispersion adalah range, standar deviasi, dan IQR (interquartle range). 

1. range

- nilai antara maksimal dan minimal

![range.JPG]({{ site.baseurl }}/assets/image-desc/range.JPG)


```python
ranges=subset_age.max()-subset_age.min()
ranges
```




    np.int64(32)



2. Standar deviasi

- disebut juga root mean square banyak digunakan untuk menjelaskan deviation persebran dari data, semakin tinggi nilainya maka perseberan data semakin luas. Rumusnya 

![std.JPG]({{ site.baseurl }}/assets/image-desc/std.JPG)

- cocok digunakan untuk menjelaskan dispersi ketika datanya berdistirbusi normal dan tidak ada outlier


```python
std_age=data['Age'].std()
print(std_age)
```

    6.943498135399795
    

3. Interquartile range

- Nilai selisih antara quartile 3 atau quantil 75% dengan quartiel 1 atau quantil 25 pada data
- cocok digunakan jika datanya ada outlier dan tidak normal

![iqr.JPG]({{ site.baseurl }}/assets/image-desc/iqr.JPG)


```python
#quartile
quantile_3=data['Age'].quantile(0.75)
quantile_1=data['Age'].quantile(0.25)
#IQR
iqr=quantile_3-quantile_1
print(iqr)
```

    9.0
    

## Shape dari data

    Terdapat 2 metode statistik untuk menjelaskan betuk dari data, yaitu skewness dan kurtosis

1. Skewness

- Menghitung ketidaksimetrisan distribusi data, jika skewness = 0 maka data simetris, jika skewness bernilai positif maka right tail lebih panjang, jika skewness negatif maka left tail lebih panjang pada histogram

![skewness.JPG]({{ site.baseurl }}/assets/image-desc/skewness.JPG)


```python
skew_age=data['Age'].skew()
print(skew_age)
```

    0.9821608255301499
    

    Skewness diatas bernilai positif dan lebih dari 0, artinya data nya skewed  Semakin besar nilainya data nya semakin positive skewed (right tail). Bagaimana bentuk dari datanya? kita dapat membuat histogram


```python
from matplotlib import pyplot as plt
#membuat histogram dengan jumlah batang 20, alpha tingkat transparansi
hist=plt.hist(data['Age'],bins=20,linewidth=2,alpha=0.4)
#menonaktifkan border atas sama kiri pada graph
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
#menghitung median mode dan mean
median=data['Age'].median().round(2)
mode=data['Age'].mode().tolist()
mean=data['Age'].mean()
#mengakses nilai batas sumbu y darri grafik
ylim=plt.gca().get_ylim()
#membuat garis vertikal
plt.axvline(median,linestyle='--',color='red')
plt.axvline(mean,linestyle='--',color='blue')
plt.axvline(mode[0],linestyle='--',color='yellow')
#menambah text
plt.text(median-0.5,ylim[1]+0.7,"Median",color='red',fontsize=8)
plt.text(mode[0]-2,ylim[1]+0.7,"Mode",color='yellow',fontsize=8)
plt.text(mean-0.5,ylim[1]+0.7,"Mean",color='blue',fontsize=8)
#skewness
plt.text(45,ylim[1]-10,f"skewness of \nthis shape : \n\n{skew_age.round(2)}")
```




    Text(45, 21.5, 'skewness of \nthis shape : \n\n0.98')




    
![png]({{ site.baseurl }}/assets/image-desc/data_descriptive_38_1.png)
    


    Dari gambar diatas bentuk distribusti nya adalah positive skewed. Tanda bahwa itu positive skewed (right tail) adalah kebanyakan data berkumpul di bagian kiri paragraf dan tail (ekor) terdapat pada kanan grafik. Pada right tail ini tailnya merupakan nilai-niali yang besar. kemudian mean pada variable ini lebih besar dibadningkan median


```python
import pandas as pd
databaru=pd.read_csv("C:/Users/User/Documents/CompanyABCProfit.csv")
databaru.columns=['Year','Profit (in 000)']
databaru.head()
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
      <th>Year</th>
      <th>Profit (in 000)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1821</td>
      <td>1645</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1822</td>
      <td>658</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1823</td>
      <td>1926</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1824</td>
      <td>865</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1825</td>
      <td>764</td>
    </tr>
  </tbody>
</table>
</div>




```python
from matplotlib import pyplot as plt
#membuat histogram dengan jumlah batang 20, alpha tingkat transparansi
hist=plt.hist(databaru['Profit (in 000)'],bins=20,linewidth=2,alpha=0.4)
#menonaktifkan border atas sama kiri pada graph
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
#menghitung median mode dan mean
median=databaru['Profit (in 000)'].median().round(2)
mode=databaru['Profit (in 000)'].mode().tolist()
mean=databaru['Profit (in 000)'].mean()
#mengakses nilai batas sumbu y darri grafik
ylim=plt.gca().get_ylim()
#membuat garis vertikal
plt.axvline(median,linestyle='--',color='red')
plt.axvline(mean,linestyle='--',color='blue')
plt.axvline(mode[0],linestyle='--',color='yellow')
#menambah text
plt.text(median-0.5,ylim[1]+0.7,"Median",color='red',fontsize=8)
plt.text(mode[0]-2,ylim[1]+0.7,"Mode",color='yellow',fontsize=8)
plt.text(mean-0.5,ylim[1]+0.7,"Mean",color='blue',fontsize=8)
#skewnes
skew_profit=databaru['Profit (in 000)'].skew().round(2)
plt.text(20,ylim[1]-10,f"skewness of \nthis shape : \n\n{skew_profit}")
```




    Text(20, 19.4, 'skewness of \nthis shape : \n\n-0.13')




    
![png]({{ site.baseurl }}/assets/image-desc/data_descriptive_41_1.png)
    


    Ini adalah salah satu contoh dataset yang memiliki distribusi normal, dengan nilai median dan mean hampir sama dan bentuknya simetric. Nilai skewness semakin mendekati 0 semakin normal distribusinya


```python
import pandas as pd
databaru_1=pd.read_csv("C:/Users/User/Documents/housing.csv")
databaru_1.head()
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
      <th>RM</th>
      <th>LSTAT</th>
      <th>PTRATIO</th>
      <th>MEDV</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6.575</td>
      <td>4.98</td>
      <td>15.3</td>
      <td>504000.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>6.421</td>
      <td>9.14</td>
      <td>17.8</td>
      <td>453600.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7.185</td>
      <td>4.03</td>
      <td>17.8</td>
      <td>728700.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6.998</td>
      <td>2.94</td>
      <td>18.7</td>
      <td>701400.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7.147</td>
      <td>5.33</td>
      <td>18.7</td>
      <td>760200.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
from matplotlib import pyplot as plt
#membuat histogram dengan jumlah batang 20, alpha tingkat transparansi
hist=plt.hist(databaru_1['PTRATIO'],bins=20,linewidth=2,alpha=0.4)
#menonaktifkan border atas sama kiri pada graph
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
#menghitung median mode dan mean
median=databaru_1['PTRATIO'].median().round(2)
mode=databaru_1['PTRATIO'].mode().tolist()
mean=databaru_1['PTRATIO'].mean()
#mengakses nilai batas sumbu y darri grafik
ylim=plt.gca().get_ylim()
#membuat garis vertikal
plt.axvline(median,linestyle='--',color='red')
plt.axvline(mean,linestyle='--',color='blue')
plt.axvline(mode[0],linestyle='--',color='yellow')
#menambah text
plt.text(median-0.5,ylim[1]+0.7,"Median",color='red',fontsize=8)
plt.text(mode[0]-2,ylim[1]+0.7,"Mode",color='yellow',fontsize=8)
plt.text(mean-0.5,ylim[1]+0.7,"Mean",color='blue',fontsize=8)
#skewnes
skew_profit=databaru_1['PTRATIO'].skew().round(2)
plt.text(14,ylim[1]-30,f"skewness of \nthis shape : \n\n{skew_profit}")
```




    Text(14, 110.69999999999999, 'skewness of \nthis shape : \n\n-0.82')




    
![png]({{ site.baseurl }}/assets/image-desc/data_descriptive_44_1.png)
    


    Dari grafik diatas merupakan bentuk left tail (negative skewed), karena sebagian besar berkumpulnya data pada bagian kanan grafik dan ekornya terdapat pada left kiri bagian grafik sehingga disebut left tail. Nilai skewnes nya negative semakin besar nilai negative maka semakin negative skewed. Nilai median disini lebih besar dengan nilai mean nya

2. Kurtosis

- Mengukur tinggi dari distrbusi data

![kurts.JPG]({{ site.baseurl }}/assets/image-desc/kurts.JPG)


```python
#data variabel age cardiogoodfitness
kurt_age=data['Age'].kurt()
#convert real kurtosis
print(kurt_age+3)
```

    3.4097099568364437
    

    Karena kurtosis lebih  dari 3 maka termasuk platykurtic dengan tidak adanya outlier dan peak (puncak histogram) yang flat,semakin mendekati 3 datanya semakin terdistribusi normal.
    - platikurtic kurtosis lebih dari 3
    - mesokurtic kurtosis sama dengan 3
    - leptokurtic kurtosis kurang dari 3


```python
from matplotlib import pyplot as plt
#membuat histogram dengan jumlah batang 20, alpha tingkat transparansi, 
#extrack informasi counts = tinggi semua batang pada histogram, bins = koorinat sudut2 masing2 batiang
counts,bins,pathces=plt.hist(data['Age'],bins=20,linewidth=2,alpha=0.4)
#membuat densiti line
center_bin=(bins[:-1]+bins[1:])/2
plt.plot(center_bin,counts,linewidth=1,color='green',label='density-line')
#menonaktifkan border atas sama kiri pada graph
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
#menghitung median mode dan mean
median=data['Age'].median().round(2)
mode=data['Age'].mode().tolist()
mean=data['Age'].mean()
#mengakses nilai batas sumbu y darri grafik
ylim=plt.gca().get_ylim()
#membuat garis vertikal
plt.axvline(median,linestyle='--',color='red')
plt.axvline(mean,linestyle='--',color='blue')
plt.axvline(mode[0],linestyle='--',color='yellow')
#menambah text
plt.text(median-0.5,ylim[1]+0.7,"Median",color='red',fontsize=8)
plt.text(mode[0]-2,ylim[1]+0.7,"Mode",color='yellow',fontsize=8)
plt.text(mean-0.5,ylim[1]+0.7,"Mean",color='blue',fontsize=8)
#skewness and kurtosis
plt.text(45,ylim[1]-10,f"skewness of \nthis shape : \n\n{skew_age.round(2)}")
plt.text(45,ylim[1]-20,f"Kurtosis of \nthis shape : \n\n{kurt_age.round(2)+3}")
plt.legend()
```




    <matplotlib.legend.Legend at 0x20907429850>




    
![png]({{ site.baseurl }}/assets/image-desc/data_descriptive_51_1.png)
    



```python
from matplotlib import pyplot as plt
#membuat histogram dengan jumlah batang 20, alpha tingkat transparansi
counts,bins,patches=plt.hist(databaru['Profit (in 000)'],bins=20,linewidth=2,alpha=0.4)
#membuat densiti line
center_bin=(bins[:-1]+bins[1:])/2
plt.plot(center_bin,counts,linewidth=1,color='green',label='density-line')
#menonaktifkan border atas sama kiri pada graph
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
#menghitung median mode dan mean
median=databaru['Profit (in 000)'].median().round(2)
mode=databaru['Profit (in 000)'].mode().tolist()
mean=databaru['Profit (in 000)'].mean()
#mengakses nilai batas sumbu y darri grafik
ylim=plt.gca().get_ylim()
#membuat garis vertikal
plt.axvline(median,linestyle='--',color='red')
plt.axvline(mean,linestyle='--',color='blue')
plt.axvline(mode[0],linestyle='--',color='yellow')
#menambah text
plt.text(median-0.5,ylim[1]+0.7,"Median",color='red',fontsize=8)
plt.text(mode[0]-2,ylim[1]+0.7,"Mode",color='yellow',fontsize=8)
plt.text(mean-0.5,ylim[1]+0.7,"Mean",color='blue',fontsize=8)
#skewnes
skew_profit=databaru['Profit (in 000)'].skew().round(2)
plt.text(20,ylim[1]-10,f"skewness of \nthis shape : \n\n{skew_profit}")
kurt_prof=databaru['Profit (in 000)'].kurt().round(2) + 3
plt.text(14,ylim[1]-17,f"kurtosis of \nthis shape : \n\n{kurt_prof}")
```




    Text(14, 12.399999999999999, 'kurtosis of \nthis shape : \n\n2.95')




    
![png]({{ site.baseurl }}/assets/image-desc/data_descriptive_52_1.png)
    


    Dilihat dari table diatas kurtosis mendekati 3 artinya terdistribusi normal dan 

## data describe using pandas

    Pandas menyediakan kita dalapm mengakses data deskripsinya, contoh"


```python
import pandas as pd
data=pd.read_csv("C:/Users/User/Documents/CardioGoodFitness.csv")
data['Age'].describe()
```




    count    180.000000
    mean      28.788889
    std        6.943498
    min       18.000000
    25%       24.000000
    50%       26.000000
    75%       33.000000
    max       50.000000
    Name: Age, dtype: float64



atau anda bisa mengakses semua variabel


```python
data.describe()
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
      <th>Age</th>
      <th>Education</th>
      <th>Usage</th>
      <th>Fitness</th>
      <th>Income</th>
      <th>Miles</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>180.000000</td>
      <td>180.000000</td>
      <td>180.000000</td>
      <td>180.000000</td>
      <td>180.000000</td>
      <td>180.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>28.788889</td>
      <td>15.572222</td>
      <td>3.455556</td>
      <td>3.311111</td>
      <td>53719.577778</td>
      <td>103.194444</td>
    </tr>
    <tr>
      <th>std</th>
      <td>6.943498</td>
      <td>1.617055</td>
      <td>1.084797</td>
      <td>0.958869</td>
      <td>16506.684226</td>
      <td>51.863605</td>
    </tr>
    <tr>
      <th>min</th>
      <td>18.000000</td>
      <td>12.000000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>29562.000000</td>
      <td>21.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>24.000000</td>
      <td>14.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>44058.750000</td>
      <td>66.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>26.000000</td>
      <td>16.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>50596.500000</td>
      <td>94.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>33.000000</td>
      <td>16.000000</td>
      <td>4.000000</td>
      <td>4.000000</td>
      <td>58668.000000</td>
      <td>114.750000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>50.000000</td>
      <td>21.000000</td>
      <td>7.000000</td>
      <td>5.000000</td>
      <td>104581.000000</td>
      <td>360.000000</td>
    </tr>
  </tbody>
</table>
</div>



# BIVARIATE ANALYSIS

    Kita sebelumnya sedang mengakses data deskripsi pada masing-masing variabel. Bagaimana kita bisa menggunakan 2 variabel dalam analisis, Analisis yang menggunakan 2 variabel disebut juga bivariate analysis. Biasanya digunakan untuk mengetahu hubungan antara kedua variable yang diukur, grafik yang sering dipakai adalah scatter diagram. Kedua variabel harus bertype number


```python
import pandas as pd
datak=pd.read_csv("C:/Users/User/Documents/Toyota.csv")
datak.tail()
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
      <th>Unnamed: 0</th>
      <th>Price</th>
      <th>Age</th>
      <th>KM</th>
      <th>FuelType</th>
      <th>HP</th>
      <th>MetColor</th>
      <th>Automatic</th>
      <th>CC</th>
      <th>Doors</th>
      <th>Weight</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1431</th>
      <td>1431</td>
      <td>7500</td>
      <td>NaN</td>
      <td>20544</td>
      <td>Petrol</td>
      <td>86</td>
      <td>1.0</td>
      <td>0</td>
      <td>1300</td>
      <td>3</td>
      <td>1025</td>
    </tr>
    <tr>
      <th>1432</th>
      <td>1432</td>
      <td>10845</td>
      <td>72.0</td>
      <td>??</td>
      <td>Petrol</td>
      <td>86</td>
      <td>0.0</td>
      <td>0</td>
      <td>1300</td>
      <td>3</td>
      <td>1015</td>
    </tr>
    <tr>
      <th>1433</th>
      <td>1433</td>
      <td>8500</td>
      <td>NaN</td>
      <td>17016</td>
      <td>Petrol</td>
      <td>86</td>
      <td>0.0</td>
      <td>0</td>
      <td>1300</td>
      <td>3</td>
      <td>1015</td>
    </tr>
    <tr>
      <th>1434</th>
      <td>1434</td>
      <td>7250</td>
      <td>70.0</td>
      <td>??</td>
      <td>NaN</td>
      <td>86</td>
      <td>1.0</td>
      <td>0</td>
      <td>1300</td>
      <td>3</td>
      <td>1015</td>
    </tr>
    <tr>
      <th>1435</th>
      <td>1435</td>
      <td>6950</td>
      <td>76.0</td>
      <td>1</td>
      <td>Petrol</td>
      <td>110</td>
      <td>0.0</td>
      <td>0</td>
      <td>1600</td>
      <td>5</td>
      <td>1114</td>
    </tr>
  </tbody>
</table>
</div>



    Kita ingin mengetahui hubungan umur vs harga pada dataset ini


```python
#ambil data
umur=datak['Age']
harga=datak['Price']
#membuat scatter plot
from matplotlib import pyplot as plt
#varibel x untuk umur, variabel y untuk harga
plt.scatter(umur,harga)
plt.xlabel('Umur')
plt.ylabel('harga')
plt.title('Scatter plot Harga vs Umur')
```




    Text(0.5, 1.0, 'Scatter plot Harga vs Umur')




    
![png]({{ site.baseurl }}/assets/image-desc/data_descriptive_63_1.png)
    


    dari grafik, hubungan antara umur dan harga adalah korelasi negatif yang berarti semakin lama/tua umur mobil maka harga mobil semakin turu, itu disebut juga negatice korelasi yang ditandai dengan menurunnya grafik data point kemudian dalam scatter plot terdapat 3 hubungan anatara 2 variabel:
    1. positive korelasi (grafik dari rendah ke tinggi ) nilai korelasinya juga positive, contoh kasus produksi vs permintaan, semakin tinggi permintaan barang maka jumlah intesitas produkis semakin besar
    2. negative korealasi, dari tinggi ke rendah nilai korealasinya negatfi, contoh harga mobil dengan usia
    3. neuutral, tidak ada hubungan dan pada grafik tidak membentu trend/pola
