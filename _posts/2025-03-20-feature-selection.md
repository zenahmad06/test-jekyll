<h1>FEATURE SELECTION</h1>
<p "text-indent: 10cm"> Proses memilih fitur/variabel yang penting saja yang digunakan untuk meningkatkan performa model dan mengurangi overfitting, karena pastinya dalam sebuah dataset pastinya ada beberapa variabel yang tidak penting digunakan atau ada 2 variabel dengan karakteristi sama harus kita lakukan feature selection</p>

contoh data


```python
kum={
    'hours_studied':[5,2,10,1],
    'sleep_hours':[6,8,5,9],
    'favorite_color':['blue','red','green','yellow'],
    'sosmed_hour':[2,5,1,6],
    'pass/fail':['pass','fail','pass','fail']
}
import pandas as pd
df=pd.DataFrame(kum)
df
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
      <th>hours_studied</th>
      <th>sleep_hours</th>
      <th>favorite_color</th>
      <th>sosmed_hour</th>
      <th>pass/fail</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5</td>
      <td>6</td>
      <td>blue</td>
      <td>2</td>
      <td>pass</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>8</td>
      <td>red</td>
      <td>5</td>
      <td>fail</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10</td>
      <td>5</td>
      <td>green</td>
      <td>1</td>
      <td>pass</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>9</td>
      <td>yellow</td>
      <td>6</td>
      <td>fail</td>
    </tr>
  </tbody>
</table>
</div>



<p> Pada data di atas, jika guru ingin membuat model dalam menentukan lulus/tidaknya sisa menggunakan variabel seberti variabel_hours, sleep_hours, favorite_color, sosmed_hour. Kita lihat secara kasat mata/logika variabel favorite_color kelihatanya bukan variabel penting untuk menentukan lulus/tidak, maka variabel yang tidak penting ini dapat dihilangkan, kemudian misal data tersebut memiliki</p>


```python
df['minutes_studied']=[40,20,30,60]
df
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
      <th>hours_studied</th>
      <th>sleep_hours</th>
      <th>favorite_color</th>
      <th>sosmed_hour</th>
      <th>pass/fail</th>
      <th>minutes_studied</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5</td>
      <td>6</td>
      <td>blue</td>
      <td>2</td>
      <td>pass</td>
      <td>40</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>8</td>
      <td>red</td>
      <td>5</td>
      <td>fail</td>
      <td>20</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10</td>
      <td>5</td>
      <td>green</td>
      <td>1</td>
      <td>pass</td>
      <td>30</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>9</td>
      <td>yellow</td>
      <td>6</td>
      <td>fail</td>
      <td>60</td>
    </tr>
  </tbody>
</table>
</div>



<div style="page-break-before: always"></div>


<p>karena minute studied dan hours studied karakteristiknya sama namun unit pengukurannya berbeda maka salah satu harus dihilangkan sehingga data  redundandcy dapat dikurangi. Dari contoh diatas kita bisa mennetukan mana data yang tidak penting dan data redundant dengan manual, namun hal tersebut belum valid sehinggak kita harus membutuhkan <b>uji statistika</b> untuk menentukan feature mana yang akan dikurangi</p>

<h2>Feature Selection berdasarkan data type</h2>


![ppt1.PNG]({{ site.baseurl }}/assets/image-feature/ppt1.PNG)

- Jika variabel input/variabel x bertipe numerikal dan output variabel nya bertipe kategorikal contoh nya tadi guru ingin memprediksi siswa lulus/tidak menggunakan numerikal maka bisa menggunakan Anova dan kendall
- jika numerical untuk kedua variabel x dan y, contoh nya memprediksi harga rumah maka bisa mengguakan Pearson dan Spearman
- jika data inputan bertipe kategorikal ordinal contoh kelas low, medium dan high dan variabel y adalah numerikal maka bisa mengguanakan kendall

<div style="page-break-before: always"></div>


<h2>Feature Selection menggunakan training model machine learning</h2>


![ppt2.PNG]({{ site.baseurl }}/assets/image-feature/ppt2.PNG)

<p>Berdarkan gambar di atas, metode ini dibagi menjadi 3 cara seperti</p>
- Berdasarkan ada atau tidaknya label pada training data
    <li>Supervised learning : dimana model Ml yang memiliki data input variabel dan label output(y). Biasanya menggunakan/menghitung metode perhitungan korealasi</li>
    <li>Unsupervised learning : dimana model hanya memiliki data input variabel X, tidak memilki label output. Karena tidaka ada label output, maka tidak bisa menggunakan metode perhitungan korelasi dan hanya memilih important feature nya dengan memperhatikan pola pada variabel X</li>
    <li>Semisupervised : Campuran, sebagian ada label output sebagian tidak ada labelnya</li>
- Unsupervised, supervised, dan semisupervise merupakan learning model (model yang mempelajari data untuk melakukan suatu tujuan, misal klasifikasi. Dalam hubungan dengan learning model nya metode feature selection dibagu menjadi 3 :
    <li>Filter : mengurangi feature pada model sebelum model learning dilatih, menghilangkan feature yang tidak relevant, redundant, dan mempunyai korelasi yang lemah dengan target variabel. COntoh Anova, chi square</li>
    <li>Wrapper : memilih variabel yang penting dengan melatih model beberapa kali dengan menggunakan kombinasi feature yang berbeda-beda, contoh forward (menambah satu-persatu variabel selama melatih beberapa model) backward(mengurangi satu-persatu selama melatih model)</li>
    <li>Hybrid : gabungan dari metode wrappper dan filter</li>

kali ini kita akan mempelajari beberapa metode feature selection:

<h2>Ablation</h2>

<p>Salah satu metode dalam machine learning dimana kita akan mengurangi atau memodifikasi variabel yang akan digunakan pada model machine learning dengan tujuan untuk mengetahui dampak terhadap performa model, contoh kasus: resep makanan, kita melakukan percobaan beberapa kali menggunakan kombinasi bahan yang berbeda-beda untuk menghasilkan makanan enak</p>

Bagaimana implementasi menggunakan python?


```python
import pandas as pd
data=pd.read_csv('C:/Users/User/Documents/USA_Housing.csv')
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
      <th>Avg. Area Income</th>
      <th>Avg. Area House Age</th>
      <th>Avg. Area Number of Rooms</th>
      <th>Avg. Area Number of Bedrooms</th>
      <th>Area Population</th>
      <th>Price</th>
      <th>Address</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>79545.458574</td>
      <td>5.682861</td>
      <td>7.009188</td>
      <td>4.09</td>
      <td>23086.800503</td>
      <td>1.059034e+06</td>
      <td>208 Michael Ferry Apt. 674\nLaurabury, NE 3701...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>79248.642455</td>
      <td>6.002900</td>
      <td>6.730821</td>
      <td>3.09</td>
      <td>40173.072174</td>
      <td>1.505891e+06</td>
      <td>188 Johnson Views Suite 079\nLake Kathleen, CA...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>61287.067179</td>
      <td>5.865890</td>
      <td>8.512727</td>
      <td>5.13</td>
      <td>36882.159400</td>
      <td>1.058988e+06</td>
      <td>9127 Elizabeth Stravenue\nDanieltown, WI 06482...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>63345.240046</td>
      <td>7.188236</td>
      <td>5.586729</td>
      <td>3.26</td>
      <td>34310.242831</td>
      <td>1.260617e+06</td>
      <td>USS Barnett\nFPO AP 44820</td>
    </tr>
    <tr>
      <th>4</th>
      <td>59982.197226</td>
      <td>5.040555</td>
      <td>7.839388</td>
      <td>4.23</td>
      <td>26354.109472</td>
      <td>6.309435e+05</td>
      <td>USNS Raymond\nFPO AE 09386</td>
    </tr>
  </tbody>
</table>
</div>



<p>Kali ini kita akan mengimplementasikan ablation pada dataset ini, Pada dataset ini kita akan coba membuat model ML dalam memprediksi harga rumah menggunakan beberapa variabel yang bisa digunakan. Karena variabe y yaitu harga rumah merupakan tipe data numerikal maka model yang digunakan adalah regresi (liniear regression). Model dimana variabel y nya bertipe data numerikal</p>


```python
data.iloc[:,:-1].corr()
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
      <th>Avg. Area Income</th>
      <th>Avg. Area House Age</th>
      <th>Avg. Area Number of Rooms</th>
      <th>Avg. Area Number of Bedrooms</th>
      <th>Area Population</th>
      <th>Price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Avg. Area Income</th>
      <td>1.000000</td>
      <td>-0.002007</td>
      <td>-0.011032</td>
      <td>0.019788</td>
      <td>-0.016234</td>
      <td>0.639734</td>
    </tr>
    <tr>
      <th>Avg. Area House Age</th>
      <td>-0.002007</td>
      <td>1.000000</td>
      <td>-0.009428</td>
      <td>0.006149</td>
      <td>-0.018743</td>
      <td>0.452543</td>
    </tr>
    <tr>
      <th>Avg. Area Number of Rooms</th>
      <td>-0.011032</td>
      <td>-0.009428</td>
      <td>1.000000</td>
      <td>0.462695</td>
      <td>0.002040</td>
      <td>0.335664</td>
    </tr>
    <tr>
      <th>Avg. Area Number of Bedrooms</th>
      <td>0.019788</td>
      <td>0.006149</td>
      <td>0.462695</td>
      <td>1.000000</td>
      <td>-0.022168</td>
      <td>0.171071</td>
    </tr>
    <tr>
      <th>Area Population</th>
      <td>-0.016234</td>
      <td>-0.018743</td>
      <td>0.002040</td>
      <td>-0.022168</td>
      <td>1.000000</td>
      <td>0.408556</td>
    </tr>
    <tr>
      <th>Price</th>
      <td>0.639734</td>
      <td>0.452543</td>
      <td>0.335664</td>
      <td>0.171071</td>
      <td>0.408556</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



syarat menggunakan model linear regressi
1. ada hubungan beberapa variabel X dan Y, pada dataset ini korelasi hubungan antara variabel X dan Y lebih 0.3 artinya bisa diterima
2. multicolinearity, hubungan diantara variabel X, pada dataset ini hubungan antara variabel X tidak kuat


```python
#standarization
data_am=data.iloc[:,:-1]
from sklearn.preprocessing import StandardScaler
standarize=StandardScaler()
kolx=[i for i in data_am.columns.tolist() if i!='Price']
data_am.head()
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
      <th>Avg. Area Income</th>
      <th>Avg. Area House Age</th>
      <th>Avg. Area Number of Rooms</th>
      <th>Avg. Area Number of Bedrooms</th>
      <th>Area Population</th>
      <th>Price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>79545.458574</td>
      <td>5.682861</td>
      <td>7.009188</td>
      <td>4.09</td>
      <td>23086.800503</td>
      <td>1.059034e+06</td>
    </tr>
    <tr>
      <th>1</th>
      <td>79248.642455</td>
      <td>6.002900</td>
      <td>6.730821</td>
      <td>3.09</td>
      <td>40173.072174</td>
      <td>1.505891e+06</td>
    </tr>
    <tr>
      <th>2</th>
      <td>61287.067179</td>
      <td>5.865890</td>
      <td>8.512727</td>
      <td>5.13</td>
      <td>36882.159400</td>
      <td>1.058988e+06</td>
    </tr>
    <tr>
      <th>3</th>
      <td>63345.240046</td>
      <td>7.188236</td>
      <td>5.586729</td>
      <td>3.26</td>
      <td>34310.242831</td>
      <td>1.260617e+06</td>
    </tr>
    <tr>
      <th>4</th>
      <td>59982.197226</td>
      <td>5.040555</td>
      <td>7.839388</td>
      <td>4.23</td>
      <td>26354.109472</td>
      <td>6.309435e+05</td>
    </tr>
  </tbody>
</table>
</div>




```python
#train test split
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(data_am[kolx],data_am['Price'],test_size=0.8,random_state=42)
#mode linear regreesion
from sklearn.linear_model import LinearRegression
model=LinearRegression()
stxtrain=standarize.fit_transform(xtrain)
stxtest=standarize.transform(xtest)

model.fit(stxtrain,ytrain)
score=model.score(stxtest,ytest)
print('original dataset: ',score)
```

    original dataset:  0.9158072562815417
    

**abalation**


```python
scoreKum=[]
koldrop=[]
for i in kolx:
    xtrain_n=xtrain.drop(columns=[i])
    xtest_n=xtest.drop(columns=[i])
    model.fit(xtrain_n,ytrain)
    score_col=model.score(xtest_n,ytest)
    scoreKum.append(score_col)
    koldrop.append(i)
hasil=pd.DataFrame()
hasil['we drop']=koldrop
hasil['score']=scoreKum
hasil
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
      <th>we drop</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Avg. Area Income</td>
      <td>0.483626</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Avg. Area House Age</td>
      <td>0.693297</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Avg. Area Number of Rooms</td>
      <td>0.821900</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Avg. Area Number of Bedrooms</td>
      <td>0.916068</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Area Population</td>
      <td>0.728988</td>
    </tr>
  </tbody>
</table>
</div>



Ketika kita meremove Avg. Area Number of Bedroom dari original dataset, model linear regresi scorenya 0.91 paling tinggi diantara lainnya, artinya Avg. Area Number of Bedrooms	 dapat di remove dari original data karena tidak berimpact terhadap performa model

di atas kita mengetahui efek perubahan akurasi dari perubahan variabel yang digunakan, Pertanyaannya bagaimana mengetahui apakah variabel yang kita punya ini penting terhadap variabel y secara signifikan/objektif? kita bisa mengetahui menggunakan <b>t-test</b>

<h3>T-test</h3>
<p>Digunakan untuk melihat apakah variabel independent atau variabel X berguna/penting terhadapa prediksi variabel y, Sebelum menggunakan t-test terdapat asumsi-asumsi yang akan kita cek terlebih dahulu diantaranya </p>

<li> Check Normalitas</li>


```python
xtrain,xtest,ytrain,ytest=train_test_split(data_am[kolx],data_am['Price'],test_size=0.8,random_state=42)
#mode linear regreesion
from sklearn.linear_model import LinearRegression
#model regresi
model=LinearRegression()
#standarization
stxtrain=standarize.fit_transform(xtrain)
stxtest=standarize.transform(xtest)
#model predictio price hose
model.fit(stxtrain,ytrain)
ypred=model.predict(stxtest)
#residual selisih ypred dan ytest
residual=ytest-ypred
residual
```




    1501     29530.533212
    2586     12483.222728
    2653    100769.007966
    1055    196988.114789
    705     -10941.259039
                ...      
    3335   -121450.661799
    1920    -59981.504668
    3715     60952.367631
    4646    -22373.643941
    946      -8730.967667
    Name: Price, Length: 4000, dtype: float64



<p> terdapat 2 cara melihat normalitas</p>
<li> secara visual (histogram/menggunakan qqplot)</li>
<li>secara statistik (metode saphiro-wild)</li>


```python
#visual
from matplotlib import pyplot as plt
import scipy.stats as stats
plt.hist(residual, bins=30)
#saphiro-wilk
stat,p=stats.shapiro(residual)
plt.text(100000,300,f"shapiro-wilk\n p : {round(p,2)} \n(p > 0.05 means normality)",fontsize=10)
```




    Text(100000, 300, 'shapiro-wilk\n p : 0.42 \n(p > 0.05 means normality)')




    
![png]({{ site.baseurl }}/assets/image-feature/feature_selection_32_1.png)
    


<p> dari gambar diatas histogram residual berdistribusi normal karena membentuk bell dan simetrik di kedua sisinyam dan juga nilai p hasil uji shapiro wilk di atas 0.05 artinya data normal. Selanjuntya kita akan melakukan t-test karena LinearRegression tidak menyediakan t-test maka kita akan memakai OLS (Ordinary Least Square)</p>

<div style="page-break-before: always"></div>


sebelumnya cara kerja t-test sebagai berikut:

1. T-test mempunyai 2 hipotesis :

![hipo.PNG]({{ site.baseurl }}/assets/image-feature/hipo.PNG)

<li> Hipotesis 0 : feature tidak ada pengaruh ke variabel Y</li>
<li> Hipotesis 1 : feature ada pengaruh ke variabel Y </li>
<p> jika p < 0.05 maka kita akan menolak hipotesis null yang berarti ada pengaruh ke variabel Y, sedangkan jika p>0.05 maka pengaruh nya ke variabel Y tidak ada sehingga bisa kita hilangkan</p>

2. Rumus t-test

1. \betha adalah koefisien feature dan SE adalah standard error dari koefisien

![TTEST.PNG]({{ site.baseurl }}/assets/image-feature/TTEST.PNG)

3. melakukan T-test dengan python


```python
#train test split
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(data_am[kolx],data_am['Price'],test_size=0.8,random_state=42)
#mode linear regreesion
from sklearn.linear_model import LinearRegression
model=LinearRegression()
stxtrain=standarize.fit_transform(xtrain)
stxtest=standarize.transform(xtest)

#data
train_standar=pd.DataFrame(stxtrain,columns=kolx)
test_standar=pd.DataFrame(stxtest,columns=kolx)
#t-test menggunakan OLS 
import statsmodels.api as sm
#add intecepet in 
train_with_cs=sm.add_constant(train_standar)
ytrain=ytrain.reset_index(drop=True)
#model sm
model_sm=sm.OLS(ytrain,train_with_cs).fit()
model_sm.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>          <td>Price</td>      <th>  R-squared:         </th> <td>   0.925</td> 
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.924</td> 
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   2446.</td> 
</tr>
<tr>
  <th>Date:</th>             <td>Tue, 18 Mar 2025</td> <th>  Prob (F-statistic):</th>  <td>  0.00</td>  
</tr>
<tr>
  <th>Time:</th>                 <td>16:52:01</td>     <th>  Log-Likelihood:    </th> <td> -12930.</td> 
</tr>
<tr>
  <th>No. Observations:</th>      <td>  1000</td>      <th>  AIC:               </th> <td>2.587e+04</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>   994</td>      <th>  BIC:               </th> <td>2.590e+04</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     5</td>      <th>                     </th>     <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
                <td></td>                  <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th>                        <td>  1.23e+06</td> <td> 3164.612</td> <td>  388.609</td> <td> 0.000</td> <td> 1.22e+06</td> <td> 1.24e+06</td>
</tr>
<tr>
  <th>Avg. Area Income</th>             <td> 2.301e+05</td> <td> 3170.065</td> <td>   72.593</td> <td> 0.000</td> <td> 2.24e+05</td> <td> 2.36e+05</td>
</tr>
<tr>
  <th>Avg. Area House Age</th>          <td> 1.625e+05</td> <td> 3169.388</td> <td>   51.273</td> <td> 0.000</td> <td> 1.56e+05</td> <td> 1.69e+05</td>
</tr>
<tr>
  <th>Avg. Area Number of Rooms</th>    <td> 1.198e+05</td> <td> 3510.511</td> <td>   34.121</td> <td> 0.000</td> <td> 1.13e+05</td> <td> 1.27e+05</td>
</tr>
<tr>
  <th>Avg. Area Number of Bedrooms</th> <td> 6986.7688</td> <td> 3511.689</td> <td>    1.990</td> <td> 0.047</td> <td>   95.595</td> <td> 1.39e+04</td>
</tr>
<tr>
  <th>Area Population</th>              <td> 1.517e+05</td> <td> 3166.882</td> <td>   47.889</td> <td> 0.000</td> <td> 1.45e+05</td> <td> 1.58e+05</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td> 4.880</td> <th>  Durbin-Watson:     </th> <td>   1.961</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.087</td> <th>  Jarque-Bera (JB):  </th> <td>   3.807</td>
</tr>
<tr>
  <th>Skew:</th>          <td>-0.000</td> <th>  Prob(JB):          </th> <td>   0.149</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 2.698</td> <th>  Cond. No.          </th> <td>    1.59</td>
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



<p> dari gambar di atas kita lihat kolom P>|t| dimana pada variabel Avg. Area Number of Bedrooms memiliki 0.047 walaupun kurang dari 0.05 dimana menolak hipotesis null, tapi kita bisa mempertimbangkan untuk mengurangi fitur ini karena nilainya mendekati 0.05 dan di antara semua variabel, variabel inilah yang memiliki nilai paling kecil, kemudian kita hilangkan variabel bedroom dan jalankan lagi modelnya</p>


```python
#train test split
data_new=data_am.copy()
data_new=data_new.drop(columns=['Avg. Area Number of Bedrooms'])
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(
    data_am[[i for i in data_new.columns.tolist() if i!='Price']],data_new['Price'],test_size=0.8,random_state=42)
#mode linear regreesion
from sklearn.linear_model import LinearRegression
model=LinearRegression()
stxtrain=standarize.fit_transform(xtrain)
stxtest=standarize.transform(xtest)

#data
train_standar=pd.DataFrame(stxtrain,columns=[i for i in data_new.columns.tolist() if i!='Price'])
test_standar=pd.DataFrame(stxtest,columns=[i for i in data_new.columns.tolist() if i!='Price'])
#t-test menggunakan OLS 
import statsmodels.api as sm
#add intecepet in 
train_with_cs=sm.add_constant(train_standar)
ytrain=ytrain.reset_index(drop=True)
#model sm
model_sm=sm.OLS(ytrain,train_with_cs).fit()
model_sm.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>          <td>Price</td>      <th>  R-squared:         </th> <td>   0.925</td> 
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.924</td> 
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   3047.</td> 
</tr>
<tr>
  <th>Date:</th>             <td>Tue, 18 Mar 2025</td> <th>  Prob (F-statistic):</th>  <td>  0.00</td>  
</tr>
<tr>
  <th>Time:</th>                 <td>16:52:01</td>     <th>  Log-Likelihood:    </th> <td> -12932.</td> 
</tr>
<tr>
  <th>No. Observations:</th>      <td>  1000</td>      <th>  AIC:               </th> <td>2.587e+04</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>   995</td>      <th>  BIC:               </th> <td>2.590e+04</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     4</td>      <th>                     </th>     <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
              <td></td>                 <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th>                     <td>  1.23e+06</td> <td> 3169.313</td> <td>  388.033</td> <td> 0.000</td> <td> 1.22e+06</td> <td> 1.24e+06</td>
</tr>
<tr>
  <th>Avg. Area Income</th>          <td> 2.303e+05</td> <td> 3173.195</td> <td>   72.585</td> <td> 0.000</td> <td> 2.24e+05</td> <td> 2.37e+05</td>
</tr>
<tr>
  <th>Avg. Area House Age</th>       <td> 1.627e+05</td> <td> 3173.183</td> <td>   51.260</td> <td> 0.000</td> <td> 1.56e+05</td> <td> 1.69e+05</td>
</tr>
<tr>
  <th>Avg. Area Number of Rooms</th> <td> 1.228e+05</td> <td> 3172.488</td> <td>   38.706</td> <td> 0.000</td> <td> 1.17e+05</td> <td> 1.29e+05</td>
</tr>
<tr>
  <th>Area Population</th>           <td> 1.518e+05</td> <td> 3170.451</td> <td>   47.888</td> <td> 0.000</td> <td> 1.46e+05</td> <td> 1.58e+05</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td> 3.519</td> <th>  Durbin-Watson:     </th> <td>   1.949</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.172</td> <th>  Jarque-Bera (JB):  </th> <td>   2.927</td>
</tr>
<tr>
  <th>Skew:</th>          <td>-0.009</td> <th>  Prob(JB):          </th> <td>   0.231</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 2.736</td> <th>  Cond. No.          </th> <td>    1.06</td>
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



Bandingkan dengan R square sebelum dihilangkan dan sesudah dihilangkan sama nilainya 0.925 tidak berubah artinya variabel bedroom tidak berimpact kepada variabel target

<h3>Boopstrap Sample</h3>
<p> Kalau tadi kan kita menggunakan dataset 1 kali saja untuk menguji performa model dan memilih variabel yang penting untuk memprediksi variabel target. Bagaimana jika menggunakan berbagai macam dataset yang berbeda-beda. Ini disebut <b> Boopstrap sample</b>. Boopstrap sample dilakukan dengan membuat dataset yang banyak yang mana anggota-anggota dari dataset itu dipilih secara random. Keuntungan menggunakan lebih dari 1 dataset dalam mengukur model adalah model dapat diuji pada kondisi yang berbeda-beda sehingga nantinya menghasilkan model yang akurat</p>


```python
import numpy as np
import pandas as pd
import pandas as pd
data=pd.read_csv('C:/Users/User/Documents/USA_Housing.csv')
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
      <th>Avg. Area Income</th>
      <th>Avg. Area House Age</th>
      <th>Avg. Area Number of Rooms</th>
      <th>Avg. Area Number of Bedrooms</th>
      <th>Area Population</th>
      <th>Price</th>
      <th>Address</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>79545.458574</td>
      <td>5.682861</td>
      <td>7.009188</td>
      <td>4.09</td>
      <td>23086.800503</td>
      <td>1.059034e+06</td>
      <td>208 Michael Ferry Apt. 674\nLaurabury, NE 3701...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>79248.642455</td>
      <td>6.002900</td>
      <td>6.730821</td>
      <td>3.09</td>
      <td>40173.072174</td>
      <td>1.505891e+06</td>
      <td>188 Johnson Views Suite 079\nLake Kathleen, CA...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>61287.067179</td>
      <td>5.865890</td>
      <td>8.512727</td>
      <td>5.13</td>
      <td>36882.159400</td>
      <td>1.058988e+06</td>
      <td>9127 Elizabeth Stravenue\nDanieltown, WI 06482...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>63345.240046</td>
      <td>7.188236</td>
      <td>5.586729</td>
      <td>3.26</td>
      <td>34310.242831</td>
      <td>1.260617e+06</td>
      <td>USS Barnett\nFPO AP 44820</td>
    </tr>
    <tr>
      <th>4</th>
      <td>59982.197226</td>
      <td>5.040555</td>
      <td>7.839388</td>
      <td>4.23</td>
      <td>26354.109472</td>
      <td>6.309435e+05</td>
      <td>USNS Raymond\nFPO AE 09386</td>
    </tr>
  </tbody>
</table>
</div>




```python
data_uji=data.drop(columns=['Address'])
#create boopstrap sample
num_boopstra=10
length=len(data_uji)
kol_x=[i for i in data_uji.columns.tolist() if i!='Price']
y=data_uji['Price']
X=data_uji[kol_x]
liskum=[]
for i in range(num_boopstra):
    pilih=np.random.choice(range(length),size=length,replace=True)
    X_data=X.iloc[pilih]
    y_data=y.iloc[pilih]
    datakum=pd.concat([X_data,y_data],axis=1,ignore_index=True)
    liskum.append(datakum)
liskum[0]
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>268</th>
      <td>89868.891927</td>
      <td>4.504275</td>
      <td>7.234480</td>
      <td>5.45</td>
      <td>30101.724574</td>
      <td>1.421715e+06</td>
    </tr>
    <tr>
      <th>2069</th>
      <td>73984.739147</td>
      <td>6.971445</td>
      <td>8.203237</td>
      <td>5.40</td>
      <td>36773.035408</td>
      <td>1.708128e+06</td>
    </tr>
    <tr>
      <th>1980</th>
      <td>79647.165240</td>
      <td>5.288196</td>
      <td>7.039477</td>
      <td>3.32</td>
      <td>48360.694364</td>
      <td>1.491812e+06</td>
    </tr>
    <tr>
      <th>1826</th>
      <td>69320.329120</td>
      <td>4.312328</td>
      <td>5.904928</td>
      <td>3.39</td>
      <td>32668.934922</td>
      <td>9.023504e+05</td>
    </tr>
    <tr>
      <th>2619</th>
      <td>65197.995424</td>
      <td>6.810647</td>
      <td>5.617602</td>
      <td>4.18</td>
      <td>28162.440645</td>
      <td>9.232469e+05</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1642</th>
      <td>63686.259892</td>
      <td>5.938428</td>
      <td>8.763599</td>
      <td>3.49</td>
      <td>18163.773617</td>
      <td>1.042144e+06</td>
    </tr>
    <tr>
      <th>311</th>
      <td>76209.995308</td>
      <td>7.009046</td>
      <td>5.464339</td>
      <td>4.22</td>
      <td>22721.649399</td>
      <td>1.066659e+06</td>
    </tr>
    <tr>
      <th>1313</th>
      <td>60960.463540</td>
      <td>3.696844</td>
      <td>7.768934</td>
      <td>3.40</td>
      <td>28171.985010</td>
      <td>5.840612e+05</td>
    </tr>
    <tr>
      <th>3523</th>
      <td>88691.128866</td>
      <td>8.023801</td>
      <td>8.417323</td>
      <td>3.22</td>
      <td>29837.279499</td>
      <td>2.096978e+06</td>
    </tr>
    <tr>
      <th>4087</th>
      <td>100741.298585</td>
      <td>5.870726</td>
      <td>6.644853</td>
      <td>4.33</td>
      <td>26041.487616</td>
      <td>1.644923e+06</td>
    </tr>
  </tbody>
</table>
<p>5000 rows × 6 columns</p>
</div>



<div style="text-align: justify;">Kita lihat pada index nya dataset diatas berbeda-beda karena menggunakan metode random dalam memilih index yang akan dijadikan dataset, kemudian bagaimana kita akan melakukan ablation? contoh kasus regresi dan kita akan menggunakan MSE/RMSE sebagai alat ukur mengukur model performance, MSE umum digunakan untuk metric kasus regresi dan karena kita secara manual mengawasi model performance, penggunaan MSE juga agar perhitungan model performance lebih jelas.</div>


```python
data_uji=data.drop(columns=['Address'])
def boopstrap_res(x,y,n):

    num_boopstra=n
    length=len(x)
    kol_x=x.columns.tolist()
    
    #train test split
    mse=[]
    for i in range(n):
        pilih=np.random.choice(range(length),size=length,replace=True)
        X_data=x.iloc[pilih]
        y_data=y.iloc[pilih]
        #split
        xtrain,xtest,ytrain,ytest=train_test_split(X_data,y_data,test_size=0.2,random_state=42)
        #standarization
        standar=StandardScaler()
        from sklearn.linear_model import LinearRegression
        model=LinearRegression()
        stxtrain=standar.fit_transform(xtrain)
        stxtest=standar.transform(xtest)

        #data
        train_standar=pd.DataFrame(stxtrain,columns=kol_x)
        test_standar=pd.DataFrame(stxtest,columns=kol_x)
        #linear regression
        model.fit(train_standar,ytrain)
        #predic
        ypred=model.predict(test_standar)
        from sklearn.metrics import mean_squared_error
        mse_val=np.sqrt(mean_squared_error(ytest,ypred))
        mse.append(mse_val)
    return mse

#original data boopstrap
XX=data_uji[[i for i in data_uji.columns.tolist() if i != 'Price']]
yy=data_uji['Price']
original_res=boopstrap_res(XX,yy,10)
    
original_res
```




    [np.float64(103528.31244601005),
     np.float64(98917.70842122546),
     np.float64(102033.87405877774),
     np.float64(98727.00010332931),
     np.float64(104134.98994299388),
     np.float64(100002.3468781471),
     np.float64(102446.2406592085),
     np.float64(101890.68722535613),
     np.float64(103437.85395848981),
     np.float64(99009.23592476359)]



<div> kenapa kita menggunakan semua perhitungan RMSE (tidak menggunakan mean)? karena nantinya kita akan digunakan dalam mengetahui performa model menggunakan Confidence interval, Selain itu juga kita bisa mengethaui performa model berdasarkan setiap boopstrapnya, selanjutnya kita akan melakukan ablation</div>.


```python
data_uji=data.drop(columns=['Address'])
def boopstrap_res(x,y,n):

    num_boopstra=n
    length=len(x)
    kol_x=x.columns.tolist()
    
    #train test split
    mse=[]
    for i in range(n):
        pilih=np.random.choice(range(length),size=length,replace=True)
        X_data=x.iloc[pilih]
        y_data=y.iloc[pilih]
        #split
        xtrain,xtest,ytrain,ytest=train_test_split(X_data,y_data,test_size=0.2,random_state=42)
        #standarization
        standar=StandardScaler()
        from sklearn.linear_model import LinearRegression
        model=LinearRegression()
        stxtrain=standar.fit_transform(xtrain)
        stxtest=standar.transform(xtest)

        #data
        train_standar=pd.DataFrame(stxtrain,columns=kol_x)
        test_standar=pd.DataFrame(stxtest,columns=kol_x)
        #linear regression
        model.fit(train_standar,ytrain)
        #predic
        ypred=model.predict(test_standar)
        from sklearn.metrics import mean_squared_error
        mse_val=np.sqrt(mean_squared_error(ytest,ypred))
        mse.append(mse_val)
    return mse

#original data boopstrap
XX=data_uji[[i for i in data_uji.columns.tolist() if i != 'Price']]
yy=data_uji['Price']
original_res=boopstrap_res(XX,yy,10)

ablated_res={}
for i in kol_x:
    datab=XX.drop(columns=[i])
    res=boopstrap_res(datab,yy,10)
    ablated_res[f"drop {i}"]=res
df_res=pd.DataFrame(ablated_res)
import  re
df_result=df_res.rename(index=lambda x: re.sub(r"^(\d)",r"boopstrap ke \1", str(x)))
df_result['ori']=original_res
df_result
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
      <th>drop Avg. Area Income</th>
      <th>drop Avg. Area House Age</th>
      <th>drop Avg. Area Number of Rooms</th>
      <th>drop Avg. Area Number of Bedrooms</th>
      <th>drop Area Population</th>
      <th>ori</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>boopstrap ke 0</th>
      <td>253328.241356</td>
      <td>194374.229469</td>
      <td>145022.164816</td>
      <td>102169.642478</td>
      <td>182362.331908</td>
      <td>100337.029065</td>
    </tr>
    <tr>
      <th>boopstrap ke 1</th>
      <td>245336.067827</td>
      <td>191843.693797</td>
      <td>146094.677899</td>
      <td>99825.309857</td>
      <td>181755.286134</td>
      <td>101511.497975</td>
    </tr>
    <tr>
      <th>boopstrap ke 2</th>
      <td>251132.571036</td>
      <td>191714.868956</td>
      <td>152019.939187</td>
      <td>97168.824663</td>
      <td>181606.464214</td>
      <td>102210.725735</td>
    </tr>
    <tr>
      <th>boopstrap ke 3</th>
      <td>249320.009238</td>
      <td>193699.048409</td>
      <td>153074.019355</td>
      <td>98868.947391</td>
      <td>181733.105189</td>
      <td>100653.607778</td>
    </tr>
    <tr>
      <th>boopstrap ke 4</th>
      <td>249741.540354</td>
      <td>193714.816274</td>
      <td>145832.071784</td>
      <td>98951.833447</td>
      <td>178545.588028</td>
      <td>97834.660658</td>
    </tr>
    <tr>
      <th>boopstrap ke 5</th>
      <td>257016.998318</td>
      <td>194277.988289</td>
      <td>150026.422221</td>
      <td>101320.749123</td>
      <td>181184.442629</td>
      <td>100194.622018</td>
    </tr>
    <tr>
      <th>boopstrap ke 6</th>
      <td>249013.534005</td>
      <td>186072.331863</td>
      <td>147644.781247</td>
      <td>99026.038053</td>
      <td>183734.429089</td>
      <td>98064.750180</td>
    </tr>
    <tr>
      <th>boopstrap ke 7</th>
      <td>256393.153260</td>
      <td>194687.967065</td>
      <td>144057.436578</td>
      <td>96298.375301</td>
      <td>188329.739173</td>
      <td>103210.583305</td>
    </tr>
    <tr>
      <th>boopstrap ke 8</th>
      <td>248246.706322</td>
      <td>193825.218282</td>
      <td>154322.268423</td>
      <td>100506.453141</td>
      <td>179132.113683</td>
      <td>99820.457696</td>
    </tr>
    <tr>
      <th>boopstrap ke 9</th>
      <td>246100.021855</td>
      <td>193760.949606</td>
      <td>144748.597606</td>
      <td>102545.452563</td>
      <td>182491.853168</td>
      <td>99138.972300</td>
    </tr>
  </tbody>
</table>
</div>



<div style="text-align:justify;"> Bandingkan variabel bedrom dengan original tidak jauh berbeda artinya variabel bedroom tidak penting bagi variabel target sehingga bisa dihilangkan. Selanjutnya kita akan mengetahui performa model sebelum dihilangkan variabel bedroom dan sesudah dihilangkan variabel bedroom berdasarkan RMSE diatas menggunakan CI (confidence interval) </div>

<h3>Confidence Interval</h3>

<div> Salah satu metode yang digunakan dengan membuat keyakinan dalam membuat statement, biasanya menggunakan tingkat kepercayaan, pada python biasanya diukur dengan np.percentile</div> 


```python
rmse_ori=df_result['ori']
rmse_dropbedroom=df_result['drop Avg. Area Number of Bedrooms']
import numpy as np
ci_original=np.percentile(rmse_ori,[2.5,97.5])
ci_drop=np.percentile(rmse_dropbedroom,[2.5,97.5])
print(ci_original,ci_drop)
```

    [ 97886.43080008 102985.61535134] [ 96494.22640719 102460.89529417]
    

<div> pada Ci 95% boopstrap sebanyak 10 kali original RMSE berada [ 99014.75629231 105802.07488711] sedangkan untuk variabel setelah drop bedroom 95% boopstrap sebanyak 10 kali original RMSE berada [ 94941.86509793 105930.70361293]. Lower bound pada drop lebih rendah dari original artinya setelah dihilangkan fitur bedroom tidak berimpact pada performa model. CI bisa juga mengukur model stability</div>


```python
ci_width=ci_original[1]-ci_original[0]
stability_ratio=(ci_width/np.mean(rmse_ori))*100
ci_widthdrop=ci_drop[1]-ci_drop[0]
stability_ratiodrop=(ci_widthdrop/np.mean(rmse_dropbedroom))*100
stability_ratio,stability_ratiodrop
```




    (np.float64(5.084049809275423), np.float64(5.986534447136629))



stability ration dari original < 10 % artinya stabil kemudian setelah dihilangkan ratio menjadi 10% moderate stabil. Kenaikan ratio relatif kecil dan model masih stabil (< 15%)

<h2> Mutual Information</h2>

<div> digunakan untuk menentukan variabel mana yang penting dalam memprediksi variabel target, skelarn menyediakan mutual information dalam menentukan variabel mana yang penting dan menghilangkan variabel yang kurang penting</div>


```python
import pandas as pd
data=pd.read_csv('C:/Users/User/Documents/USA_Housing.csv')
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
      <th>Avg. Area Income</th>
      <th>Avg. Area House Age</th>
      <th>Avg. Area Number of Rooms</th>
      <th>Avg. Area Number of Bedrooms</th>
      <th>Area Population</th>
      <th>Price</th>
      <th>Address</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>79545.458574</td>
      <td>5.682861</td>
      <td>7.009188</td>
      <td>4.09</td>
      <td>23086.800503</td>
      <td>1.059034e+06</td>
      <td>208 Michael Ferry Apt. 674\nLaurabury, NE 3701...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>79248.642455</td>
      <td>6.002900</td>
      <td>6.730821</td>
      <td>3.09</td>
      <td>40173.072174</td>
      <td>1.505891e+06</td>
      <td>188 Johnson Views Suite 079\nLake Kathleen, CA...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>61287.067179</td>
      <td>5.865890</td>
      <td>8.512727</td>
      <td>5.13</td>
      <td>36882.159400</td>
      <td>1.058988e+06</td>
      <td>9127 Elizabeth Stravenue\nDanieltown, WI 06482...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>63345.240046</td>
      <td>7.188236</td>
      <td>5.586729</td>
      <td>3.26</td>
      <td>34310.242831</td>
      <td>1.260617e+06</td>
      <td>USS Barnett\nFPO AP 44820</td>
    </tr>
    <tr>
      <th>4</th>
      <td>59982.197226</td>
      <td>5.040555</td>
      <td>7.839388</td>
      <td>4.23</td>
      <td>26354.109472</td>
      <td>6.309435e+05</td>
      <td>USNS Raymond\nFPO AE 09386</td>
    </tr>
  </tbody>
</table>
</div>




```python
data_use=data.drop(columns=['Address'])
from sklearn.feature_selection import mutual_info_regression
Xdat=data_use[[i for i in data_use.columns.tolist() if i!='Price']]
ydat=data_use['Price']
mi_scores=mutual_info_regression(Xdat,ydat)
result=pd.DataFrame({'feature':[i for i in data_use.columns.tolist() if i!='Price'],
                    'mi_scores':mi_scores})
result
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
      <th>feature</th>
      <th>mi_scores</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Avg. Area Income</td>
      <td>0.254113</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Avg. Area House Age</td>
      <td>0.120268</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Avg. Area Number of Rooms</td>
      <td>0.062521</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Avg. Area Number of Bedrooms</td>
      <td>0.020654</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Area Population</td>
      <td>0.089090</td>
    </tr>
  </tbody>
</table>
</div>



<div> variabel bedroom memiliki nilai Mi terendah sehingga variabel ini tidak penting untuk prediksi variabel target</div>

<h2> Chi Squared</h2>

salah satu metode feature slection, dilakukan sebelum model machine learning, bagaimana jika variabel x dan y bertipe kategorikal? anda bisa menggunakan chi squared, contoh:


```python
import pandas as pd
data=pd.read_csv('C:/Users/User/Documents/bike_buyers.csv')
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
      <th>ID</th>
      <th>Marital Status</th>
      <th>Gender</th>
      <th>Income</th>
      <th>Children</th>
      <th>Education</th>
      <th>Occupation</th>
      <th>Home Owner</th>
      <th>Cars</th>
      <th>Commute Distance</th>
      <th>Region</th>
      <th>Age</th>
      <th>Purchased Bike</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>12496</td>
      <td>M</td>
      <td>F</td>
      <td>$40,000.00</td>
      <td>1</td>
      <td>Bachelors</td>
      <td>Skilled Manual</td>
      <td>Yes</td>
      <td>0</td>
      <td>0-1 Miles</td>
      <td>Europe</td>
      <td>42</td>
      <td>No</td>
    </tr>
    <tr>
      <th>1</th>
      <td>24107</td>
      <td>M</td>
      <td>M</td>
      <td>$30,000.00</td>
      <td>3</td>
      <td>Partial College</td>
      <td>Clerical</td>
      <td>Yes</td>
      <td>1</td>
      <td>0-1 Miles</td>
      <td>Europe</td>
      <td>43</td>
      <td>No</td>
    </tr>
    <tr>
      <th>2</th>
      <td>14177</td>
      <td>M</td>
      <td>M</td>
      <td>$80,000.00</td>
      <td>5</td>
      <td>Partial College</td>
      <td>Professional</td>
      <td>No</td>
      <td>2</td>
      <td>2-5 Miles</td>
      <td>Europe</td>
      <td>60</td>
      <td>No</td>
    </tr>
    <tr>
      <th>3</th>
      <td>24381</td>
      <td>S</td>
      <td>M</td>
      <td>$70,000.00</td>
      <td>0</td>
      <td>Bachelors</td>
      <td>Professional</td>
      <td>Yes</td>
      <td>1</td>
      <td>5-10 Miles</td>
      <td>Pacific</td>
      <td>41</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>4</th>
      <td>25597</td>
      <td>S</td>
      <td>M</td>
      <td>$30,000.00</td>
      <td>0</td>
      <td>Bachelors</td>
      <td>Clerical</td>
      <td>No</td>
      <td>0</td>
      <td>0-1 Miles</td>
      <td>Europe</td>
      <td>36</td>
      <td>Yes</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.isnull().sum()
```




    ID                  0
    Marital Status      0
    Gender              0
    Income              0
    Children            0
    Education           0
    Occupation          0
    Home Owner          0
    Cars                0
    Commute Distance    0
    Region              0
    Age                 0
    Purchased Bike      0
    dtype: int64



kita ingin mengetahui apakah variabel kategorikal sepert marital status, punya mobil, gender education, occupation, commute distance, dan age-group dapat memprediksi cariabel target

1. defining the variable


```python
datanew=data.drop(columns=['ID','Income'])
cat_var=datanew.select_dtypes(include=['object']).columns.tolist()
for i in cat_var:
    print(f"variable {i} has unique {datanew[i].unique().tolist()}")
```

    variable Marital Status has unique ['M', 'S']
    variable Gender has unique ['F', 'M']
    variable Education has unique ['Bachelors', 'Partial College', 'High School', 'Partial High School', 'Graduate Degree']
    variable Occupation has unique ['Skilled Manual', 'Clerical', 'Professional', 'Manual', 'Management']
    variable Home Owner has unique ['Yes', 'No']
    variable Commute Distance has unique ['0-1 Miles', '2-5 Miles', '5-10 Miles', '1-2 Miles', '10+ Miles']
    variable Region has unique ['Europe', 'Pacific', 'North America']
    variable Purchased Bike has unique ['No', 'Yes']
    

<p> variable nominal : region dan occupation, variabel ordinal : education dan commute distance, sedangkan variabel biner gender, home owner dan marital status</p>

2. convert age ke age_group dan  have car ke biner


```python
def age_group(x):
    if x>=1 and x<7:
        return 'Baby'
    elif x>=7 and x<=14:
        return 'Children'
    elif x>14 and x<=24:
        return 'Young'
    elif x>25 and x<=54:
        return 'Middle'
    else:
        return 'Senior'
datanew['age_group']=datanew['Age'].apply(age_group)
datanew['have_car']=datanew['Cars'].apply(lambda x: 'yes' if x>0 else 'no')
datanew
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
      <th>Marital Status</th>
      <th>Gender</th>
      <th>Children</th>
      <th>Education</th>
      <th>Occupation</th>
      <th>Home Owner</th>
      <th>Cars</th>
      <th>Commute Distance</th>
      <th>Region</th>
      <th>Age</th>
      <th>Purchased Bike</th>
      <th>age_group</th>
      <th>have_car</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>M</td>
      <td>F</td>
      <td>1</td>
      <td>Bachelors</td>
      <td>Skilled Manual</td>
      <td>Yes</td>
      <td>0</td>
      <td>0-1 Miles</td>
      <td>Europe</td>
      <td>42</td>
      <td>No</td>
      <td>Middle</td>
      <td>no</td>
    </tr>
    <tr>
      <th>1</th>
      <td>M</td>
      <td>M</td>
      <td>3</td>
      <td>Partial College</td>
      <td>Clerical</td>
      <td>Yes</td>
      <td>1</td>
      <td>0-1 Miles</td>
      <td>Europe</td>
      <td>43</td>
      <td>No</td>
      <td>Middle</td>
      <td>yes</td>
    </tr>
    <tr>
      <th>2</th>
      <td>M</td>
      <td>M</td>
      <td>5</td>
      <td>Partial College</td>
      <td>Professional</td>
      <td>No</td>
      <td>2</td>
      <td>2-5 Miles</td>
      <td>Europe</td>
      <td>60</td>
      <td>No</td>
      <td>Senior</td>
      <td>yes</td>
    </tr>
    <tr>
      <th>3</th>
      <td>S</td>
      <td>M</td>
      <td>0</td>
      <td>Bachelors</td>
      <td>Professional</td>
      <td>Yes</td>
      <td>1</td>
      <td>5-10 Miles</td>
      <td>Pacific</td>
      <td>41</td>
      <td>Yes</td>
      <td>Middle</td>
      <td>yes</td>
    </tr>
    <tr>
      <th>4</th>
      <td>S</td>
      <td>M</td>
      <td>0</td>
      <td>Bachelors</td>
      <td>Clerical</td>
      <td>No</td>
      <td>0</td>
      <td>0-1 Miles</td>
      <td>Europe</td>
      <td>36</td>
      <td>Yes</td>
      <td>Middle</td>
      <td>no</td>
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
      <th>1021</th>
      <td>S</td>
      <td>F</td>
      <td>0</td>
      <td>Partial High School</td>
      <td>Manual</td>
      <td>No</td>
      <td>2</td>
      <td>0-1 Miles</td>
      <td>Europe</td>
      <td>32</td>
      <td>Yes</td>
      <td>Middle</td>
      <td>yes</td>
    </tr>
    <tr>
      <th>1022</th>
      <td>M</td>
      <td>F</td>
      <td>2</td>
      <td>Partial College</td>
      <td>Manual</td>
      <td>Yes</td>
      <td>0</td>
      <td>0-1 Miles</td>
      <td>Europe</td>
      <td>63</td>
      <td>No</td>
      <td>Senior</td>
      <td>no</td>
    </tr>
    <tr>
      <th>1023</th>
      <td>M</td>
      <td>M</td>
      <td>0</td>
      <td>Partial College</td>
      <td>Manual</td>
      <td>No</td>
      <td>1</td>
      <td>0-1 Miles</td>
      <td>Pacific</td>
      <td>26</td>
      <td>Yes</td>
      <td>Middle</td>
      <td>yes</td>
    </tr>
    <tr>
      <th>1024</th>
      <td>S</td>
      <td>F</td>
      <td>0</td>
      <td>High School</td>
      <td>Manual</td>
      <td>No</td>
      <td>1</td>
      <td>5-10 Miles</td>
      <td>Europe</td>
      <td>31</td>
      <td>No</td>
      <td>Middle</td>
      <td>yes</td>
    </tr>
    <tr>
      <th>1025</th>
      <td>S</td>
      <td>M</td>
      <td>2</td>
      <td>High School</td>
      <td>Skilled Manual</td>
      <td>No</td>
      <td>2</td>
      <td>1-2 Miles</td>
      <td>Pacific</td>
      <td>50</td>
      <td>Yes</td>
      <td>Middle</td>
      <td>yes</td>
    </tr>
  </tbody>
</table>
<p>1026 rows × 13 columns</p>
</div>




```python
databar=datanew.copy()
databr=databar.drop(columns=['Cars','Age','Children'])
```

3. transformation to numerical

- Biner


```python
databr['Marital Status']=databr['Marital Status'].apply(lambda x:1 if 'M' else 0)
databr['Gender']=databr['Gender'].apply(lambda x:1 if 'F' else 0)
def map_bin(x):
    if x.lower()=='yes':
        return 1
    else:
        return 0
for i in ['Home Owner','Purchased Bike','have_car']:
    databr[i]=databr[i].apply(map_bin)
databr
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
      <th>Marital Status</th>
      <th>Gender</th>
      <th>Education</th>
      <th>Occupation</th>
      <th>Home Owner</th>
      <th>Commute Distance</th>
      <th>Region</th>
      <th>Purchased Bike</th>
      <th>age_group</th>
      <th>have_car</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>Bachelors</td>
      <td>Skilled Manual</td>
      <td>1</td>
      <td>0-1 Miles</td>
      <td>Europe</td>
      <td>0</td>
      <td>Middle</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>Partial College</td>
      <td>Clerical</td>
      <td>1</td>
      <td>0-1 Miles</td>
      <td>Europe</td>
      <td>0</td>
      <td>Middle</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>1</td>
      <td>Partial College</td>
      <td>Professional</td>
      <td>0</td>
      <td>2-5 Miles</td>
      <td>Europe</td>
      <td>0</td>
      <td>Senior</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>Bachelors</td>
      <td>Professional</td>
      <td>1</td>
      <td>5-10 Miles</td>
      <td>Pacific</td>
      <td>1</td>
      <td>Middle</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>1</td>
      <td>Bachelors</td>
      <td>Clerical</td>
      <td>0</td>
      <td>0-1 Miles</td>
      <td>Europe</td>
      <td>1</td>
      <td>Middle</td>
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
    </tr>
    <tr>
      <th>1021</th>
      <td>1</td>
      <td>1</td>
      <td>Partial High School</td>
      <td>Manual</td>
      <td>0</td>
      <td>0-1 Miles</td>
      <td>Europe</td>
      <td>1</td>
      <td>Middle</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1022</th>
      <td>1</td>
      <td>1</td>
      <td>Partial College</td>
      <td>Manual</td>
      <td>1</td>
      <td>0-1 Miles</td>
      <td>Europe</td>
      <td>0</td>
      <td>Senior</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1023</th>
      <td>1</td>
      <td>1</td>
      <td>Partial College</td>
      <td>Manual</td>
      <td>0</td>
      <td>0-1 Miles</td>
      <td>Pacific</td>
      <td>1</td>
      <td>Middle</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1024</th>
      <td>1</td>
      <td>1</td>
      <td>High School</td>
      <td>Manual</td>
      <td>0</td>
      <td>5-10 Miles</td>
      <td>Europe</td>
      <td>0</td>
      <td>Middle</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1025</th>
      <td>1</td>
      <td>1</td>
      <td>High School</td>
      <td>Skilled Manual</td>
      <td>0</td>
      <td>1-2 Miles</td>
      <td>Pacific</td>
      <td>1</td>
      <td>Middle</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>1026 rows × 10 columns</p>
</div>



- nominal


```python
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse_output=False,drop='first')
for i in ['Region','Occupation']:
    
    encodedata=ohe.fit_transform(databr[[i]])
    df_en=pd.DataFrame(encodedata,columns=ohe.get_feature_names_out([i]))
    databr=pd.concat([databr.drop(columns=[i]),df_en],axis=1)

df_n=databr.copy()

```


```python
df_n.head()
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
      <th>Marital Status</th>
      <th>Gender</th>
      <th>Education</th>
      <th>Home Owner</th>
      <th>Commute Distance</th>
      <th>Purchased Bike</th>
      <th>age_group</th>
      <th>have_car</th>
      <th>Region_North America</th>
      <th>Region_Pacific</th>
      <th>Occupation_Management</th>
      <th>Occupation_Manual</th>
      <th>Occupation_Professional</th>
      <th>Occupation_Skilled Manual</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>Bachelors</td>
      <td>1</td>
      <td>0-1 Miles</td>
      <td>0</td>
      <td>Middle</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>Partial College</td>
      <td>1</td>
      <td>0-1 Miles</td>
      <td>0</td>
      <td>Middle</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>1</td>
      <td>Partial College</td>
      <td>0</td>
      <td>2-5 Miles</td>
      <td>0</td>
      <td>Senior</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>Bachelors</td>
      <td>1</td>
      <td>5-10 Miles</td>
      <td>1</td>
      <td>Middle</td>
      <td>1</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>1</td>
      <td>Bachelors</td>
      <td>0</td>
      <td>0-1 Miles</td>
      <td>1</td>
      <td>Middle</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



- Ordinal


```python
edu_order=['Partial High School','High School','Partial College','Bachelors','Graduate Degree']
comut_order=['0-1 Miles','1-2 Miles','2-5 Miles','5-10 Miles','10+ Miles']
age_group=['Baby','Children','Young','Middle','Senior']
order=[edu_order,comut_order,]
for i,j in enumerate(['Education','Commute Distance']):
    val=df_n[j].unique().tolist()
    kum={}
    dat=order[i]
    for k,l in zip(dat,range(len(val))):
        kum[k]=l
    df_n[j]=df_n[j].map(kum)
df_n['age_group']=df_n['age_group'].map({'Middle':0,'Senior':1})
df_n
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
      <th>Marital Status</th>
      <th>Gender</th>
      <th>Education</th>
      <th>Home Owner</th>
      <th>Commute Distance</th>
      <th>Purchased Bike</th>
      <th>age_group</th>
      <th>have_car</th>
      <th>Region_North America</th>
      <th>Region_Pacific</th>
      <th>Occupation_Management</th>
      <th>Occupation_Manual</th>
      <th>Occupation_Professional</th>
      <th>Occupation_Skilled Manual</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
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
      <td>...</td>
    </tr>
    <tr>
      <th>1021</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1022</th>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1023</th>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1024</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1025</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
<p>1026 rows × 14 columns</p>
</div>



4. Applying Chi Squared Test


```python
from sklearn.feature_selection import chi2
x=df_n[[i for i in df_n.columns.tolist() if i !='Purchased Bike']]
y=df_n['Purchased Bike']
chi_score,p_values=chi2(x,y)
df_chi=pd.DataFrame({'feature':x.columns,'chi-score':chi_score,'pval':p_values})
df_chi
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
      <th>feature</th>
      <th>chi-score</th>
      <th>pval</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Marital Status</td>
      <td>0.000000</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Gender</td>
      <td>0.000000</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Education</td>
      <td>10.302686</td>
      <td>1.328367e-03</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Home Owner</td>
      <td>0.104860</td>
      <td>7.460734e-01</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Commute Distance</td>
      <td>30.515928</td>
      <td>3.311366e-08</td>
    </tr>
    <tr>
      <th>5</th>
      <td>age_group</td>
      <td>20.369685</td>
      <td>6.383305e-06</td>
    </tr>
    <tr>
      <th>6</th>
      <td>have_car</td>
      <td>4.812020</td>
      <td>2.826190e-02</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Region_North America</td>
      <td>4.961965</td>
      <td>2.591076e-02</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Region_Pacific</td>
      <td>9.202179</td>
      <td>2.417273e-03</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Occupation_Management</td>
      <td>2.758451</td>
      <td>9.674189e-02</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Occupation_Manual</td>
      <td>0.101783</td>
      <td>7.497004e-01</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Occupation_Professional</td>
      <td>3.180718</td>
      <td>7.451200e-02</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Occupation_Skilled Manual</td>
      <td>0.748224</td>
      <td>3.870390e-01</td>
    </tr>
  </tbody>
</table>
</div>



<div> semakin tinggi nilai chi square maka semakin kuat variabelnya untuk memprediksi variabel target, dari hasil diatas variabel marital status dan gender paling kecil chi score nya dalam memprediksi keputusan pembelian sepeda, sehingga variabel tersebut dapat dihilangkan</div>

<h1> Feature Creation</h1>

<div> Mmebuat feature baru berdasarkan domain knowledge yang kita punya atau dari pattern data yang terbentuk</div>

<h2>Binning</h2>
<p>merubah variabel kontinyu ke dalam variabel kategorikal dengan mengelopokan ke dalam masing-masing kelompok yang telah ditentukan</p>


```python
import pandas as pd
data=pd.read_csv('C:/Users/User/Documents/diabetes.csv')
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
      <th>Pregnancies</th>
      <th>Glucose</th>
      <th>BloodPressure</th>
      <th>SkinThickness</th>
      <th>Insulin</th>
      <th>BMI</th>
      <th>DiabetesPedigreeFunction</th>
      <th>Age</th>
      <th>Outcome</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6</td>
      <td>148</td>
      <td>72</td>
      <td>35</td>
      <td>0</td>
      <td>33.6</td>
      <td>0.627</td>
      <td>50</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>85</td>
      <td>66</td>
      <td>29</td>
      <td>0</td>
      <td>26.6</td>
      <td>0.351</td>
      <td>31</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8</td>
      <td>183</td>
      <td>64</td>
      <td>0</td>
      <td>0</td>
      <td>23.3</td>
      <td>0.672</td>
      <td>32</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>89</td>
      <td>66</td>
      <td>23</td>
      <td>94</td>
      <td>28.1</td>
      <td>0.167</td>
      <td>21</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>137</td>
      <td>40</td>
      <td>35</td>
      <td>168</td>
      <td>43.1</td>
      <td>2.288</td>
      <td>33</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



<div> Kali ini kita akan membinning variabel umur</div>

1. tentukan range umur masing-masing kelompok, 0-10 children, 10-30 young, 30-50 adult, lebih dari 50 tahun senior


```python
range_umur=[0,10,30,50,100]

```

2. Binning dengan menggunakan pandas cut


```python
data['kel_umur']=pd.cut(data['Age'],bins=range_umur,labels=['children','young','adult','senior'])
am=data[['Age','kel_umur']]
am
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
      <th>kel_umur</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>50</td>
      <td>adult</td>
    </tr>
    <tr>
      <th>1</th>
      <td>31</td>
      <td>adult</td>
    </tr>
    <tr>
      <th>2</th>
      <td>32</td>
      <td>adult</td>
    </tr>
    <tr>
      <th>3</th>
      <td>21</td>
      <td>young</td>
    </tr>
    <tr>
      <th>4</th>
      <td>33</td>
      <td>adult</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>763</th>
      <td>63</td>
      <td>senior</td>
    </tr>
    <tr>
      <th>764</th>
      <td>27</td>
      <td>young</td>
    </tr>
    <tr>
      <th>765</th>
      <td>30</td>
      <td>young</td>
    </tr>
    <tr>
      <th>766</th>
      <td>47</td>
      <td>adult</td>
    </tr>
    <tr>
      <th>767</th>
      <td>23</td>
      <td>young</td>
    </tr>
  </tbody>
</table>
<p>768 rows × 2 columns</p>
</div>



3. membuat bar chart


```python
hitung=data['kel_umur'].value_counts()
import numpy as np
from matplotlib import pyplot as plt
xkor=[i for i in np.arange(1,5,1.2)]
plt.bar(xkor,hitung.values)
plt.xticks(xkor,labels=hitung.index)
```




    ([<matplotlib.axis.XTick at 0x17c4c2b5d00>,
      <matplotlib.axis.XTick at 0x17c4c2b5cd0>,
      <matplotlib.axis.XTick at 0x17c4c2b5520>,
      <matplotlib.axis.XTick at 0x17c4c2ee6d0>],
     [Text(1.0, 0, 'young'),
      Text(2.2, 0, 'adult'),
      Text(3.4000000000000004, 0, 'senior'),
      Text(4.6000000000000005, 0, 'children')])




    
![png]({{ site.baseurl }}/assets/image-feature/feature_selection_98_1.png)
    


<h2>One Hot Encoding</h2>
<p> Merubah variabel kateogrikal ke dalam variabel bertipe biner bernilai 0 dan 1, karena machine learning tidak bisa menghandle data text atau string maka harus dirubah terlebih dauhulu. OneHot Encoding biasanya menghandle variabel nominal, jika variabel ordinal menggunakan label encode</p>


```python
import pandas as pd
data = {'color': ['red', 'green', 'blue', 'red', 'green']}
df = pd.DataFrame(data)
encoded_df = pd.get_dummies(df, columns=['color']).astype(int)
print(encoded_df)
```

       color_blue  color_green  color_red
    0           0            0          1
    1           0            1          0
    2           1            0          0
    3           0            0          1
    4           0            1          0
    

color red bernilai 0 0 1, sedangkan green 0 1 0, sedangkan biru 1 0 0, semakin banyak nilai unik pada variabel kategorikal semakin banyak pula (redundant) feature yang dimiliki, biasanya onehotencoding digunakan variabel nominal karena jika kita secara langsung melabeli 0 1 2 pada 1 kolom nominal maka Ml model potensi membaca salah variabel tersebut (ML bacanya nanti ordinal, padahal tipe aslinya nominal)

dalam dummy variabel ini ada jebakan variabel dummy dimana dummy yang kita bentuk saling memiliki hubungan contoh nya pada diatas jika red 0 blue 0 dan kita tahu pastinya green 1. Hubungan antar variabel x ini nantinya juga berpengaruh terhadapa model ML seperti logistic regression dan linear regression. Salah satu solusi yang bisa dilakukan adalah emgnhilangkan 1 feature (drop first) pada dummy


```python
import pandas as pd
data = {'color': ['red', 'green', 'blue', 'red', 'green']}
df = pd.DataFrame(data)
encoded_df = pd.get_dummies(df, columns=['color'], drop_first=True).astype(int)
print(encoded_df)
```

       color_green  color_red
    0            0          1
    1            1          0
    2            0          0
    3            0          1
    4            1          0
    

nilai biru di atas jika nilai red 0 dan nilai green juga 0, selain menggunakan pandas anda dapat menggunakan OneHotEncoding


```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
#sparse False = menghasilkan numpy
encode=OneHotEncoder(sparse_output=False,drop='first')
data = {'color': ['red', 'green', 'blue', 'red', 'green']}
df = pd.DataFrame(data)
encoded_df = encode.fit_transform(df[['color']])
df=pd.DataFrame(encoded_df,columns=encode.get_feature_names_out(['color']))
df
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
      <th>color_green</th>
      <th>color_red</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



<h1>Feature hashing</h1>

Proses merubah variabel kategorikal menjadi variabel number secara cepat dan memakan memori yang sedikit. Contoh mudahnya seperti kita akan mengorganize mainan berbeda ke dalama suatu keranjang, daripada membuat keranjang masing-masing mainan kita hanya membuat beberapa keranjang saja


```python
sentence="The Quick Brown Fox Jumps Over The Lazy Dog"
word=sentence.split()
word
```




    ['The', 'Quick', 'Brown', 'Fox', 'Jumps', 'Over', 'The', 'Lazy', 'Dog']



terdapat 9 kata dalam kalimat ini, dibanding membuat 9 fitur dari kalimat ini kita bisa menggunakan 6 fitur saja untuk menyimpan kata2 ini mengggunakan hashing method


```python
from sklearn.feature_extraction import FeatureHasher
df=pd.DataFrame({'words':word})
df['words']=df['words'].apply(lambda x:[x])
hasher=FeatureHasher(n_features=6,input_type='string')
hasher_res=hasher.transform(df['words'])
df_hasher=pd.DataFrame(hasher_res.toarray())
df_res=pd.concat([df,df_hasher],axis=1)
df_res
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
      <th>words</th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[The]</td>
      <td>0.0</td>
      <td>-1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>[Quick]</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>[Brown]</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>[Fox]</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>[Jumps]</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>[Over]</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>[The]</td>
      <td>0.0</td>
      <td>-1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>[Lazy]</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>[Dog]</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



- kata "the " masuk hash yang sama yaitu 1 
- tabrakan/overlap has pada brown dan lazy , fox dengan over disebut juga hash collision, hash collision umum terjadi jika n feature kurang dari jumlah word yang diuji

di atas kita melakukan hashing pada setiap word, bagaimana menghandle nama kota misalnya "New York" atau kata-kata yang mempunyai makna lebih dari 1 kata. Tidak mungkin kata new york kita pisahkan jadi dua karena "new York sendiri adalah 1 kata penuh makna nama kota. Untuk itu harus mengetahui interaction featurenya sebelum dilakukan hashing
