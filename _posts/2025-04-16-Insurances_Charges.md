---
layout : posts
permalink : /insurances-charges/
---

# Predicting Insurances Charges

How much that people should make a payment to get the facilities like healthy facilities and more facilities


```python
import pandas as pd
data = pd.read_csv('C:/Users/User/Documents/Insurances/insurance.csv')
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
      <th>index</th>
      <th>age</th>
      <th>sex</th>
      <th>bmi</th>
      <th>children</th>
      <th>smoker</th>
      <th>region</th>
      <th>charges</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>19</td>
      <td>female</td>
      <td>27.900</td>
      <td>0</td>
      <td>yes</td>
      <td>southwest</td>
      <td>16884.92400</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>18</td>
      <td>male</td>
      <td>33.770</td>
      <td>1</td>
      <td>no</td>
      <td>southeast</td>
      <td>1725.55230</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>28</td>
      <td>male</td>
      <td>33.000</td>
      <td>3</td>
      <td>no</td>
      <td>southeast</td>
      <td>4449.46200</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>33</td>
      <td>male</td>
      <td>22.705</td>
      <td>0</td>
      <td>no</td>
      <td>northwest</td>
      <td>21984.47061</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>32</td>
      <td>male</td>
      <td>28.880</td>
      <td>0</td>
      <td>no</td>
      <td>northwest</td>
      <td>3866.85520</td>
    </tr>
  </tbody>
</table>
</div>



## Check missing value


```python
data.isnull().sum()
```




    index       0
    age         0
    sex         0
    bmi         0
    children    0
    smoker      0
    region      0
    charges     0
    dtype: int64



there's no missing value in this dataset

## Data Description


```python
print('the lenght of data :', len(data))
print('the variable of this dataset are ',data.columns.tolist())
```

    the lenght of data : 1338
    the variable of this dataset are  ['index', 'age', 'sex', 'bmi', 'children', 'smoker', 'region', 'charges']
    

there's six variable x in predicting insurance charges
1. age : the old people has more risk about healthy, so the payment of insurance higher than young man/woman
2. Sex : there's no evidence that sex influence in insurence charges
3. Children : How many children that we have ? more children that we have, more payment that we should pay for insurance
4. Smoker or not : Smoker person heve higher risk healthy than no smoker so influnce in insurance charges
5. region : there're may be influence in insurance charges or not
6. BMI: body mass index is index to show the people has obesity or not, if BMI high, the risk of healthy is high too so it can influenced the payment of insurance


```python
data['sex'].value_counts()
```




    sex
    male      676
    female    662
    Name: count, dtype: int64



## Explanatory Data Analysis


```python
def pie_chart(x,y):
    tab_count = x.value_counts()
    labels = tab_count.index.tolist()
    values = tab_count.values
    y.pie(values,labels=labels, autopct='%1.1f%%',wedgeprops={'width':0.4,'edgecolor':'white'})
    y.set_title(tab_count.index.name)
    return y
def bar_chart(x,y):
    tab_count = x.value_counts().sort_values()
    labels = [i for i,j in enumerate (tab_count.index.tolist())]
    values = tab_count.values
    y.barh(labels,values,color='lightblue',alpha=0.4)
    y.spines['top'].set_visible(False)
    y.spines['right'].set_visible(False)
    y.spines['bottom'].set_visible(False)
    y.xaxis.set_visible(False)
    y.set_yticks(labels)
    y.set_yticklabels(tab_count.index.tolist())
    for i,j in zip(values,labels):
        y.text(i-((25/100)*i),j,f"{i}",fontsize=8)
    y.set_title(tab_count.index.name)
    return y
def make_boxplot(x,y):
    values = data[x].unique().tolist()
    kum = {}
    for i in values:
        am = data[data[x]==i]['charges']
        kum[i] = am
    df = pd.DataFrame(kum)
    return df.boxplot(vert=False,ax=y)
```


```python
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
#divided 2 
fig,ax =plt.subplots(figsize=(12,7))
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)
whole_container = gridspec.GridSpec(1,2,width_ratios=[0.5,0.5],wspace=0)
#container left
ax_left = fig.add_subplot(whole_container[0])
ax_left.xaxis.set_visible(False)
ax_left.yaxis.set_visible(False)
ax_left.spines['right'].set_visible(False)
#left divide two row
con_left = gridspec.GridSpecFromSubplotSpec(2,1,subplot_spec=ax_left.get_subplotspec(),height_ratios=[0.5,0.5],hspace=0)
ax_left_top = fig.add_subplot(con_left[0])
ax_left_top.xaxis.set_visible(False)
ax_left_top.yaxis.set_visible(False)
plot_sex = ax_left_top.inset_axes([0,0.2,0.6,0.6])
pie_chart(data['sex'],plot_sex)
plot_smoker = ax_left_top.inset_axes([0.4,0.2,0.6,0.6])
pie_chart(data['smoker'],plot_smoker)
#left bottom
ax_left_bottom = fig.add_subplot(con_left[1])
ax_left_bottom .xaxis.set_visible(False)
ax_left_bottom .yaxis.set_visible(False)
plot_region = ax_left_bottom.inset_axes([0.25,0.2,0.3,0.6])
bar_chart(data['region'],plot_region)
plot_chl= ax_left_bottom.inset_axes([0.65,0.2,0.3,0.6])
bar_chart(data['children'],plot_chl)
ax_left_top.spines['right'].set_visible(False)
ax_left_bottom.spines['right'].set_visible(False)


#container right
ax_right = fig.add_subplot(whole_container[1])
ax_right.xaxis.set_visible(False)
ax_right.yaxis.set_visible(False)
#divide by two
con_right = gridspec.GridSpecFromSubplotSpec(2,1,subplot_spec=ax_right.get_subplotspec(),height_ratios=[0.3,0.7],hspace=0)
ax_right_top = fig.add_subplot(con_right[0])
ax_right_top.xaxis.set_visible(False)
ax_right_top.yaxis.set_visible(False)
#plot age and bmi
plot_boxage= ax_right_top.inset_axes([0.1,0.2,0.4,0.5])
plot_boxage.boxplot(data['age'],vert=False);
plot_boxage.set_yticklabels(['Age'])
ax_right_top.yaxis.set_visible(False)
plot_boxbmi= ax_right_top.inset_axes([0.6,0.2,0.35,0.5])
plot_boxbmi.boxplot(data['bmi'],vert=False);
plot_boxbmi.set_yticklabels(['Bmi'])
#plot boxplot categorical vs variabe y
ax_right_bottom = fig.add_subplot(con_right[1])
ax_right_bottom.xaxis.set_visible(False)
ax_right_bottom.yaxis.set_visible(False)
plot_boxsex= ax_right_bottom.inset_axes([0.05,0.6,0.4,0.35])
make_boxplot('sex',plot_boxsex)
plot_boxregion= ax_right_bottom.inset_axes([0.05,0.1,0.4,0.35])
make_boxplot('region',plot_boxregion)
plot_boxsmoker = ax_right_bottom.inset_axes([0.55,0.6,0.4,0.35])
make_boxplot('smoker',plot_boxsmoker)
plot_boxchildren = ax_right_bottom.inset_axes([0.55,0.1,0.4,0.35])
make_boxplot('children',plot_boxchildren)

ax_right_top.spines['left'].set_visible(False)
ax_right_bottom.spines['left'].set_visible(False)
ax_right.spines['left'].set_visible(False)

```


    
![png]({{ site.baseurl }}/assets/image-insurances/Insurances_Charges_12_0.png)
    


- male and female distribution is equal
- No Smoker is higher than people who smoking
- People from southeast is largest in this dataset
- Mosthly in this dataset is sigle, dont have children
- The median of age is 40 years old
-  relationship categorical data vs  target variabel there different charges between No Smoker and Yes Smoker, it means that status of this variable influenced the target variable. People who have smoking habit is higher payment insurance than non smoker people

## Data Preparation for model


```python
from sklearn.preprocessing import StandardScaler
#get dummi region
dumm_region = pd.get_dummies(data['region']).astype(int)
# convert binary to number
data['sex'] = data['sex'].apply(lambda x : 1 if x=='female' else 0)
data['smoker'] = data['smoker'].apply(lambda x: 1 if x=='yes' else 0)
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
      <th>index</th>
      <th>age</th>
      <th>sex</th>
      <th>bmi</th>
      <th>children</th>
      <th>smoker</th>
      <th>region</th>
      <th>charges</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>19</td>
      <td>1</td>
      <td>27.900</td>
      <td>0</td>
      <td>1</td>
      <td>southwest</td>
      <td>16884.92400</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>18</td>
      <td>0</td>
      <td>33.770</td>
      <td>1</td>
      <td>0</td>
      <td>southeast</td>
      <td>1725.55230</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>28</td>
      <td>0</td>
      <td>33.000</td>
      <td>3</td>
      <td>0</td>
      <td>southeast</td>
      <td>4449.46200</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>33</td>
      <td>0</td>
      <td>22.705</td>
      <td>0</td>
      <td>0</td>
      <td>northwest</td>
      <td>21984.47061</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>32</td>
      <td>0</td>
      <td>28.880</td>
      <td>0</td>
      <td>0</td>
      <td>northwest</td>
      <td>3866.85520</td>
    </tr>
  </tbody>
</table>
</div>




```python
from sklearn.model_selection import train_test_split
#variabel x and y
x = data[[i for i in data.columns.tolist() if i!='charges']]
y = data['charges']
#split dataset
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.2,random_state=42)
    
#dummy variabel region in train and test
dummy_train = pd.get_dummies(xtrain['region']).astype(int)
xtrain_new = xtrain.drop(['region','index'],axis=1)
new_train = pd.concat([xtrain_new,dummy_train],axis=1)
#for test 
dummy_test = pd.get_dummies(xtest['region']).astype(int)
xtest_new = xtest.drop(['region','index'],axis=1)
new_test = pd.concat([xtest_new,dummy_test],axis=1)

#standaraisasi numerical variabel for age and BMI
from sklearn.preprocessing import StandardScaler
standarizer = StandardScaler()
new_train['age'] = standarizer.fit_transform(new_train[['age']])
new_test['age'] = standarizer.fit_transform(new_test[['age']])
#BMI
new_train['bmi'] = standarizer.fit_transform(new_train[['bmi']])
new_test['bmi'] = standarizer.fit_transform(new_test[['bmi']])
new_test
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
      <th>age</th>
      <th>sex</th>
      <th>bmi</th>
      <th>children</th>
      <th>smoker</th>
      <th>northeast</th>
      <th>northwest</th>
      <th>southeast</th>
      <th>southwest</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>764</th>
      <td>0.458596</td>
      <td>1</td>
      <td>-0.937152</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>887</th>
      <td>-0.187133</td>
      <td>1</td>
      <td>-0.167527</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>890</th>
      <td>1.821801</td>
      <td>1</td>
      <td>-0.665519</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1293</th>
      <td>0.530343</td>
      <td>0</td>
      <td>-0.846608</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>259</th>
      <td>-1.406842</td>
      <td>0</td>
      <td>0.134287</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
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
    </tr>
    <tr>
      <th>109</th>
      <td>1.750053</td>
      <td>0</td>
      <td>0.637839</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>575</th>
      <td>1.391315</td>
      <td>1</td>
      <td>-0.620247</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>535</th>
      <td>-0.043638</td>
      <td>0</td>
      <td>-0.484431</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>543</th>
      <td>1.104325</td>
      <td>1</td>
      <td>2.594863</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>846</th>
      <td>0.889082</td>
      <td>1</td>
      <td>0.496463</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>268 rows × 9 columns</p>
</div>



Since the variable has completed to convert from text to number, we can move to next step

## Train the model

When the y variabel is numerical and continous we can use linear regression model, we want model can capture more generalize variability of the data so we can use cross-valdiation, so the model more robust and more generalization


```python
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
result = []
r2_result = []
model = LinearRegression()
#make fold
kf = KFold(n_splits=5, shuffle=True, random_state=42)
for i,(j,k) in enumerate(kf.split(new_train)):
    #subset data
    x_train,x_test = new_train.iloc[j], new_train.iloc[k]
    y_train,y_test = ytrain.iloc[j], ytrain.iloc[k]
    #pred model
    model.fit(x_train,y_train)
    ypred = model.predict(x_test)
    #evaluate we using mae because target variable is continous and MAE dont influenced of outlier
    mae = mean_absolute_error(y_test,ypred)
    result.append(mae)
    #r2 score
    r2 = r2_score(y_test,ypred)
    r2_result.append(r2)
for i,j in enumerate(r2_result):
    print('R2 for fold',i+1,'is',j)
```

    R2 for fold 1 is 0.726802980650122
    R2 for fold 2 is 0.7064483761148304
    R2 for fold 3 is 0.7747167341519964
    R2 for fold 4 is 0.7091969312286515
    R2 for fold 5 is 0.7770987387609974
    

R squared is measure how model performance predict true, in this model we get 0.7 means 70% of all data can deine by model. In insurance world whicj there're many unpredectable factor that can lead variance, 0.7 is reasonably good

## test the model


```python
ypred = model.predict(new_test)
#r2 score
r2_score = r2_score(ytest,ypred)
print(r2_score)
```

    0.7824789495519924
    

The r2 score is 0.7824789495519924 which is reasonable good since there're unpredictable factor is larger

## Who is more contribute in the model for all variable predictor


```python
import statsmodels.api as sm
#add constant
X_const = sm.add_constant(new_test.reset_index(drop=True))
#get the greatest contribute to the model
model_sm = sm.OLS(ytest.reset_index(drop=True),X_const).fit()
model_sm.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>         <td>charges</td>     <th>  R-squared:         </th> <td>   0.790</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.783</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   121.7</td>
</tr>
<tr>
  <th>Date:</th>             <td>Tue, 15 Apr 2025</td> <th>  Prob (F-statistic):</th> <td>3.50e-83</td>
</tr>
<tr>
  <th>Time:</th>                 <td>18:58:09</td>     <th>  Log-Likelihood:    </th> <td> -2698.6</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>   268</td>      <th>  AIC:               </th> <td>   5415.</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>   259</td>      <th>  BIC:               </th> <td>   5447.</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     8</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
      <td></td>         <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th>     <td> 5648.1885</td> <td>  521.177</td> <td>   10.837</td> <td> 0.000</td> <td> 4621.905</td> <td> 6674.472</td>
</tr>
<tr>
  <th>age</th>       <td> 3520.3604</td> <td>  360.987</td> <td>    9.752</td> <td> 0.000</td> <td> 2809.516</td> <td> 4231.204</td>
</tr>
<tr>
  <th>sex</th>       <td>  473.9431</td> <td>  730.907</td> <td>    0.648</td> <td> 0.517</td> <td> -965.335</td> <td> 1913.221</td>
</tr>
<tr>
  <th>bmi</th>       <td> 2245.9003</td> <td>  387.954</td> <td>    5.789</td> <td> 0.000</td> <td> 1481.955</td> <td> 3009.846</td>
</tr>
<tr>
  <th>children</th>  <td>  730.5722</td> <td>  310.547</td> <td>    2.353</td> <td> 0.019</td> <td>  119.053</td> <td> 1342.092</td>
</tr>
<tr>
  <th>smoker</th>    <td>  2.48e+04</td> <td>  900.524</td> <td>   27.543</td> <td> 0.000</td> <td>  2.3e+04</td> <td> 2.66e+04</td>
</tr>
<tr>
  <th>northeast</th> <td> 2613.0588</td> <td>  672.765</td> <td>    3.884</td> <td> 0.000</td> <td> 1288.274</td> <td> 3937.844</td>
</tr>
<tr>
  <th>northwest</th> <td> 2178.7190</td> <td>  645.865</td> <td>    3.373</td> <td> 0.001</td> <td>  906.904</td> <td> 3450.534</td>
</tr>
<tr>
  <th>southeast</th> <td>    0.4621</td> <td>  629.031</td> <td>    0.001</td> <td> 0.999</td> <td>-1238.204</td> <td> 1239.128</td>
</tr>
<tr>
  <th>southwest</th> <td>  855.9486</td> <td>  664.442</td> <td>    1.288</td> <td> 0.199</td> <td> -452.447</td> <td> 2164.344</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>55.607</td> <th>  Durbin-Watson:     </th> <td>   2.190</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td> <th>  Jarque-Bera (JB):  </th> <td> 111.094</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 1.052</td> <th>  Prob(JB):          </th> <td>7.52e-25</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 5.350</td> <th>  Cond. No.          </th> <td>1.46e+16</td>
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.<br/>[2] The smallest eigenvalue is 4.3e-30. This might indicate that there are<br/>strong multicollinearity problems or that the design matrix is singular.



we look at the coef column, smoker variable become most largest contribute to the model linear regression with 2.48e+04 coef, following region, BMI, age, and lowest is sex
