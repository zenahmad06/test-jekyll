<h1>Get all list movie name year 2015</h1>


```python
from bs4 import BeautifulSoup
import requests
response=requests.get('https://id.wikipedia.org/wiki/Kategori:Film_tahun_2015')
#parser
page=BeautifulSoup(response.text,'html.parser')
#find container
container_movie=page.find_all('div',{'class':'mw-category mw-category-columns'})
#find all a tag in container
a_tag=container_movie[-1].find_all('a')
lismovie=[a_tag[i].text.strip() for i,j in enumerate(a_tag) ]
#clean the text from word film
import regex as re
clean_list=[]
for i in lismovie:
    rule=r'\s*\(.*?\bfilm\b.*?\)' #\s space, \( matching open parenthesis, .*? macthing anything before.\boundary
    clean=re.sub(rule,"",i,re.IGNORECASE).replace("()","").strip()
    clean_list.append(clean)
```


```python
list_movie_clean=clean_list.copy()
list_movie_clean[:5]
len(list_movie_clean)
```




    195



<h1>Get Movie Description in web "https://www.omdbapi.com/"</h1>


```python
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
```

**Explaine**:
1. the syntax from selenium.webdriver.common.keys import Keys is to stimulate keyboard actions like enter, tab, or back space
2. The syntax from selenium.webdriver.support.ui import WebDriverWait to wait the content before we interact them. Ussualy selenium search the class or tag too fast so when the content is no yet load due internet speed or javascript, it will become fail so we set time for us to wait load content
3. The syntax this is from selenium.webdriver.support import ExpectedCondition as EC to wait webdriver until this syntag appear

<p> Before we write scraping code always doing manual inspectio to know whic tag and class name will we used</p>

<h2>FLOW THE AUTOMATION</h2>

1. fill the text
2. click button
3. wait driver 10 until div ID search by title response appear
4. search pre inside div
5. print result
6. click button reset
7. before we go to next search, we validating pre should empty value


```python
import json
import time
import requests  
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
kol=['Title','Released','Year','Genre']
import pandas as pd
df=pd.DataFrame(columns=kol)
driver = webdriver.Chrome()
driver.get("https://www.omdbapi.com/")  # Replace with actual search page
for i in list_movie_clean[:5]:
    #get container form
    form=driver.find_element(By.ID,'search-by-title-form')
    # get input field
    input_field=form.find_element(By.ID,"t")
    #clear the input
    input_field.clear()
    #fill the input_field
    input_field.send_keys(i)
    #we need previous value response to compare existing loop
    #in loop 1 we compare value from loop at 0, etc
    try:
        prev_response=driver.find_element(By.XPATH,"div[@id='search-by-title-response']/pre").text.strip()
    except:
        prev_response=""
    #we need 3 repeat to validating when result show we break the loop
    n=0
    stat="False"
    while n < 3:
        #get button
        button_search=form.find_element(By.ID,'search-by-title-button')
        button_search.click()
        #get respon, we need waiting time
        WebDriverWait(driver,20).until(EC.visibility_of_element_located((By.XPATH,"//div[@id='search-by-title-response']/pre")))
        try:
            WebDriverWait(driver,20).until(
                lambda driver:driver.find_element(By.XPATH,"//div[@id='search-by-title-response']/pre").text.strip()!="" and 
                driver.find_element(By.XPATH,"//div[@id='search-by-title-response']/pre").text.strip() !=prev_response)
            response=driver.find_element(By.XPATH,"//div[@id='search-by-title-response']/pre").text.strip()
        except TimeoutException as e:
            print (e)
            response=""
        if response !="" and response !=prev_response:
            print(response)
            stat="True"
            break
     
        
        n+=1
        time.sleep(5)
    if stat== 'True' and response!=prev_response:
        json_dat=json.loads(response)
        resp_status=json_dat['Response']
        if resp_status=='False':
            df=df
        else:
            kum={}
            for i in kol:
                kum[i]=json_dat[i]
            dat=pd.DataFrame([kum])
            df=pd.concat([df,dat],ignore_index=True)
    #button reset
    button_reset=driver.find_element(By.ID,'search-by-title-reset')
    button_reset.click()
    #validating pre should empty after reset
    WebDriverWait(driver,20).until(
        lambda driver:driver.find_element(By.XPATH,"//div[@id='search-by-title-response']/pre").text.strip()=="" )
                
    time.sleep(10)
    
```

    Message: 
    
    Message: 
    
    Message: 
    
    {"Title":"11 Minutes","Year":"2015","Rated":"Not Rated","Released":"08 Apr 2016","Runtime":"81 min","Genre":"Drama, Thriller","Director":"Jerzy Skolimowski","Writer":"Jerzy Skolimowski","Actors":"Richard Dormer, Paulina Chapko, Wojciech Mecwaldowski","Plot":"The lives of several Varsovians are intertwined for just 11 minutes. These minutes turn out to be crucial for their ultimate fate.","Language":"Polish, English","Country":"Poland, Ireland","Awards":"10 wins & 9 nominations total","Poster":"https://m.media-amazon.com/images/M/MV5BYzAzOWM0ZWYtMmM1Yi00YzY4LWEwYjAtOTY3ZWExMzc4NzJkXkEyXkFqcGc@._V1_SX300.jpg","Ratings":[{"Source":"Internet Movie Database","Value":"5.7/10"},{"Source":"Rotten Tomatoes","Value":"76%"},{"Source":"Metacritic","Value":"51/100"}],"Metascore":"51","imdbRating":"5.7","imdbVotes":"2,103","imdbID":"tt3865478","Type":"movie","DVD":"N/A","BoxOffice":"N/A","Production":"N/A","Website":"N/A","Response":"True"}
    Message: 
    
    {"Response":"False","Error":"Movie not found!"}
    Message: 
    
    {"Title":"1400","Year":"2015","Rated":"N/A","Released":"26 Aug 2016","Runtime":"89 min","Genre":"Drama, Music, Romance","Director":"Derrick Lui","Writer":"Terence Ang, Derrick Lui","Actors":"Desmond Tan, Yahui Xu, Vincent Tee","Plot":"4 intertwined stories explore the notion of love in a hotel, how important love is, to what extremes we go to make it happen, but in difference situations and circumstances, the outcomes may be more than what we bargain for.","Language":"Mandarin, English, Hokkien, Japanese","Country":"Singapore","Awards":"1 win & 7 nominations","Poster":"https://m.media-amazon.com/images/M/MV5BMTQxNzc5NDc1Ml5BMl5BanBnXkFtZTgwMDYxMDQ3MjE@._V1_SX300.jpg","Ratings":[{"Source":"Internet Movie Database","Value":"4.3/10"}],"Metascore":"N/A","imdbRating":"4.3","imdbVotes":"32","imdbID":"tt4029928","Type":"movie","DVD":"N/A","BoxOffice":"N/A","Production":"N/A","Website":"N/A","Response":"True"}
    Message: 
    
    {"Title":"1944","Year":"2015","Rated":"N/A","Released":"04 Nov 2015","Runtime":"100 min","Genre":"Drama, History, War","Director":"Elmo Nüganen","Writer":"Leo Kunnas","Actors":"Kaspar Velberg, Kristjan Üksküla, Maiken Pius","Plot":"In 1944 Estonia, a fratricide war ensues when Estonians of the retreating German forces fight against Estonians conscripted into the advancing Soviet Red Army.","Language":"Estonian, Russian, German","Country":"Estonia, Finland","Awards":"2 wins & 5 nominations total","Poster":"https://m.media-amazon.com/images/M/MV5BYjcyZjI5MjUtYWYwMS00MjA1LWJhMDEtMzI1NThiMjAwNWJlXkEyXkFqcGc@._V1_SX300.jpg","Ratings":[{"Source":"Internet Movie Database","Value":"7.0/10"}],"Metascore":"N/A","imdbRating":"7.0","imdbVotes":"6,501","imdbID":"tt3213684","Type":"movie","DVD":"N/A","BoxOffice":"N/A","Production":"N/A","Website":"N/A","Response":"True"}
    


```python
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
      <th>Title</th>
      <th>Released</th>
      <th>Year</th>
      <th>Genre</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>11 Minutes</td>
      <td>08 Apr 2016</td>
      <td>2015</td>
      <td>Drama, Thriller</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1400</td>
      <td>26 Aug 2016</td>
      <td>2015</td>
      <td>Drama, Music, Romance</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1944</td>
      <td>04 Nov 2015</td>
      <td>2015</td>
      <td>Drama, History, War</td>
    </tr>
  </tbody>
</table>
</div>


