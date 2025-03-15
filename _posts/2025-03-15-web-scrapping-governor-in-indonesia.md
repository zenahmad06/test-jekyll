---
layout: posts
permalink : /web-scrapping-governor/
---



```python
from bs4 import BeautifulSoup
import requests
url="https://id.wikipedia.org/wiki/Daftar_gubernur_dan_wakil_gubernur_di_Indonesia"
#get the response
response=requests.get(url)
#parser the html
page=BeautifulSoup(response.text,"html.parser")
```

<h2> Inspect the wikipedia </h2> 
<p>identify the class name and id</p>

1.Governor name

- ![image.png]({{ site.baseurl }}/assets/image-scrap/image.png)

2.Province

![image.png]({{ site.baseurl }}/assets/image-scrap/image.png)

3.Personal wikipedia

![image.png]({{ site.baseurl }}/assets/image-scrap/image.png)

<h2> Start to Scrapping</h2>


```python
#get the container first
table_container=page.find('table',class_='wikitable')
# get all t4
import regex as re
tr_tag=table_container.find_all('tr')
governor_name=[]
province=[]
link=[]
for i,j in enumerate(tr_tag):
    if i!=0:
        a_tag=tr_tag[i].find_all('a')
        href=a_tag[5].get('href')
        for k,j in zip(a_tag[1],a_tag[5]):
            rule=r"\(.*?\)" #parenthesis
            clean=re.sub(rule,"",j.text) #remove parenthesis
            governor_name.append(clean)
            province.append(k.text)
        wiki="https://id.wikipedia.org"
        link.append(wiki+href)
import pandas as pd

id_value=[f"TEST00{i}" for i in range(1,len(governor_name)+1)]
kol=['ID','Name','Province','Url']
kum={}
for i,j in zip(kol,[id_value,governor_name,province,link]):
    kum[i]=j
df=pd.DataFrame(kum)
df.to_csv("C:/Users/User/Documents/Extracting_Data_from_Wikipedia_soal_1.csv")
```

<h2>EXERCISE 2 </h2>


**get profile governor name for each link wikipedia profile**


```python
url="https://id.wikipedia.org/wiki/Daftar_gubernur_dan_wakil_gubernur_di_Indonesia"
#get the response
response=requests.get(url)
#parser the html
page=BeautifulSoup(response.text,"html.parser")
#get the container first
table_container=page.find('table',class_='wikitable')
# get all t4
tr_tag=table_container.find_all('tr')
link_new=[]
for i,j in enumerate(tr_tag):
    if i!=0:
        a_tag=tr_tag[i].find_all('a')
        href=a_tag[5].get('href')
        wiki="https://id.wikipedia.org"
        link_new.append(wiki+href)
len(link_new)
```




    38



**loop every link**. identify class name/id name

1. full name

![image.png]({{ site.baseurl }}/assets/image-scrap/image.png)

2. Place of Birth and Date

![image.png]({{ site.baseurl }}/assets/image-scrap/image.png)

**get the data**


```python
import requests
from bs4 import BeautifulSoup
import regex as re
#make function regular expression
def clean_name(x):
    rule=r"\(.*?\)" #identifi word that inside ()
    clean=re.sub(rule,"",x)
    return clean

def clean_umur(x):
    rl=r"\s*\(.*?\bumur\b.*?\)" # identify word umur inside () that before () is space
    srcKata=re.search(rl,x)
    if srcKata!=None:
        cleanKata=re.sub(rl,";",x)
        li=cleanKata.split(";")
        return li
    else:
        return None
#defining the list
name=[]
date=[]
place=[]
for i in link_new:
    respon=requests.get(i)
    page_gov=BeautifulSoup(respon.text,"html.parser")
    #get full name
    full_name=page_gov.find(id="firstHeading")
    nama=clean_name(full_name.text)
    name.append(nama)
    
    #get date and place birth
    td_infobox=page_gov.find_all('td',{"class":"infobox-data"})
    for j in td_infobox:
        kal=j.text
        li=clean_umur(kal.strip())
        if li==None:
            text = kal
            rs=r"(\d{2} \w+ \d{4})(\w+.*)"  #identifi formal 11 word 2022: grup 1 and grup 2 : follow all character

            match=re.search(pattern,text)
            

            if match:
                tgl=match.group(1).strip()
                date.append(tgl)
                place.append(match.group(2).strip())
            else:
                continue
        else:
            tanggal=li[0]
            tept=li[1]
            check=re.match(r"^\D",tanggal) #start sentence with string
            if check!=None:
                rla=r"^(.*?)(\d{1,2} \w+ \d{4})" #format tanggal
                ka=re.search(rla,tanggal)
                kat_br=re.sub(rla,ka.group(2),tanggal)
                
                date.append(kat_br)
            else:
                date.append(tanggal)
                
            cl_tp=re.sub(r"\[\d+\]","",tept) #identify number that inside []
            place.append(cl_tp)
            
import pandas as pd

id_val=[f"TEST00{i}" for i in range(1,len(name)+1)]
kol=['ID','Nama','Tempat lahir','Tanggal Lahir','Wikipedia Link']
kum={}
for i,j in zip(kol,[id_val,name,place,date,link_new]):
    kum[i]=j
df=pd.DataFrame(kum)
df.to_csv("C:/Users/User/Documents/Extracting_Data_from_Wikipedia_soal_2.csv")
```
