---
title: Read data
---
### csv file
Using Pandas package  [read_csv](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html).


```python
import pandas as pd
df = pd.read_csv("data.csv", encoding = "utf-8") # check encoding type such like "utf-16"
 ```

### data from website
Reading data about house price and house feature from a website as an example:


```python
import requests
from bs4 import BeautifulSoup

def soup_to_df(s):
    #define column name
    dfcols = ['outcode', 'last_published_date','latitude', 'longitude', 'post_town', 'num_bathrooms', 'num_bedrooms', 'num_floors', 
              'num_recepts', 'property_type', 'street_name', "price"]
    df_xml = pd.DataFrame(columns=dfcols)

    for node in s.find_all("listing"):
        outcode =  node.find('outcode').get_text()
        last_published_date = node.find('last_published_date').get_text()
        latitude = node.find('latitude').get_text()
        longitude = node.find('longitude').get_text()
        post_town = node.find('post_town').get_text()
        num_bathrooms = node.find('num_bathrooms').get_text()
        num_bedrooms = node.find('num_bedrooms').get_text()
        num_floors = node.find('num_floors').get_text()
        num_recepts = node.find('num_recepts').get_text()
        property_type = node.find('property_type').get_text()
        street_name = node.find('street_name').get_text()
        price = node.find('price').get_text()

        df_xml = df_xml.append(pd.Series([outcode, float(latitude), float(longitude), post_town, last_published_date, int(num_bathrooms),
                                          int(num_bedrooms), int(num_floors), int(num_recepts), property_type, street_name, int(price)], index=dfcols),ignore_index=True)   
    return df_xml

dfcols = ['outcode', 'latitude', 'longitude', 'post_town', 'last_published_date', 'num_bathrooms', 'num_bedrooms', 'num_floors', 
              'num_recepts', 'property_type', 'street_name', "price"]

df_all = pd.DataFrame(columns=dfcols)

for i in range(1, 58): #page range
# Ctrl+Shift+I in that website to find the url for which we would like to scrape. 
    baseurl = f"https://api.zoopla.co.uk/api/v1/property_listings?api_key=9zpbeza9n858g3u2g633u3rb&county=Somerset&country=England&listing_status=sale&include_sold=1&page_number={i}&page_size=100"
    page = requests.get(baseurl)
    soup = BeautifulSoup(page.content)
    #save data from one page into a dataframe
    df = soup_to_df(soup)
    # combine all data from all pages into one dataframe
    df_all = pd.concat([df_all, df], axis = 0) 
```
