# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 00:05:49 2021

@author: Carlos Trujillo
"""

import requests
import json

import time
from datetime import date, timedelta

import itertools

from ftfy import fix_encoding
import unidecode

import pandas as pd

class admetricks_api:

  """
    A class to generate requests to the Admetricks REST API and get a report.
    
    The operation of the methods is subject to the correct input of the variables and
    to the admetricks user you are using having access to the admetricks REST API.
    
    
    To learn more about the API documentation, go to https://dev.admetricks.com/#introduccion
    ...

    Attributes
    ----------
    username : str
        user used to login in Admetricks
    password : str
        password used to login in Admetricks

    Methods
    -------
    reports_generator(country = None, ad_type = None, device = None, since_date = None):
        Returns a dataframe with a full report with the information of the API
    
    screenshots_data(country = None, site = None, since_date = None, until_date = None):
        Returns a dataframe with the raw information of captured screenshots
  
  """    


  dictionary_countrys = {1:'chile',
                          2:'colombia',
                          3:'argentina',
                          4:'brasil',
                          5:'españa',
                          6:'peru',
                          7:'mexico',
                          8:'honduras',
                          9:'puerto rico',
                          10:'panama',
                          11:'uruguay',
                          12:'costa rica',
                          13:'guatemala',
                          14:'ecuador',
                          15:'venezuela',
                          16:'nicaragua',
                          17:'salvador',
                          18:'republica dominicana',
                          19:'paraguay'}

  device = {1:'desktop', 2:'mobile'}
  ad_type = {1:'display', 2:'video', 3:'text'}
  
  current_date = date.today().isoformat()   
  days_before = (date.today()-timedelta(days=30)).isoformat()

  combinations = list(itertools.product(list(device.values()),list(ad_type.values())))

  def __init__(self, username = None, password = None):
      
    """
    You provide the necessary data to authenticate within Admetricks.

    Parameters
    ----------
        username : str
            username used to login in Admetricks
        password : str
            password used to login in Admetricks
    """
    
    self.username = username
    self.password = password

    url = """https://clientela.admetricks.com/o/token/?username={username}&password={password}&client_id=IW8M80h7qgCaSz4hPm3gr3wJP89NiJTPyhkwPurT&client_secret=KnBW84uyHlxwlNrKOXyym6Ro1IT6IlYdhScdop63hHddCzJIxUwDG7VItNgEONb1U2ebEH6fBmkYgX9LrZD4uqFJlYscHYn9MLxOm2qVccNE2WGEuePpKA7t3jQ2CvMu&grant_type=password"""
    response = requests.post(url.format(username = self.username, password = self.password))

    res = json.loads(response.text)
    self.token = res.get('access_token')
    print('Your active token is {}'.format(self.token))
    print(response)

  def reports_generator(self, country = None, ad_type = None, device = None, since_date = None):
    """
    A function that returns a dataframe with a full report with the information of the API.

    Parameters
    ----------
        country : str
            name of your country.
        ad_type : str
            Type of ad you want to study. The options are: [all, display, video, text]
        device : str
            Type of device you want to study. The options are: [all, desktop, mobile]
        since_date : str
            From what date do you want to export data.
      
      Returns
      -------
        DataFrame
        
    """
    
    if isinstance(country, type(None)):
      country_error = 'Define your country'
      raise country_error
    
    if isinstance(ad_type, type(None)):
      ad_type = 'all'  

    if isinstance(device, type(None)):
      device = 'all'  

    if isinstance(since_date, type(None)):
      since_date = str(self.days_before)  
    
    country = country.lower()
    country = unidecode.unidecode(country)
    
    my_dataframe = pd.DataFrame()
    
    header = {
      'Authorization': 'Bearer '+ self.token,
      'content-type': 'application/json'}

    country_value = list(self.dictionary_countrys.keys())[list(self.dictionary_countrys.values()).index(country)]

    if ad_type == 'all':
      if device == 'all':
        for devices, ad_types in self.combinations:
          device_value = list(self.device.keys())[list(self.device.values()).index(devices)]
          ad_type_value = list(self.ad_type.keys())[list(self.ad_type.values()).index(ad_types)]

          params = (('day', since_date), ('country', str(country_value)), ('device', str(device_value)), ('ad_type', str(ad_type_value)),)

          requested = requests.post(url = 'https://clientela.admetricks.com/market-report/data/v3/', headers = header, params = params)
          data = json.loads(requested.text)

          my_dataframe = pd.concat([my_dataframe, pd.DataFrame.from_dict(data['data'])])
          time.sleep(0.5)
      else:
        device_value = list(self.device.keys())[list(self.device.values()).index(device)]
        for value, names in self.ad_type.items():
          params = (('day', since_date), ('country', str(country_value)), ('device', str(device_value)), ('ad_type', str(value)),)
          requested = requests.post(url = 'https://clientela.admetricks.com/market-report/data/v3/', headers = header, params = params)
          data = json.loads(requested.text)

          my_dataframe = pd.concat([my_dataframe, pd.DataFrame.from_dict(data['data'])])
          time.sleep(0.5)
    else:
      if device == 'all':

        ad_type_value = list(self.ad_type.keys())[list(self.ad_type.values()).index(ad_type)]

        for value, names in self.device.items():
          params = (('day', since_date), ('country', str(country_value)), ('device', str(value)), ('ad_type', str(ad_type_value)),)
          requested = requests.post(url = 'https://clientela.admetricks.com/market-report/data/v3/', headers = header, params = params)
          data = json.loads(requested.text)

          my_dataframe = pd.concat([my_dataframe, pd.DataFrame.from_dict(data['data'])])
          time.sleep(0.5)

      else:  
        device_value = list(self.device.keys())[list(self.device.values()).index(device)]
        ad_type_value = list(self.ad_type.keys())[list(self.ad_type.values()).index(ad_type)]

        params = (('day', since_date), ('country', str(country_value)), ('device', str(device_value)), ('ad_type', str(ad_type_value)),)
        
        requested = requests.post(url = 'https://clientela.admetricks.com/market-report/data/v3/', headers = header, params = params)
        data = json.loads(requested.text)

        my_dataframe = pd.concat([my_dataframe, pd.DataFrame.from_dict(data['data'])])
    
    my_dataframe.reset_index(drop = True, inplace = True)

    my_dataframe.campaign_name = my_dataframe.campaign_name.apply(lambda x: fix_encoding(x))
    my_dataframe.campaign_tags = my_dataframe.campaign_tags.apply(lambda x: fix_encoding(x))
    
    return my_dataframe


  def screenshots_data(self, country = None, site = None, since_date = None, until_date = None):
    """
    A function that returns a dataframe with a full report with the information of the API.

    Parameters
    ----------
        country : str
            name of your country.
        site : str
            The name of the site
        until_date : str
            The end date of the period to export the data.
        since_date : str
            The start date of the period to export the data.
      
      Returns
      -------
        DataFrame
        
    """

    if isinstance(country, type(None)):
      country_error = 'Define your country'
      raise country_error
    
    if isinstance(site, type(None)):
      site_error = 'Define your site'
      raise site_error

    if isinstance(until_date, type(None)):
      until_date = str(self.current_date)
      print('the default end date is {}'.format(until_date))

    if isinstance(since_date, type(None)):
      since_date = str(self.days_before)
      print('the default start date is {}'.format(since_date))
    
    country = country.lower()
    country = unidecode.unidecode(country)
    
    country_value = list(self.dictionary_countrys.keys())[list(self.dictionary_countrys.values()).index(country)]


    headers = {
        'accept': 'application/json, text/plain, */*',
        'Authorization': 'Bearer '+ self.token,
        'content-type': 'application/json;charset=UTF-8',
    }

    params = (
        ('countries', country_value),
        ('domain_autocomplete', site),
    )

    response = requests.get('https://clientela.admetricks.com/website/', headers=headers, params=params)
    data = json.loads(response.text)

    my_ids = [item['id'] for item in data]
    keys = 'id'

    my_dict = dict.fromkeys(my_ids,keys)

    reverse_dict = [{y:int(x)} for x,y in my_dict.items()]

    query ={"websites":{"include":f'{reverse_dict}'},
          "ad_types":{"include":[{"id":1},{"id":2},{"id":3}]},
          "countries":{"include":[{"id":country_value}]},
          "date_range":{"start":"{}T00:00:00.000".format(since_date),
          "end":"{}T23:59:59.999".format(until_date),
          "group_by":"day"},
          "devices":{"include":[{"id":1},{"id":2}]},
          "order_by":"valuation"}


    data = '{}'.format(query)

    response = requests.post('https://clientela.admetricks.com/screenshot/', headers=headers, data=data.replace("\'", '"').replace('"[', '[').replace('"]', ']').replace(']"}', ']}'))
    value = json.loads(response.text)

    dataframe = pd.DataFrame(value['data'])

    return dataframe