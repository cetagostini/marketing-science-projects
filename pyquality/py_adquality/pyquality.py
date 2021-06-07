# -*- coding: utf-8 -*-
"""
Created on Sat Jun  5 00:01:44 2021

@author: Carlos Eduardo Trujillo Agostini
"""

import requests
import json
import time

import pandas as pd

from datetime import date, timedelta

class adquality_reports:
  """
    A class to generate requests to the Adquality REST API and get a report.

    ...

    Attributes
    ----------
    username : str
        user used to login in Adquality
    password : str
        password used to login in Adquality

    Methods
    -------
    report_generator(since = None, until = None, country = None, report_type = None):
        Returns a dataframe with all the API data
    
    image_extractor(since = None, until = None, country = None):
        Returns a dataframe with all the data of the ads obtained by Adquality
  
  """
  current_date = date.today().isoformat()   
  days_before = (date.today()-timedelta(days=30)).isoformat()

  def __init__(self, username = None, password = None):
    """
    You provide the necessary data to authenticate within Adquality.

    Parameters
    ----------
        username : str
            username used to login in Adquality
        password : str
            password used to login in Adquality
    """
    if isinstance(username, type(None)):
      username_error = 'Define your username'
      raise username_error
    
    if isinstance(password, type(None)):
      password_error = 'Define your password'
      raise password_error

    self.username = username
    self.password = password
    
    data_rest = {"username": self.username, "password": self.password}
    
    response = requests.post('https://adcuality.com/api/login', data = data_rest)
    print(response)
    res = json.loads(response.text)
    self.token = res.get('token')    
    
    self.header = {'Origin': 'https://adcuality.com/login',
      'Authorization': 'Bearer ' + self.token,
      'Content-Type': 'application/json',
      'Accept': "application/json",
      'Cache-control': "no-cache",
      'User-Agent': 'Chrome/54.0.2840.90'}    
    
  @classmethod
  def dataframe_transform_sov(self, requested_value):
    """
    It receives the result of the request and returns a concatenated
    dataframe of the different values ​​of the json that adquality.

    Parameters
    ----------
        requested_value : str
            Response of request

    """
    adquality_report = json.loads(requested_value.text)
    dataframe_report = pd.DataFrame(adquality_report[0]['tableData'])

    frames = pd.DataFrame()
    for i in range(len(dataframe_report)):

      values = pd.DataFrame(dataframe_report.histogram[i])
      values['indexes'] = 1
      temporal = dataframe_report[['id', 'label', 'percentage', 'total', 'viewability']].iloc[i:i+1]
      temporal['indexes'] = 1

      frames = pd.concat([frames, pd.merge(temporal, values, how = 'left', on = 'indexes').drop(['indexes'], axis = 1)])
      frames.reset_index(inplace= True, drop = True)
    
    return frames

  @classmethod
  def dataframe_transform_soi(self, requested_value):
    """
    It receives the result of the request and returns a concatenated
    dataframe of the different values ​​of the json that adquality.

    Parameters
    ----------
        requested_value : str
            Response of request

    """
    adquality_report = json.loads(requested_value.text)
    dataframe_report = pd.DataFrame(adquality_report[0]['tableData'])

    frames = pd.DataFrame()
    for i in range(len(dataframe_report)):

      values = pd.DataFrame(dataframe_report.histogram[i])
      values['indexes'] = 1
      temporal = dataframe_report[['id', 'label', 'percentage', 'amount', 'prints']].iloc[i:i+1]
      temporal['indexes'] = 1

      frames = pd.concat([frames, pd.merge(temporal, values, how = 'left', on = 'indexes').drop(['indexes'], axis = 1)])
      frames.reset_index(inplace= True, drop = True)
    
    frames.columns = ['id', 'label', 'percentage', 'total_amount', 'total_prints', 'date', 'amount', 'prints']
    
    return frames
    
  @classmethod
  def recursive_image_requests(self, number = None, mydataframe = None, params = None, username = None, password = None):
      """
    A recursive function that returns all the images captured by Adquality.

    Parameters
    ----------
        number : int
            Number greater than 1.
        mydataframe : DataFrame
            DataFrame empty.
        params : list
            A list that contains a dictionary with the parameters of the API query
        username : str
            The user delivered when creating the class
        password : str
            The password delivered when creating the class
      
      Returns
      -------
        DataFrame
        
      """
      if isinstance(number, type(None)):
        number = 1
    
      if isinstance(mydataframe, type(None)):
        mydataframe = pd.DataFrame()       

      if isinstance(params, type(None)):
        params_error = 'Define your params'
        raise params_error

      requested = requests.post(
        url = 'https://adcuality.com/api/v2/gallery/' + '{}'.format(number) + '/1000?unique=1',
        headers = {'Origin': 'https://adcuality.com/login',
                  'Authorization': 'Bearer ' + json.loads(requests.post('https://adcuality.com/api/login', 
                                                                        data = {"username": username, "password": password}).text).get('token'),
                  'Content-Type': 'application/json',
                  'Accept': "application/json",
                  'Cache-control': "no-cache",
                  'User-Agent': 'Chrome/54.0.2840.90'},
        json = params)
      
      time.sleep(0.5)
      adquality_report = json.loads(requested.text)
      pages = adquality_report[0]['metadata']['pages']
    
      temporal = pd.DataFrame(adquality_report[0]['items'])
      
      other = pd.concat([mydataframe, temporal])
    
      if number <= pages:
        return adquality_reports.recursive_image_requests(number = number + 1, mydataframe = other, params = params, username = username, password = password)
      else:
        return other

  
  def report_generator_sov(self, since = None, until = None, country = None, report_type = None):

    """
    A function that generates a platform report on DataFrame format.

    Parameters
    ----------
        since : int
            The start date of the report. The default is 30 days prior to the current day.
        until : DataFrame
            The end date of the report. The default is the current day.
        country : list
            The possible countries to choose entered as a string
        report_type : str
            The type of report generated by the API. The options are 'publishers', 'formats',
            'advertisers', 'industries', 'sources' or 'categories'
      
      Returns
      -------
        DataFrame

    """      
      

    if isinstance(since, type(None)):
      since = str(self.days_before)
      print('if you didnt define the start date, the default is {}'.format(self.days_before))
    
    if isinstance(until, type(None)):
      until = str(self.current_date)
      print('if you didnt define the until date, the default is {}'.format(self.current_date))

    if isinstance(country, type(None)):
      error_country = 'Define your country'
      raise error_country

    if isinstance(report_type, type(None)):
      report_type_error = '¡Define your report! Use one of the following options: "publishers", "formats", "advertisers", "industries", "sources", "categories"'
      raise report_type_error
    
    param = [{
    		"since": since,
    		"until": until,
    		"country": country,
    		"advertisers": [],
    		"industries": [],
    		"platforms": [],
    		"products": [],
    		"categories": [],
    		"formats": [],
    		"publishers": [],
    		"excludedPublishers": [],
    		"sources": []
    	    }]
    requested = requests.post(
      url = 'https://adcuality.com/api/v2/sov/' + report_type + '?api=1',
      headers = self.header,
      json = param)
    
    print(requested)
    
    report = adquality_reports.dataframe_transform_sov(requested)
    return report


  def report_generator_soi(self, since = None, until = None, country = None, report_type = None):

    """
    A function that generates a platform report on DataFrame format.

    Parameters
    ----------
        since : int
            The start date of the report. The default is 30 days prior to the current day.
        until : DataFrame
            The end date of the report. The default is the current day.
        country : list
            The possible countries to choose entered as a string
        report_type : str
            The type of report generated by the API. The options are 'publishers', 'formats',
            'advertisers', 'industries', 'sources' or 'categories'
      
      Returns
      -------
        DataFrame

    """      
      

    if isinstance(since, type(None)):
      since = str(self.days_before)
      print('if you didnt define the start date, the default is {}'.format(self.days_before))
    
    if isinstance(until, type(None)):
      until = str(self.current_date)
      print('if you didnt define the until date, the default is {}'.format(self.current_date))

    if isinstance(country, type(None)):
      error_country = 'Define your country'
      raise error_country

    if isinstance(report_type, type(None)):
      report_type_error = '¡Define your report! Use one of the following options: "publishers", "formats", "advertisers", "industries", "sources", "categories"'
      raise report_type_error
    
    param = [{
    		"since": since,
    		"until": until,
    		"country": country,
    		"advertisers": [],
    		"industries": [],
    		"platforms": [],
    		"products": [],
    		"categories": [],
    		"formats": [],
    		"publishers": [],
    		"excludedPublishers": [],
    		"sources": []
    	    }]
    requested = requests.post(
      url = 'https://adcuality.com/api/v2/soi/' + report_type + '?api=1',
      headers = self.header,
      json = param)
    
    print(requested)
    
    report = adquality_reports.dataframe_transform_soi(requested)
    return report


  def image_extractor(self, since = None, until = None, country = None):

    """
    A function that makes a request to the API and uses the previous recursive function
    to obtain a result of a DataFrame with all the elements captured by adquality

    Parameters
    ----------
        since : int
            The start date of the report. The default is 30 days prior to the current day.
        until : DataFrame
            The end date of the report. The default is the current day.
        country : list
            The possible countries to choose entered as a string
      
      Returns
      -------
        DataFrame

    """  
      
    if isinstance(since, type(None)):
      since = str(self.days_before)
      print('if you didnt define the start date, the default is {}'.format(self.days_before))
    
    if isinstance(until, type(None)):
      until = str(self.current_date)
      print('if you didnt define the until date, the default is {}'.format(self.current_date))

    if isinstance(country, type(None)):
      error_country = 'Define your country'
      raise error_country

    param = [{
    		"since": since,
    		"until": until,
    		"country": country,
    		"advertisers": [],
    		"industries": [],
    		"platforms": [],
    		"products": [],
    		"categories": [],
    		"formats": [],
    		"publishers": [],
    		"excludedPublishers": [],
    		"sources": []
    	    }]
    
    image_extractions = adquality_reports.recursive_image_requests(params = param, username = self.username, password = self.password)
    return image_extractions
