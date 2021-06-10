# Marketing Science | Admetricks library

Developed by Carlos Trujillo, Marketing Scientist.

## Introduction

If you are part of the marketing world you have surely heard about Admetricks, Comscore, Lotame or Adcuality. Tools that provide us with data
which we can use to carry out different types of studies to many of the phenomena that we face in this area.

Every day, the adoption of technology is more important than ever in order to take our analysis a little beyond the conventional.
This can represent a very accomplished challenge, because many of the tools that we use every day do not have their connectors enabled for everyone.
users, resulting in not being able to do large data downloads.

And those users who manage to obtain a connector to any of these platforms, a great way awaits them to understand the connector, its operability and
results, with the little information that exists on the internet.

Therefore, I decided to start small projects where I will be creating pseudo packages, for those who are in the world of marketing and programming. Waiting
to be able to shorten the adaptation curve of some of the professionals that we meet here.

This is my second package in this project. I present PyMetricks, a simplification of the connection to the Admetricks REST API in Python.

I await your feedback!

``` python

pip install py_metricks

from py_admetricks.pymetricks import admetricks_api
```

Initially the package has two methods, where each one returns a DataFrame. The main method generates a report in Pandas format where you can see all
the data obtained by the Admetricks REST API.

``` python

admetricks = admetricks_api(username = 'your_mail@enterprise.com', password = 'admetricks_password')
report = admetricks.reports_generator(country='chile', since_date='2021-05-31', device='mobile', ad_type='display')
report.head(10)

#country options 'chile','colombia','argentina','brasil','españa','peru','mexico','honduras','puerto rico','panama','uruguay','costa rica',
#                 'guatemala','ecuador','venezuela','nicaragua','salvador','republica dominicana' or 'paraguay'
```

The second package obtains all the images captured by Admetricks within a certain period, in the raw format, therefore they will see a complete capture
of the entire screen and not just the ad.

``` python

admetricks = admetricks_api(username = 'your_mail@enterprise.com', password = 'admetricks_password')
screenshots = admetricks.screenshots_data(country = 'chile', site = 'facebook', since_date = '2021-01-08', until_date = '2021-01-08')
screenshots.head(10)
```

Very soon, automatic learning modules will be added to exploit the information we have within the platform.

What do you think is the best module we can start with?