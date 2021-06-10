# Marketing Science | AdCuality library

Developed by Carlos Trujillo, Marketing Scientist.

## Introduction

If you are part of the marketing world you have surely heard about Admetricks, Comscore, Lotame or AdCuality. Tools that provide us with data
which we can use to carry out different types of studies to many of the phenomena that we face in this area.

Every day, the adoption of technology is more important than ever in order to take our analysis a little beyond the conventional.
This can represent a very accomplished challenge, because many of the tools that we use every day do not have their connectors enabled for everyone.
users, resulting in not being able to do large data downloads.

And those users who manage to obtain a connector to any of these platforms, a great way awaits them to understand the connector, its operability and
results, with the little information that exists on the internet.

Therefore, I decided to start small projects where I will be creating pseudo packages, for those who are in the world of marketing and programming. Waiting
to be able to shorten the adaptation curve of some of the professionals that we meet here.

The first package in this project is PyQuality, a simplification of the connection to the Adquality REST API in Python.

I await your feedback!

``` python

pip install py_adquality

from py_adquality.pyquality import adquality_reports
```

Initially the package has two methods, where each one returns a DataFrame. The main method generates a report in Pandas format where you can see all
the data obtained by the AdCuality REST API.

``` python

adquality = reports(username = 'my_email_or_adquality_user', password = 'my_password')
publishers = adquality.report_generator_soi(country = 'CO', report_type = 'publishers')

#You can also choose the methood "report_generator_sov"
#Report options "publishers", "formats", "advertisers", "industries", "sources" or "categories"
```

The second packet gets all the images captured by AdCuality within a given period.

``` python

adquality = reports(username = 'my_email_or_adquality_user', password = 'my_password')
images = adquality.image_extractor(country = 'CO')
```


Very soon, automatic learning modules will be added to exploit the information we have within the platform.

What do you think is the best module we can start with? 