# Marketing Science | Admetricks library

Desarrollado por Carlos Trujillo, Marketing Scientist.

## Introduction

Si formas parte del mundo del marketing seguramente has escuchado acerca de Admetricks, Comscore, Lotame o Adcuality. Herramientas que nos entregan datos
los cuales podemos utilizar para realizar distintos tipos de estudios a muchos de los fenomenos a los cuales nos enfretamos en el este rubro.

Cada día, la adopción de la tecnologia es más importante que nunca para poder llevar nuestro analisis un poco más allá de lo convencional.
Esto puede representar un reto muy cumplicado, debido a que muchas de las herramientas que utilizamos día a día no tienen habilitadas sus conectores para todos
los usuarios, dando como resultado que no podamos hacer grandes descargas de data.

Y aquellos usuarios que logran obtener un conector hacia alguna de estas plataformas, le espera un gran camino para entender el conector, su operabilidad y
resultados, con la poca información que existe en internet.

Por ello, decidi iniciar pequeños proyectos donde estaré creando pseudo paquetes, para aquellos que estan en el mundo del marketing y la programación. Esperando
poder acortar la curva de adaptación de algunos de los profesionales que nos encontramos aquí.

Este es mi segundo paquete en este proyecto. Les presento PyMetricks, una simplificación de la conection a el API REST de Admetricks en Python.

¡Espero tu feedback!

``` python

pip install py_metricks

from py_admetricks.pymetricks import admetricks_api
```

Incialmente el paquete cuenta con dos metodos, donde cada uno devuelve un DataFrame. El metodo principal, genera un reporte en formato Pandas donde se ven todos
los datos obtenidos por la API REST de Admetricks.

``` python

admetricks = admetricks_api(username = 'your_mail@enterprise.com', password = 'admetricks_password')
report = admetricks.reports_generator(country='chile', since_date='2021-05-31', device='mobile', ad_type='display')
report.head(10)

#country options 'chile','colombia','argentina','brasil','españa','peru','mexico','honduras','puerto rico','panama','uruguay','costa rica',
#                 'guatemala','ecuador','venezuela','nicaragua','salvador','republica dominicana' or 'paraguay'
```

El segundo paquete obtiene todas las imagenes capturadas por Admetricks dentro de un periodo determinado, en el formato crudo, por tanto veran una captura completa
de toda la pantalla y no solo del anuncio.

``` python

admetricks = admetricks_api(username = 'your_mail@enterprise.com', password = 'admetricks_password')
screenshots = admetricks.screenshots_data(country = 'chile', site = 'facebook', since_date = '2021-01-08', until_date = '2021-01-08')
screenshots.head(10)
```

Muy pronto se estaran añadiendo modulos de aprendizaje automatico para explotar la información que tenemos dentro de la plataforma.

¿Cuales creen que es el mejor modulo con el que podemos empezar?