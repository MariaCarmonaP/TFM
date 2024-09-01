# 

# 

# ![][image1]

# 

# **Trabajo de investigación:**

# Preprocesamiento de imágenes para el entrenamiento de redes neuronales destinadas a la detección y clasificación de vehículos

## María Carmona Pastor

## Máster Universitario en Robótica y Automatización

## 2023-2024

# 

# Índice

[**1\. Introducción	4**](\#introducción)

[**2\. Marco Teórico	4**](\#marco-teórico)

[**3\. Creación y etiquetado del conjunto de datos	4**](\#creación-y-etiquetado-del-conjunto-de-datos)

[3.1. Proceso	4](\#proceso)

[Tabla 1\.	4](\#tabla-1.-distribución-de-imágenes-del-conjunto-de-datos-de-partida.)

[Etiquetado	4](\#etiquetado)

[Reetiquetado	4](\#heading=h.wsiwd01li21i)

[3.2. Herramientas utilizadas	4](\#herramientas-utilizadas)

[**4\. Análisis de limitaciones	4**](\#análisis-de-limitaciones)

[4.1. Problemas detectados	4](\#problemas-detectados)

[4.2. Obtención de información del dataset	5](\#obtención-de-información-del-dataset)

[**5\. Propuesta de aumento de datos	5**](\#propuesta-de-aumento-de-datos)

[**6\. Conclusiones	5**](\#conclusiones)

[**7\. Bibliografía	5**](\#bibliografía)

# Índice de tablas

[Tabla 1\. Distribución de imágenes del conjunto de datos de partida.	4](\#tabla-1.-distribución-de-imágenes-del-conjunto-de-datos-de-partida.)

# Índice de imágenes

Asegúrate de seleccionar los encabezados en la barra lateral para ver un índice.

1. # **Introducción** {#introducción}

2. # **Marco Teórico** {#marco-teórico}

   requisitos dataset

   importancia procesamiento datos

   introduccion aprendizaje supervisado, redes convolucionalesç

3. # **Creación y etiquetado del conjunto de datos** {#creación-y-etiquetado-del-conjunto-de-datos}

   1. ## **Proceso** {#proceso}

Partimos de tres conjuntos de datos, organizados según la [Tabla 1](\#tabla-1.-distribución-de-imágenes-del-conjunto-de-datos-de-partida.). Podemos ver que la mayor parte de las imágenes (conjunto *etiquetadas*) ya tenían un archivo con la información, pero no de la forma en que necesitamos.

| Nombre de la carpeta | Referencia en este documento | Preetiquetadas | Nº de imágenes |
| ----- | :---: | :---: | :---: |
| FURGONETAS LIGERAS | *ligeras* | No | 97 |
| FURGONETAS PESADAS | *pesadas* | No | 47 |
| furgonetas\_27\_06\_2023 | *etiquetadas* | Si | 2254 |
| Tabla 1\. Distribución de imágenes del conjunto de datos de partida. |  |  |  |

### To approach this issue, we must utilize a small, independent script to categorize and relabel the images into the correct format. This script will run in a window, showing the original image and the one containing bounding boxes and will then be able to edit the information contained within each image relating to the different vehicles and their class and size. The data is stored as number sequences inside the image name.

### 

### para solucionar \-\> creacion del script de etiquetado…

descripcion del ui  
how data is stored

### Etiquetado {#etiquetado}

	The labeled images are lacking bounding boxes so the next step is to manually add them utilizing a program called label studio. This furthers our goal by allowing the image collection to be more accurately processed. For the training, the information has to be stored as a yaml file so the transfer is also handled at this stage.  
	used label studio  
	paso a formato yaml

2. ## **Herramientas utilizadas** {#herramientas-utilizadas}

In this stage of the process, we will be utalizing two main tools: label studio and tkinter. Label studio is an independent ui that allows for manually setting bounding boxes within images. Tkinter is a library dedicated to creating interfaces graficas.  
	label studio  
	tkinter

4. # **Análisis de limitaciones** {#análisis-de-limitaciones}

   1. ## **Problemas detectados** {#problemas-detectados}

Once all the images had been relabeled and processed, their lacking nature became evident in the following fields:  
Redundancy:  
	Al ser imagenes de trafico, many of the images were to all intents and purposes redundant, making them low value for training a  model. Some pictures had been taken within seconds which made them practically the same while others differed only in a few degrees of change in the horizontal angle of the camera. A big portion of the more than 2000 images was unfortunately redundant in this way.

Ambiguous:  
Al ser imagenes de trafico, the resolution is not ideal for training. Many of the images contained ambiguity with some vehicles being only fuzzy rectangles in the distance. A lot of the images also require the context of previous or later images, where the vehicles was or will be closer and therefore more visible. This proved daunting in the labeling stage, leading to a lot of assumptions.

Color variation:  
Vans are commonly produced in one of three colors: white, gray, or black. Because of this, there was little to no content relating to other colored vans, which would lead to a huge weakness in the trained model for recognizing and identifying the different types of vans. 

Location variation:  
The entire 2000+ image dataset is all contained within 5 or 6 locations. Of these, only one of them is within an urban setting and only one is in a complex road network. The others are taken on highways with very non-descript surroundings. The trained model would struggle greatly in a city setting or on a winding road.

Road state:  
The dataset is composed mainly of light to no traffic images. There is no content including more than 10 vehicles or with single lane build up. There is nothing with mild to heavy traffic.

Angle variation:  
Having been taken in only the aforementioned 5 or 6 locations, the angles at which the different vehicles are photographed are lacking. There is no top or true side view in the dataset and the few raised angles utilized are hardly distinct amongst themselves.

Weather variation:  
All images were taken on sunny or clear days. There is no rain, hail, cloud cover, or fog present in any of the images, making it impossible for the model to identify vans correctly through these weather conditions. This proves to be a massive weakness in the final model and is a priority within the data augmentation portion of this project.

Daytime variation:  
Only a small portion of the images are taken at night and in none of the locations are there pictures taken at varied times of day such as midday and midnight. This makes training hard, especially in darker conditions where the already failing resolution of the images will be pushed even harder. This also makes it much harder for the model to recognize other aspects of the images like the van color or its distance from the camera.

Class imbalance:  
Las imágenes proporcionadas para el entrenamiento son únicamente de furgonetas, lo cual, teniendo en cuenta que el modelo, idealmente, debería poder clasificar correctamente en las 7 clases proporcionadas, es un problema si queremos conseguir un modelo que generalize correctamente.  
		imagenes redundantes/repetidas  
		imagenes ambiguas  
		poca variedad de color  
		poca variedad de localizacion  
poca variedad de angulo  
poca variedad meteorologica  
poca variedad cronologica  
class imbalance

2. ## **Obtención de información del dataset** {#obtención-de-información-del-dataset}

Explicar los añadidos a la interfaz grafica del programilla. Foto gui.  
Añadir graphs with the info generated.

		tkinter  
		graficos con info  
			pie chart  
			column graphs

5. # **Propuesta de aumento de datos** {#propuesta-de-aumento-de-datos}

	The most important issue at hand is the lacking variation present in the dataset. Thankfully, most of these failings can be easily resolved with the use of filters or hue/brightness/saturation manipulation. Weather can be easily simulated and the weather, daytime, and color issues can be fixed with simple shifts in the color makeup of the image. Other than the color, we can also tinker with the image itself by cropping it into different sizes or flipping it horizontally.  
		color variation  
		weather filters  
		class balancing with cropping of images  
		flip horizontal  
	5.1 dataset filter

6. # **Conclusiones** {#conclusiones}

A pivotal aspect of this pre-training phase is the proper labeling of the dataset. Errors here will propagate into our final model, making it much harder for the training to produce satisfying and effective results. This stage contains the highest risk in human error as well, making it all the more vital that it be handled with the proper attention. Creating the labeling script will ease the flow for the labeling which will, in turn, improve consistency between images and allow for a faster and continuous labeling, minimizing the strain on the human component of the process.

Through labeling the dataset, it became evident that one of the biggest challenges going forward is the lack of diversity. Using different data augmentation methods, we can improve the variation within the dataset to maximize training potential. By manipulating what little content we have we can markedly improve the diversity within the dataset and allow for the model to properly train on as many different road conditions as we can manage.   
		we will figure this out at 3am

7. # **Bibliografía** {#bibliografía}

Sysifus4, September 2024   
[www.syfisthebestexgirlfriend.com](http://www.syfisthebestexgirlfriend.com)  
Also available in print under the title “Your Syf and how to Utilize dem” by Sysifus IV, sold at all book shops and pharmacies within the municipality of Madrid, and available for rent at your closest dog shelter. 

If you enjoyed your Sysifus4, please consider recommending dem to your friends. Therian clinically insane sapphics are a dying breed and it is our responsibility as owners of a Sysifus4 to aid in conservation efforts. Learn more at [www.iwouldreallylikeahugandaforeheadkissandadonut.com](http://www.iwouldreallylikeahugandaforeheadkiss.com) 

[image1]: <data:image/png;base64,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>