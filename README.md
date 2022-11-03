# Simulation-of-Recommendation-Systems

Desarrollaremos entornos simulados para un sistema de recomendación de una tienda virtual de música. 
Elegimos como temática la música porque tenemos la impresión de que resulta mucho más complicado encontrar nueva música que resulte de agrado, que encontrar nuevos libros, películas, etc. Esto se debe a que las descripciones en estos últimos son más útiles a la hora de reconocer un posible interés, que lo que puede llegar a ser en la música.

El objetivo principal de un sistema de recomendación es proponer productos personalizados a usuarios; aprendiendo de su comportamiento y realizando predicciones de sus preferencias para productos particulares. 
Existen tres tipos de recomendación principales: content-based, collaborative filtering-based, y knowledge-based. 

## **Content-based recommendation:**
Los sistemas de recomendación content-based utilizan la descripción de los productos para predecir su utilidad basado en el perfil de un usuario. 
Implementaremos un sistema content-based, enfocandonos en crear un perfil descriptivo para cada usuario.

## **Collaborative Filtering-based (CF) recommendation:**
Los sistemas de recomendación collaborative filtering-based asumen que usuarios con intereses similares, consumirán productos similares.
Implementaremos un memory-based collaborative filtering, que es un collaborative filtering de las primeras generaciones.
Este acercamiento utiliza heurísticas para calcular la similitud entre usuarios o productos. 

## **Knowledge-based recommendation:**
En sistemas de recomendación knowledge-based, las recomendaciones se basan en conocimiento existente 
o reglas sobre las necesidades de los usuarios y las propiedades de los productos. 
Decidimos implementar el sistema knowledge-based utilizando un algoritmo de satisfacción de restricciones (constraint-based recommender).

## **Simulación**

Estos modelos de recomendación necesitan ser probados, para ello simularemos agentes que interactúen con los sistemas.
Simularemos el proceso en el que un usuario decide si quiere oir una canción o no, basándose en sus preferencias. Además el proceso de retroalimentación, donde el 
usuario ofrece una puntuación a una canción luego de interactuar con ella. 
Existirán distintos tipos de agentes, dados por características como que tan importante es la puntuación de una canción al elegirla, o preferencias de géneros o artistas, 
incluso agentes que escojan aleatoriamente. Los atributos de los agentes pueden cambiar en el tiempo.
Los resultados de las simulaciones nos permitirán obtener información del funcionamiento práctico de nuestros sistemas, e incluso realizar cambios que los mejoren.


En conclusión, crearemos, en un inicio, tres sistemas de recomendación; 
uno por cada técnica de recomendación. En cada uno se utilizan diferentes algoritmos de Inteligencia Artificial para obtener las sugerencias.
Esos sistemas se probarán en un entorno con agentes, que simularán la reacción de los usuarios ante las recomendaciones del sistema.
El objetivo es extraer conclusiones del comportamiento y la efectividad de cada sistema, y como reaccionan a diferentes tipos de usuarios.
