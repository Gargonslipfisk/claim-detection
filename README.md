# claim-detection

***** __A fecha del 15 de abril de 2021__ *****

Se propone una solución para problema de clasificación binaria sobre lenguaje natural

En vignettes puede encontrarse un notebook interpretable desde Colab donde se detallan los pasos seguidos para el tratamiento de datos y posterior entrenamiento del modelo.

Acompañan a la entrega algunas funciones de utilidades, junto a su correspondientes test unitarios, con el ánimo de evaluar las destrezas como desarrollador.

Referencias a estado del arte podemos encontrarlo en: [Papers with code: BERT for Evidence Retrieval and Claim Verification ](https://paperswithcode.com/paper/bert-for-evidence-retrieval-and-claim)

Trabajo a futuro: 
* modificar la función de pérdida a optimizar para que sea sensible a desbalanceo
* introducir un bias como inicializador de la capa de salida
* validación cruzada de datasets balanceados
* consecutivas interacciones para entender y refinar el aprendizaje del modelo. Recomiendo: [Dodrio](https://github.com/poloclub/dodrio)
