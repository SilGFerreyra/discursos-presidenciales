# Cómo redactar las cuatro carrier sentences

Las carriers son **cuatro formulaciones del mismo concepto**. La app las utiliza
para buscar expresiones cercanas en los discursos, no como cuatro categorías
independientes ni como una lista de palabras clave.

## Recomendaciones

- Exprese un concepto común desde ángulos complementarios.
- Utilice oraciones completas y autocontenidas, de aproximadamente 25–55 palabras.
- Varíe el vocabulario sin cambiar el significado objetivo.
- Evite nombres de presidentes, países o instituciones concretas si busca comparar
  el mismo concepto a lo largo de todo el corpus.
- No use referencias ambiguas como «este gobierno» o «en nuestra región».
- Revise los resultados: cercanía semántica no equivale automáticamente a una
  clasificación histórica correcta.

## Ejemplo: educación pública

1. El acceso de toda la población a una enseñanza pública de calidad permite
   ampliar los conocimientos y reducir las desigualdades sociales.
2. La creación y el sostenimiento de escuelas favorecen la formación de las
   personas y ofrecen oportunidades de aprendizaje a distintos sectores sociales.
3. La preparación de docentes y la mejora de los establecimientos educativos
   fortalecen la enseñanza y contribuyen al desarrollo de la comunidad.
4. Las políticas que promueven la alfabetización y la formación de la ciudadanía
   buscan extender los beneficios de la educación al conjunto de la población.

## Prompt reutilizable

```text
Genera exactamente cuatro oraciones en español para buscar el siguiente concepto
en discursos políticos mediante embeddings semánticos:

CONCEPTO: [escriba aquí el concepto]

Las cuatro oraciones deben expresar el mismo concepto desde perspectivas
complementarias, con vocabulario diverso y sin nombres propios ni referencias
a un país o gobierno particular. Deben ser autocontenidas y tener entre 25 y 55
palabras cada una. Devuelve solo cuatro líneas, una oración por línea.
```

Después de revisar las frases, introdúzcalas en la app y pulse **Analizar cuatro
carriers**. La sección de cohesión muestra cuánto se parecen entre sí; si una
frase se aleja demasiado del significado común, reformúlela y vuelva a analizar.
