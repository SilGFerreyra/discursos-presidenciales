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

## Ejemplo: denuncia de corrupción judicial

1. La corrupción en el poder judicial se manifiesta cuando los jueces resuelven en función de sobornos, prebendas o intereses ajenos al derecho, traicionando el mandato de imparcialidad que la sociedad les confía
2. La venalidad de ciertos magistrados y la prevaricación en sus fallos erosionan la confianza ciudadana y convierten la administración de justicia en un mercado de favores y privilegios.
3. El clientelismo, el nepotismo y el tráfico de influencias en la designación y promoción de los cargos judiciales configuran un entramado que subordina la carrera judicial a lealtades políticas y a intereses particulares
4. Las irregularidades en la tramitación de causas, la arbitrariedad en las decisiones y la impunidad de los responsables profundizan el descrédito de los tribunales y debilitan el estado de derecho.

## Prompt que se utilizó para la investigación sobre corrupción

Dado un TEMA o FRASE SEMILLA sobre discurso político, genera EXACTAMENTE 4 oraciones
en español que sirvan como carrier sentences para contrastar ese tema contra un corpus
de chunks de discursos presidenciales de Argentina y Chile mediante similitud de coseno
con embeddings (modelo BGE-M3).

REGLAS DE GENERACIÓN

1. COBERTURA: las 4 oraciones deben girar en torno al mismo tema de la frase semilla,
   abordándolo desde ángulos complementarios (no redundantes entre sí).

2. DIVERSIDAD LÉXICA: cada oración debe reformular el tema con vocabulario distinto.
   No repitas el mismo sustantivo nuclear ni la misma construcción más de una vez.

3. REGLA DEL CONCEPTO EXPLÍCITO:
   - Si la frase semilla contiene "corrupción", "transparencia" o cualquier término
     equivalente del diccionario de variaciones, SOLO UNA de las 4 oraciones puede
     usar ese término explícito (o su familia morfológica).
   - Las otras 3 oraciones deben expresar la misma idea mediante sinónimos,
     perífrasis o variantes del diccionario.
   - Si la frase semilla NO contiene ninguno de esos términos, aplica la misma lógica
     al concepto nuclear que sí aparezca: solo una oración lo nombrará explícitamente.

4. DICCIONARIO DE VARIACIONES (usar preferentemente estos patrones para reformular):
   corromp*, arbitrariedad, agiotaje, asociacion ilicita, camarilla, caudillismo,
   clientelismo, cohech*, comision investigadora, concusion*, *corrup*, dadivas,
   defraudacion, demagog*, despotismo*, dilapida*, empleomania, enriquecimiento ilicito,
   exaccion*, favoritismo, fraud*, irregularidades, malversa*, nepotismo, peculado,
   prebenda, prevarica*, soborn*, trafico influencias, venalidad, vicio*

5. NEUTRALIDAD REFERENCIAL (crítico para el experimento):
   - NO incluyas nombres propios de personas, países, ciudades, regiones, partidos
     políticos, instituciones específicas ni cargos concretos.
   - NO uses deícticos situados ("en nuestro país", "en la región", "este gobierno").
   - Usa sujetos genéricos: "el Estado", "los gobiernos", "las instituciones",
     "los empresarios", "las corporaciones", "los funcionarios", "el poder político",
     "el poder económico", etc., según corresponda al tema.

6. REGISTRO Y LONGITUD:
   - Registro institucional/argumentativo, autocontenido (sin dependencia de contexto).
   - Longitud comparable a chunks de 40-70 tokens (aprox. 25-55 palabras por oración).

7. SALIDA LIMPIA: no agregues encabezados, numeración, viñetas, comillas ni comentarios.
   Devuelve exactamente 4 líneas, una oración por línea.

ENTRADA
FRASE SEMILLA: {{frase}}


```

Después de revisar las frases, introdúzcalas en la app y pulse **Analizar cuatro
carriers**. La sección de cohesión muestra cuánto se parecen entre sí; si una
frase se aleja demasiado del significado común, reformúlela y vuelva a analizar.
