# Pre-entrega 4: Sistema RAG escalable en la nube con Pinecone

Curso AI Engineering (Coderhouse), módulo 4.

Un módulo de recuperación escalable. Sube documentos técnicos a un índice de Pinecone Serverless con metadatos, los recupera con un buscador híbrido (BM25 + vectores) y mide la calidad de la recuperación con un Golden Set.

```python
sistema = RAGSystem()
documentos = sistema.retrieve("¿Qué significa el código de error ATL-4012?")   # top-5
```

## Cómo ejecutarlo

Requiere Python 3.12 o superior.

```bash
git clone https://github.com/mgrandinelli/coder_ia_engineering_pre_entrega_4.git
cd coder_ia_engineering_pre_entrega_4

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env      # y completar PINECONE_API_KEY, OPENAI_API_KEY e INDEX_NAME
```

## Pasos para replicar el índice de Pinecone

1. Crear una cuenta en Pinecone y generar una clave de API. El plan gratuito alcanza: permite índices Serverless.
2. Completar el archivo `.env`:

   ```
   PINECONE_API_KEY=...
   OPENAI_API_KEY=...
   INDEX_NAME=atlas-docs
   ```

3. Crear el índice:

   ```bash
   python setup_index.py
   ```

   El script verifica si el índice `INDEX_NAME` existe. Si no existe, lo crea en modo Serverless (nube `aws`, región `us-east-1`) con métrica coseno y 1536 dimensiones, que es la dimensión de `text-embedding-3-small`. Si ya existe, no lo toca.

4. Subir los documentos:

   ```bash
   python ingest.py
   ```

   Carga los 6 documentos de `data/`, los divide en 12 fragmentos y los sube al namespace `documentacion`.

5. Probar la recuperación y la evaluación:

   ```bash
   python main.py        # una consulta de ejemplo y el reporte de métricas
   python evaluate.py    # solo el reporte de métricas
   ```

El índice queda con esta configuración:

| Parámetro | Valor |
|---|---|
| Tipo | Serverless (`aws`, `us-east-1`) |
| Dimensión | 1536 (`text-embedding-3-small`) |
| Métrica | Coseno |
| Namespace | `documentacion` |
| Vectores | 12, uno por fragmento |

## Variables de entorno

| Variable | Obligatoria | Qué es |
|---|---|---|
| `PINECONE_API_KEY` | Para usar Pinecone | Clave de la API de Pinecone |
| `OPENAI_API_KEY` | Para embeddings de OpenAI | Clave de la API de OpenAI |
| `INDEX_NAME` | No (por defecto `atlas-docs`) | Nombre del índice |
| `PINECONE_NAMESPACE` | No (por defecto `documentacion`) | Namespace donde se guardan los vectores |
| `PINECONE_CLOUD`, `PINECONE_REGION` | No (por defecto `aws`, `us-east-1`) | Ubicación del índice Serverless |
| `EMBEDDINGS_PROVIDER` | No | `openai` o `local`. Vacío: `openai` si hay `OPENAI_API_KEY`, `local` en caso contrario |

Las claves se leen del archivo `.env`, que está en `.gitignore`. Al repositorio solo se sube `.env.example`, sin valores.

## Archivos

| Archivo | Contenido |
|---|---|
| `data/` | Dataset: 6 documentos Markdown de la plataforma de pagos ficticia Atlas |
| `config.py` | Variables de entorno y modelo de embeddings |
| `setup_index.py` | Verifica si el índice existe y lo crea si hace falta |
| `ingest.py` | Pipeline de ingesta: carga, chunking, metadatos y subida por lotes |
| `rag_system.py` | Clase `RAGSystem`, que encapsula el `EnsembleRetriever` |
| `evaluate.py` | Recall@5, Precision@5 y MRR sobre el Golden Set |
| `golden_set.json` | Benchmark: 5 preguntas con su documento esperado |
| `schemas.py` | Esquemas Pydantic: metadatos de cada fragmento y casos del Golden Set |
| `main.py` | Demostración: una consulta y el reporte de evaluación |
| `test_main.py` | 28 tests, sin red ni claves |

## Arquitectura y decisiones de diseño

El sistema tiene dos flujos:

- **Ingesta:** archivos de `data/` → limpieza → `RecursiveCharacterTextSplitter` → metadatos validados → embeddings → upsert por lotes en Pinecone
- **Consulta:** pregunta → BM25 (léxico) y Pinecone (semántico) en paralelo → `EnsembleRetriever` fusiona los dos rankings → top-5 fragmentos

### Setup de Pinecone (`setup_index.py`)

`setup_index()` hace tres cosas:

1. Consulta si el índice existe con `has_index`.
2. Si no existe, lo crea con `ServerlessSpec` y espera a que esté listo.
3. Si existe, compara su dimensión con la del modelo de embeddings. Si no coinciden, lanza `DimensionMismatchError` antes de intentar subir nada.

La dimensión no está escrita a mano: sale de una tabla en `config.py` que asocia cada modelo con la suya (1536 para `text-embedding-3-small`).

### Pipeline de ingesta (`ingest.py`)

1. `cargar_documentos()` lee cada archivo y su cabecera, que declara título, categoría y etiquetas.
2. `fragmentar()` divide con `RecursiveCharacterTextSplitter`, con 500 tokens por fragmento y 50 de solapamiento, medidos con `tiktoken`.
3. Cada fragmento recibe sus metadatos, validados con el esquema Pydantic `ChunkMetadata`.
4. `ingest()` sube los fragmentos con `PineconeVectorStore`, en lotes de 100.

Metadatos de cada vector:

| Campo | Ejemplo | Para qué sirve |
|---|---|---|
| `text` | El contenido del fragmento | Devolver el texto sin consultar otra base de datos |
| `fuente` | `api.md` | Citar el origen |
| `pagina` | `2` | Posición del fragmento dentro del documento |
| `categoria` | `referencia` | Filtrar la búsqueda por tipo de documento |
| `etiquetas` | `["api", "errores"]` | Filtrar por tema |
| `documento_id` | `api` | Identificar el documento en la evaluación |
| `titulo` | `API pública de la plataforma Atlas` | Mostrar el resultado |

Decisiones de la ingesta:

- **El texto original va en los metadatos.** `PineconeVectorStore` lo guarda en el campo `text`, junto al vector.
- **Esquema de metadatos estricto.** `ChunkMetadata` rechaza campos con otros nombres, para que todos los procesos escriban los mismos campos y los filtros no fallen.
- **Namespace.** Todos los vectores van al namespace `documentacion`, separados de otros datos que comparta el índice.
- **Ids deterministas.** Cada vector se identifica con `documento#página#hash del contenido`; volver a ingerir sobrescribe en lugar de duplicar.

### Recuperador híbrido (`rag_system.py`)

```python
self.bm25 = BM25Retriever.from_documents(self.documentos, preprocess_func=tokenizar, k=k)
self.vectorial = self.vectorstore.as_retriever(search_kwargs={"k": k})
self.retriever = EnsembleRetriever(retrievers=[self.bm25, self.vectorial], weights=list(pesos))  # (0.5, 0.5)
```

- **BM25** busca coincidencias exactas de términos. Es la parte que encuentra códigos y nombres propios como `ATL-4012` o `X-Atlas-Idempotency-Key`. El tokenizador conserva los códigos con guion como un solo término.
- **La búsqueda vectorial** en Pinecone encuentra fragmentos por significado, aunque usen otras palabras.
- **`EnsembleRetriever`** fusiona los dos rankings con Reciprocal Rank Fusion: usa la posición de cada fragmento y no su puntaje, porque BM25 y la similitud coseno están en escalas distintas.
- **Pesos iguales.** Con un peso mayor para una de las partes, sus cinco resultados quedaban por delante del mejor resultado de la otra y el top-5 dejaba de ser híbrido.

`RAGSystem.retrieve(query)` devuelve los 5 mejores fragmentos; `aretrieve(query)` es la versión asíncrona.

### Modo local

Sin `PINECONE_API_KEY`, el sistema reemplaza Pinecone por un almacén vectorial en memoria, y sin `OPENAI_API_KEY` usa un modelo de embeddings local de 384 dimensiones. El resto del código es el mismo. Sirve para ejecutar el recuperador y la evaluación sin cuentas en la nube; el log avisa cuando está activo.

## Evaluación

`golden_set.json` tiene 5 preguntas, cada una con el documento que la responde:

```json
{"pregunta": "¿Qué significa el código de error ATL-4012?", "documento_id_esperado": "api"}
```

`evaluate.py` ejecuta cada pregunta y calcula:

- **Recall@5:** 1 si el documento correcto está entre los 5 fragmentos recuperados, 0 si no.
- **Precision@5:** proporción de los 5 fragmentos recuperados que pertenecen al documento correcto.
- **MRR:** 1 dividido por la posición del primer fragmento correcto. Se agregó porque distingue si el acierto llega primero o quinto.

Además del recuperador híbrido, el reporte evalúa cada parte por separado, como referencia.

### Resumen de resultados

Resultados de `python evaluate.py` en modo local (almacén en memoria y embeddings locales `paraphrase-multilingual-MiniLM-L12-v2`):

| Recuperador | Recall@5 | Precision@5 | MRR |
|---|---|---|---|
| Híbrido (BM25 + vectores) | 1,00 | 0,36 | 0,77 |
| Solo léxico (BM25) | 1,00 | 0,32 | 0,90 |
| Solo semántico (vectores) | 1,00 | 0,32 | 0,77 |

Lectura de los resultados:

- **El documento correcto aparece siempre en el top-5.** El Recall@5 es 1,00 en los tres casos. Con solo 12 fragmentos en total, esta métrica es fácil de cumplir y no alcanza para comparar recuperadores.
- **El híbrido tiene la mejor precisión,** 0,36 contra 0,32. Cada documento ocupa 2 fragmentos, así que el máximo posible de Precision@5 es 0,40.
- **BM25 ubica antes el primer acierto** (MRR 0,90 contra 0,77). Las preguntas del benchmark comparten palabras con los documentos, y el modelo de embeddings local es chico.
- Estos valores corresponden al modo local. Con `text-embedding-3-small` y el índice de Pinecone, la parte semántica debería mejorar; hay que volver a correr `evaluate.py` para medirlo.

### Salida completa en consola

```
=== Consulta de ejemplo (modo local) ===
¿Qué significa el código de error ATL-4012?

1. api.md (página 2, categoría referencia): ## Códigos de error Los errores tienen un código propio de A...
2. api.md (página 1, categoría referencia): # Plataforma Atlas: API pública ## Autenticación Todas las s...
3. incidentes.md (página 2, categoría operaciones): La prioridad es restablecer el servicio, no encontrar la cau...
4. despliegues.md (página 2, categoría operaciones): ## Ventanas de despliegue Los despliegues a producción se ha...
5. seguridad.md (página 1, categoría seguridad): # Plataforma Atlas: seguridad ## Datos de tarjetas Los datos...

=== Evaluación del recuperador (k=5, modo local) ===

- ¿Qué significa el código de error ATL-4012?
  esperado: api | recuperados: api, api, incidentes, despliegues, seguridad
  Recall@5: 1.00 | Precision@5: 0.40
- ¿Cada cuánto se renuevan las claves de acceso de los servicios?
  esperado: seguridad | recuperados: seguridad, api, seguridad, observabilidad, observabilidad
  Recall@5: 1.00 | Precision@5: 0.40
- ¿En qué días y horarios se puede desplegar a producción?
  esperado: despliegues | recuperados: incidentes, despliegues, seguridad, observabilidad, observabilidad
  Recall@5: 1.00 | Precision@5: 0.20
- ¿Quién atiende una alerta si la guardia primaria no responde?
  esperado: incidentes | recuperados: incidentes, incidentes, arquitectura, seguridad, api
  Recall@5: 1.00 | Precision@5: 0.40
- ¿Dónde se ve en qué servicio se demoró una solicitud lenta?
  esperado: observabilidad | recuperados: incidentes, seguridad, observabilidad, api, observabilidad
  Recall@5: 1.00 | Precision@5: 0.40

Recuperador                   Recall@5   Precision@5     MRR
Híbrido (BM25 + vectores)         1.00          0.36    0.77
Solo léxico (BM25)                1.00          0.32    0.90
Solo semántico (vectores)         1.00          0.32    0.77
```

## Tests

`test_main.py` tiene 28 tests. No usan la red ni claves: el cliente de Pinecone se reemplaza por un doble, los embeddings por `DeterministicFakeEmbedding` de LangChain y el índice por un almacén en memoria.

```bash
pytest -v
```

Cubren los cuatro criterios de la rúbrica:

- **Configuración e infraestructura:** el índice se crea Serverless con la dimensión del modelo, no se vuelve a crear si existe, y un índice con otra dimensión se detecta; las claves no están en el repositorio.
- **Ingesta y metadatos:** carga con cabecera, esquema de metadatos, tamaño de los fragmentos en tokens, ids deterministas, subida en lotes de 100.
- **Recuperador híbrido:** `RAGSystem` encapsula un `EnsembleRetriever` de BM25 y vectores, devuelve 5 fragmentos sin repetir, y un código exacto llega al resultado gracias a BM25.
- **Evaluación:** el Golden Set tiene 5 preguntas válidas; Recall@5, Precision@5 y MRR se calculan bien; el reporte se imprime en consola.

## Reporte de tests

Salida de `pytest -v`:

```
============================= test session starts ==============================
collecting ... collected 28 items

test_main.py::test_setup_crea_el_indice_serverless_si_no_existe PASSED   [  3%]
test_main.py::test_setup_no_vuelve_a_crear_un_indice_existente PASSED    [  7%]
test_main.py::test_setup_detecta_el_mismatch_de_dimensiones PASSED       [ 10%]
test_main.py::test_setup_sin_clave_informa_y_no_se_rompe PASSED          [ 14%]
test_main.py::test_la_dimension_acompana_al_modelo_de_embeddings PASSED  [ 17%]
test_main.py::test_las_claves_no_estan_en_el_repositorio PASSED          [ 21%]
test_main.py::test_carga_los_documentos_con_los_metadatos_de_la_cabecera PASSED [ 25%]
test_main.py::test_cada_fragmento_cumple_el_esquema_de_metadatos PASSED  [ 28%]
test_main.py::test_el_esquema_de_metadatos_rechaza_campos_con_otro_nombre PASSED [ 32%]
test_main.py::test_los_fragmentos_respetan_el_tamano_en_tokens PASSED    [ 35%]
test_main.py::test_los_ids_son_deterministas_y_unicos PASSED             [ 39%]
test_main.py::test_ingest_sube_en_lotes_de_100 PASSED                    [ 42%]
test_main.py::test_ingest_guarda_el_texto_y_la_fuente_junto_al_vector PASSED [ 46%]
test_main.py::test_ingest_sin_archivos_no_falla PASSED                   [ 50%]
test_main.py::test_rag_system_encapsula_un_ensemble_de_bm25_y_vectores PASSED [ 53%]
test_main.py::test_devuelve_los_top_5_sin_repetidos PASSED               [ 57%]
test_main.py::test_la_parte_lexica_encuentra_un_codigo_exacto PASSED     [ 60%]
test_main.py::test_tokenizar_conserva_los_codigos_con_guion PASSED       [ 64%]
test_main.py::test_aretrieve_es_asincrono_y_devuelve_lo_mismo PASSED     [ 67%]
test_main.py::test_sin_pinecone_el_sistema_trabaja_en_modo_local PASSED  [ 71%]
test_main.py::test_el_golden_set_tiene_5_preguntas_con_documentos_que_existen PASSED [ 75%]
test_main.py::test_recall_at_k PASSED                                    [ 78%]
test_main.py::test_precision_at_k PASSED                                 [ 82%]
test_main.py::test_reciprocal_rank PASSED                                [ 85%]
test_main.py::test_evaluate_calcula_las_metricas_de_los_tres_recuperadores PASSED [ 89%]
test_main.py::test_el_reporte_se_imprime_en_consola PASSED               [ 92%]
test_main.py::test_un_golden_set_mal_formado_se_rechaza PASSED           [ 96%]
test_main.py::test_main_corre_sin_claves PASSED                          [100%]

============================== 28 passed in 0.73s ==============================
```
