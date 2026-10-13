"""Tests de la pre-entrega 4. No usan la red ni claves: Pinecone y los embeddings son dobles.

Ejecutar con: pytest -v
"""

import json
import logging
from types import SimpleNamespace as NS

import pytest
import tiktoken
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.vectorstores import InMemoryVectorStore
from pydantic import ValidationError

import config
import evaluate as evaluacion
import ingest as ingesta
import main
import setup_index as setup
from rag_system import RAGSystem, tokenizar
from schemas import ChunkMetadata, GoldenItem

DOCUMENTOS = ["api", "arquitectura", "despliegues", "incidentes", "observabilidad", "seguridad"]


class PineconeFalso:
    """Imita la parte del cliente de Pinecone que usa setup_index."""

    def __init__(self, indices=None):
        self.indices = dict(indices or {})  # nombre -> dimensión
        self.creados = []

    def has_index(self, nombre):
        return nombre in self.indices

    def create_index(self, name, dimension, metric, spec):
        self.creados.append(NS(name=name, dimension=dimension, metric=metric, spec=spec))
        self.indices[name] = dimension

    def describe_index(self, nombre):
        return NS(dimension=self.indices[nombre], status={"ready": True})


class AlmacenEspia:
    """Registra cada llamada a add_documents para verificar los lotes."""

    def __init__(self):
        self.lotes = []

    def add_documents(self, documentos, ids):
        self.lotes.append((documentos, ids))


def doc(documento_id: str) -> Document:
    return Document(page_content=documento_id, metadata={"documento_id": documento_id})


@pytest.fixture
def sin_claves(monkeypatch):
    for variable in ("PINECONE_API_KEY", "OPENAI_API_KEY", "EMBEDDINGS_PROVIDER"):
        monkeypatch.delenv(variable, raising=False)


@pytest.fixture
def sistema():
    """Recuperador híbrido completo sobre un almacén en memoria con embeddings falsos."""
    fragmentos = ingesta.fragmentar(ingesta.cargar_documentos())
    almacen = InMemoryVectorStore(DeterministicFakeEmbedding(size=32))
    ingesta.ingest(vectorstore=almacen, fragmentos=fragmentos)
    return RAGSystem(vectorstore=almacen, documentos=fragmentos)


# --- Configuración e infraestructura (Pinecone) ---


def test_setup_crea_el_indice_serverless_si_no_existe(monkeypatch, sin_claves):
    monkeypatch.setenv("OPENAI_API_KEY", "clave-de-prueba")
    cliente = PineconeFalso()

    nombre = setup.setup_index(client=cliente, espera_s=0)

    creado = cliente.creados[0]
    assert nombre == creado.name == config.INDEX_NAME
    assert (creado.dimension, creado.metric) == (1536, "cosine")  # text-embedding-3-small
    assert (creado.spec.cloud, creado.spec.region) == (config.PINECONE_CLOUD, config.PINECONE_REGION)


def test_setup_no_vuelve_a_crear_un_indice_existente(sin_claves, caplog):
    cliente = PineconeFalso({config.INDEX_NAME: config.embeddings_dimension()})

    with caplog.at_level(logging.INFO, logger="rag.setup"):
        setup.setup_index(client=cliente)

    assert cliente.creados == []
    assert "ya existe" in caplog.text


def test_setup_detecta_el_mismatch_de_dimensiones(sin_claves):
    cliente = PineconeFalso({config.INDEX_NAME: 768})

    with pytest.raises(setup.DimensionMismatchError, match="768"):
        setup.setup_index(client=cliente)


def test_setup_sin_clave_informa_y_no_se_rompe(sin_claves, caplog):
    with caplog.at_level(logging.ERROR, logger="rag.setup"):
        assert setup.setup_index() is None

    assert "Falta la variable de entorno PINECONE_API_KEY" in caplog.text


def test_la_dimension_acompana_al_modelo_de_embeddings(monkeypatch, sin_claves):
    assert (config.embeddings_provider(), config.embeddings_dimension()) == ("local", 384)

    monkeypatch.setenv("OPENAI_API_KEY", "clave-de-prueba")
    assert (config.embeddings_model(), config.embeddings_dimension()) == ("text-embedding-3-small", 1536)


def test_las_claves_no_estan_en_el_repositorio():
    base = ingesta.DATA_DIR.parent
    plantilla = (base / ".env.example").read_text()

    for variable in ("PINECONE_API_KEY", "OPENAI_API_KEY"):
        assert f"{variable}=\n" in plantilla
    assert "INDEX_NAME=" in plantilla
    assert ".env" in (base / ".gitignore").read_text().split()


# --- Pipeline de ingesta y metadatos ---


def test_carga_los_documentos_con_los_metadatos_de_la_cabecera():
    documentos = ingesta.cargar_documentos()
    api = documentos[0]

    assert [d.metadata["documento_id"] for d in documentos] == DOCUMENTOS
    assert api.metadata["fuente"] == "api.md"
    assert api.metadata["categoria"] == "referencia"
    assert api.metadata["etiquetas"] == ["api", "errores", "idempotencia", "webhooks"]
    assert not api.page_content.startswith("---")  # la cabecera no forma parte del texto


def test_cada_fragmento_cumple_el_esquema_de_metadatos():
    fragmentos = ingesta.fragmentar(ingesta.cargar_documentos())

    for fragmento in fragmentos:
        metadatos = ChunkMetadata(**fragmento.metadata)
        assert metadatos.pagina >= 1
    paginas_api = [f.metadata["pagina"] for f in fragmentos if f.metadata["documento_id"] == "api"]
    assert paginas_api == list(range(1, len(paginas_api) + 1))


def test_el_esquema_de_metadatos_rechaza_campos_con_otro_nombre():
    with pytest.raises(ValidationError):
        ChunkMetadata(documento_id="a", fuente="a.md", titulo="A", pagina=1, categoria="x", source="a.md")


def test_los_fragmentos_respetan_el_tamano_en_tokens():
    tokens = tiktoken.get_encoding("cl100k_base")
    fragmentos = ingesta.fragmentar(ingesta.cargar_documentos())

    assert all(len(tokens.encode(f.page_content)) <= ingesta.CHUNK_SIZE for f in fragmentos)
    assert len(fragmentos) > len(DOCUMENTOS)  # los documentos no entran en un solo fragmento


def test_los_ids_son_deterministas_y_unicos():
    primera = [f.id for f in ingesta.fragmentar(ingesta.cargar_documentos())]
    segunda = [f.id for f in ingesta.fragmentar(ingesta.cargar_documentos())]

    assert primera == segunda
    assert len(set(primera)) == len(primera)
    assert primera[0].startswith("api#1#")


def test_ingest_sube_en_lotes_de_100():
    fragmentos = [Document(page_content=str(i), id=str(i), metadata={"documento_id": "x"}) for i in range(250)]
    almacen = AlmacenEspia()

    subidos = ingesta.ingest(vectorstore=almacen, fragmentos=fragmentos)

    assert subidos == 250
    assert [len(documentos) for documentos, _ in almacen.lotes] == [100, 100, 50]
    assert almacen.lotes[0][1][:2] == ["0", "1"]


def test_ingest_guarda_el_texto_y_la_fuente_junto_al_vector():
    almacen = InMemoryVectorStore(DeterministicFakeEmbedding(size=32))
    ingesta.ingest(vectorstore=almacen)

    recuperado = almacen.similarity_search("cobros", k=1)[0]

    assert recuperado.page_content
    assert {"fuente", "pagina", "categoria", "etiquetas", "documento_id"} <= set(recuperado.metadata)


def test_ingest_sin_archivos_no_falla(tmp_path):
    assert ingesta.ingest(data_dir=tmp_path, vectorstore=AlmacenEspia()) == 0


# --- Recuperador híbrido ---


def test_rag_system_encapsula_un_ensemble_de_bm25_y_vectores(sistema):
    assert isinstance(sistema.retriever, EnsembleRetriever)
    assert sistema.retriever.retrievers == [sistema.bm25, sistema.vectorial]
    assert isinstance(sistema.bm25, BM25Retriever)
    assert sistema.retriever.weights == [0.5, 0.5]


def test_devuelve_los_top_5_sin_repetidos(sistema):
    recuperados = sistema.retrieve("¿Qué significa el código de error ATL-4012?")

    assert len(recuperados) == 5
    assert len({d.page_content for d in recuperados}) == 5


def test_la_parte_lexica_encuentra_un_codigo_exacto(sistema):
    primero = sistema.bm25.invoke("ATL-4012")[0]

    assert primero.metadata["documento_id"] == "api"
    assert "ATL-4012" in primero.page_content
    # Los embeddings falsos no entienden el texto: si llega al resultado híbrido es gracias a BM25.
    assert primero.page_content in [d.page_content for d in sistema.retrieve("ATL-4012")]


def test_tokenizar_conserva_los_codigos_con_guion():
    assert tokenizar("Error ATL-4012 del banco") == ["error", "atl-4012", "del", "banco"]


@pytest.mark.asyncio
async def test_aretrieve_es_asincrono_y_devuelve_lo_mismo(sistema):
    consulta = "¿En qué días y horarios se puede desplegar a producción?"

    assert await sistema.aretrieve(consulta) == sistema.retrieve(consulta)


def test_sin_pinecone_el_sistema_trabaja_en_modo_local(monkeypatch, sin_claves):
    monkeypatch.setattr(config, "build_embeddings", lambda: DeterministicFakeEmbedding(size=32))

    sistema = RAGSystem()

    assert sistema.modo == "local"
    assert len(sistema.retrieve()) == 5


# --- Evaluación ---


def test_el_golden_set_tiene_5_preguntas_con_documentos_que_existen():
    golden = evaluacion.cargar_golden_set()
    crudo = json.loads(evaluacion.GOLDEN_SET.read_text())

    assert len(golden) == 5
    assert set(crudo[0]) == {"pregunta", "documento_id_esperado"}
    assert {g.documento_id_esperado for g in golden} <= set(DOCUMENTOS)


def test_recall_at_k():
    recuperados = [doc("a"), doc("b"), doc("c"), doc("d"), doc("e"), doc("x")]

    assert evaluacion.recall_at_k(recuperados, "c") == 1.0
    assert evaluacion.recall_at_k(recuperados, "x") == 0.0  # está, pero fuera del top 5
    assert evaluacion.recall_at_k(recuperados, "z") == 0.0


def test_precision_at_k():
    recuperados = [doc("a"), doc("a"), doc("b"), doc("c"), doc("d")]

    assert evaluacion.precision_at_k(recuperados, "a") == 0.4
    assert evaluacion.precision_at_k(recuperados, "b") == 0.2
    assert evaluacion.precision_at_k(recuperados, "z") == 0.0


def test_reciprocal_rank():
    recuperados = [doc("a"), doc("b"), doc("c")]

    assert evaluacion.reciprocal_rank(recuperados, "a") == 1.0
    assert evaluacion.reciprocal_rank(recuperados, "c") == pytest.approx(1 / 3)
    assert evaluacion.reciprocal_rank(recuperados, "z") == 0.0


def test_evaluate_calcula_las_metricas_de_los_tres_recuperadores(sistema):
    resultados = evaluacion.evaluate(sistema)

    assert len(resultados["preguntas"]) == 5
    assert list(resultados["resumen"]) == [
        "Híbrido (BM25 + vectores)", "Solo léxico (BM25)", "Solo semántico (vectores)",
    ]
    for metricas in resultados["resumen"].values():
        assert 0 <= metricas["precision"] <= metricas["recall"] <= 1


def test_el_reporte_se_imprime_en_consola(sistema, capsys):
    evaluacion.imprimir_reporte(evaluacion.evaluate(sistema))

    impreso = capsys.readouterr().out
    assert "Recall@5" in impreso and "Precision@5" in impreso
    assert "Híbrido (BM25 + vectores)" in impreso


def test_un_golden_set_mal_formado_se_rechaza():
    with pytest.raises(ValidationError):
        GoldenItem(pregunta="¿...?", documento_esperado="api")


def test_main_corre_sin_claves(monkeypatch, sin_claves, capsys):
    monkeypatch.setattr(config, "build_embeddings", lambda: DeterministicFakeEmbedding(size=32))

    resultados = main.main()

    assert resultados["modo"] == "local"
    assert "Consulta de ejemplo (modo local)" in capsys.readouterr().out
