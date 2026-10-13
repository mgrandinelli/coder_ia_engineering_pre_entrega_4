"""Pipeline de ingesta: documentos de data/ -> fragmentos con metadatos -> índice de Pinecone.

Uso: python ingest.py
"""

import hashlib
import logging
import os
import re
from collections.abc import Iterator
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore, VectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config
from schemas import ChunkMetadata
from setup_index import setup_index

log = logging.getLogger("rag.ingest")

DATA_DIR = Path(__file__).parent / "data"
CHUNK_SIZE = 500  # tokens
CHUNK_OVERLAP = 50  # tokens
BATCH_SIZE = 100  # vectores por upsert


def limpiar(texto: str) -> str:
    """Quita el ruido que empeora los embeddings: espacios al final y saltos de línea repetidos."""
    texto = re.sub(r"[ \t]+\n", "\n", texto)
    return re.sub(r"\n{3,}", "\n\n", texto).strip()


def _separar_cabecera(texto: str) -> tuple[dict[str, str], str]:
    """Lee la cabecera `--- clave: valor ---` del archivo y devuelve (campos, cuerpo)."""
    coincidencia = re.match(r"---\n(.*?)\n---\n(.*)", texto, flags=re.DOTALL)
    if not coincidencia:
        return {}, texto
    campos = dict(linea.split(":", 1) for linea in coincidencia.group(1).splitlines() if ":" in linea)
    return {k.strip(): v.strip() for k, v in campos.items()}, coincidencia.group(2)


def cargar_documentos(data_dir: Path = DATA_DIR) -> list[Document]:
    documentos = []
    for ruta in sorted(p for p in Path(data_dir).glob("*") if p.suffix in (".md", ".txt")):
        cabecera, cuerpo = _separar_cabecera(ruta.read_text(encoding="utf-8"))
        documentos.append(
            Document(
                page_content=limpiar(cuerpo),
                metadata={
                    "documento_id": ruta.stem,
                    "fuente": ruta.name,
                    "titulo": cabecera.get("titulo", ruta.stem),
                    "categoria": cabecera.get("categoria", "general"),
                    "etiquetas": [e.strip() for e in cabecera.get("etiquetas", "").split(",") if e.strip()],
                },
            )
        )
    return documentos


def build_splitter() -> RecursiveCharacterTextSplitter:
    try:
        # El tamaño se mide en tokens reales, que es como cobran y limitan los modelos.
        return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base", chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )
    except Exception as exc:
        # Sin el tokenizador (por ejemplo, sin red para descargarlo) se aproxima a 4 caracteres por token.
        log.warning("tiktoken no disponible (%s); se fragmenta por caracteres", type(exc).__name__)
        return RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE * 4, chunk_overlap=CHUNK_OVERLAP * 4)


def fragmentar(documentos: list[Document]) -> list[Document]:
    """Divide con RecursiveCharacterTextSplitter y completa los metadatos de cada fragmento."""
    fragmentos = build_splitter().split_documents(documentos)
    paginas: dict[str, int] = {}
    for fragmento in fragmentos:
        documento_id = fragmento.metadata["documento_id"]
        pagina = paginas[documento_id] = paginas.get(documento_id, 0) + 1
        # El esquema valida que ningún fragmento salga con metadatos incompletos o con otros nombres.
        fragmento.metadata = ChunkMetadata(**fragmento.metadata, pagina=pagina).model_dump()
        huella = hashlib.sha1(fragmento.page_content.encode("utf-8")).hexdigest()[:10]
        # Id determinista: reingerir el mismo contenido sobrescribe el vector, no lo duplica.
        fragmento.id = f"{documento_id}#{pagina}#{huella}"
    return fragmentos


def en_lotes(elementos: list, tamano: int = BATCH_SIZE) -> Iterator[list]:
    for inicio in range(0, len(elementos), tamano):
        yield elementos[inicio : inicio + tamano]


def get_vectorstore() -> VectorStore:
    """PineconeVectorStore sobre el índice y el namespace configurados.

    El texto original de cada fragmento se guarda en el metadato `text`, junto al vector,
    así la búsqueda devuelve el contenido sin consultar otra base de datos.
    Sin PINECONE_API_KEY devuelve un almacén en memoria, para poder probar sin la nube.
    """
    embeddings = config.build_embeddings()
    if not config.pinecone_configurado():
        log.warning("Sin PINECONE_API_KEY: se usa un almacén vectorial en memoria (modo local)")
        return InMemoryVectorStore(embeddings)

    from langchain_pinecone import PineconeVectorStore

    return PineconeVectorStore(
        index_name=config.INDEX_NAME,
        embedding=embeddings,
        namespace=config.NAMESPACE,
        text_key="text",
        pinecone_api_key=os.environ["PINECONE_API_KEY"],
    )


def ingest(
    data_dir: Path = DATA_DIR,
    vectorstore: VectorStore | None = None,
    fragmentos: list[Document] | None = None,
) -> int:
    """Sube los fragmentos al almacén vectorial por lotes y devuelve cuántos subió."""
    if vectorstore is None:
        if config.pinecone_configurado():
            setup_index()  # crea el índice si no existe y verifica la dimensión
        vectorstore = get_vectorstore()

    fragmentos = fragmentos if fragmentos is not None else fragmentar(cargar_documentos(data_dir))
    if not fragmentos:
        log.error("No hay archivos .md o .txt en %s", data_dir)
        return 0

    for numero, lote in enumerate(en_lotes(fragmentos), start=1):
        vectorstore.add_documents(lote, ids=[f.id for f in lote])
        log.info("Lote %d: %d fragmentos subidos", numero, len(lote))
    log.info(
        "Ingesta lista: %d fragmentos de %d documentos, embeddings %s (%d dimensiones)",
        len(fragmentos), len({f.metadata["documento_id"] for f in fragmentos}),
        config.embeddings_model(), config.embeddings_dimension(),
    )
    return len(fragmentos)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
    ingest()
