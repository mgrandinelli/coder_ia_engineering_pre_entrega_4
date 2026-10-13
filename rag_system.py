"""Recuperador híbrido: búsqueda léxica (BM25) y semántica (vectores) combinadas con EnsembleRetriever."""

import logging
import re

from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore, VectorStore

from ingest import cargar_documentos, fragmentar, get_vectorstore, ingest

log = logging.getLogger("rag.system")

TOP_K = 5
# (léxico, semántico). Pesos iguales: con RRF, un peso mayor en una de las partes hace que sus
# cinco resultados queden por delante del mejor resultado de la otra, y el top-5 deja de ser híbrido.
PESOS = (0.5, 0.5)
PREGUNTA_EJEMPLO = "¿Qué significa el código de error ATL-4012?"


def tokenizar(texto: str) -> list[str]:
    """Palabras en minúscula. Conserva los códigos con guion (ATL-4012) como un solo término."""
    return re.findall(r"[\w-]+", texto.lower())


class RAGSystem:
    """Recibe una consulta y devuelve los top-5 fragmentos combinando resultados léxicos y semánticos."""

    def __init__(
        self,
        vectorstore: VectorStore | None = None,
        documentos: list[Document] | None = None,
        k: int = TOP_K,
        pesos: tuple[float, float] = PESOS,
    ) -> None:
        self.k = k
        # BM25 trabaja en memoria sobre los mismos fragmentos que se suben al índice.
        self.documentos = documentos if documentos is not None else fragmentar(cargar_documentos())
        self.vectorstore = vectorstore if vectorstore is not None else get_vectorstore()
        if vectorstore is None and isinstance(self.vectorstore, InMemoryVectorStore):
            # Modo local: el almacén en memoria empieza vacío en cada ejecución.
            ingest(vectorstore=self.vectorstore, fragmentos=self.documentos)

        self.bm25 = BM25Retriever.from_documents(self.documentos, preprocess_func=tokenizar, k=k)
        self.vectorial = self.vectorstore.as_retriever(search_kwargs={"k": k})
        # EnsembleRetriever fusiona los dos rankings con Reciprocal Rank Fusion: usa las posiciones,
        # no los puntajes, porque BM25 y la similitud coseno están en escalas distintas.
        self.retriever = EnsembleRetriever(retrievers=[self.bm25, self.vectorial], weights=list(pesos))

    @property
    def modo(self) -> str:
        return "local" if isinstance(self.vectorstore, InMemoryVectorStore) else "pinecone"

    def retrieve(self, query: str = PREGUNTA_EJEMPLO) -> list[Document]:
        return self.retriever.invoke(query)[: self.k]

    async def aretrieve(self, query: str = PREGUNTA_EJEMPLO) -> list[Document]:
        return (await self.retriever.ainvoke(query))[: self.k]
