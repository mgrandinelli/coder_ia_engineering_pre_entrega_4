"""Configuración leída del entorno y construcción del modelo de embeddings."""

import os

from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings

load_dotenv()

INDEX_NAME = os.getenv("INDEX_NAME") or "atlas-docs"
NAMESPACE = os.getenv("PINECONE_NAMESPACE") or "documentacion"
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD") or "aws"
PINECONE_REGION = os.getenv("PINECONE_REGION") or "us-east-1"
METRIC = "cosine"
SIN_CLAVE = "sin-configurar"  # permite construir los clientes e importar los módulos sin credenciales

# Modelo y dimensión van juntos: el índice se crea con la dimensión del modelo que se va a usar.
EMBEDDINGS = {
    "openai": ("text-embedding-3-small", 1536),
    "local": ("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", 384),
}


def pinecone_configurado() -> bool:
    return bool(os.getenv("PINECONE_API_KEY"))


def embeddings_provider() -> str:
    """'openai' o 'local'. Sin EMBEDDINGS_PROVIDER, usa OpenAI solo si hay clave configurada."""
    elegido = os.getenv("EMBEDDINGS_PROVIDER")
    if elegido in EMBEDDINGS:
        return elegido
    return "openai" if os.getenv("OPENAI_API_KEY") else "local"


def embeddings_model() -> str:
    return EMBEDDINGS[embeddings_provider()][0]


def embeddings_dimension() -> int:
    return EMBEDDINGS[embeddings_provider()][1]


def build_embeddings() -> Embeddings:
    """Único punto donde se crea el modelo de embeddings: lo usan la ingesta y la consulta."""
    if embeddings_provider() == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=embeddings_model(), api_key=os.getenv("OPENAI_API_KEY") or SIN_CLAVE
        )
    from langchain_community.embeddings import FastEmbedEmbeddings

    return FastEmbedEmbeddings(model_name=embeddings_model())
