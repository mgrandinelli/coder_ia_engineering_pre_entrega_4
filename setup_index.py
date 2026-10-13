"""Inicialización de Pinecone: verifica si el índice existe y lo crea si hace falta (Serverless).

Uso: python setup_index.py
"""

import logging
import os
import time

from pinecone import Pinecone, ServerlessSpec

import config

log = logging.getLogger("rag.setup")


class DimensionMismatchError(Exception):
    """El índice existe pero su dimensión no coincide con la del modelo de embeddings."""


def _leer(objeto, campo):
    return objeto[campo] if isinstance(objeto, dict) else getattr(objeto, campo)


def setup_index(client: Pinecone | None = None, espera_s: float = 1.0) -> str | None:
    """Devuelve el nombre del índice listo para usar, o None si Pinecone no está configurado."""
    if client is None:
        if not config.pinecone_configurado():
            log.error("Falta la variable de entorno PINECONE_API_KEY")
            return None
        client = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

    nombre = config.INDEX_NAME
    dimension = config.embeddings_dimension()

    if not client.has_index(nombre):
        log.info("El índice '%s' no existe; se crea (serverless, dimensión %d, métrica %s)",
                 nombre, dimension, config.METRIC)
        client.create_index(
            name=nombre,
            dimension=dimension,
            metric=config.METRIC,
            spec=ServerlessSpec(cloud=config.PINECONE_CLOUD, region=config.PINECONE_REGION),
        )
        while not _leer(_leer(client.describe_index(nombre), "status"), "ready"):
            time.sleep(espera_s)
        log.info("Índice '%s' listo", nombre)
        return nombre

    existente = _leer(client.describe_index(nombre), "dimension")
    if existente != dimension:
        raise DimensionMismatchError(
            f"El índice '{nombre}' tiene dimensión {existente} y el modelo "
            f"{config.embeddings_model()} genera vectores de {dimension}"
        )
    log.info("El índice '%s' ya existe (dimensión %d); no se vuelve a crear", nombre, existente)
    return nombre


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
    setup_index()
