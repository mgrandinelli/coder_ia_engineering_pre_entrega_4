"""Pre-entrega 4 - Demostración: una consulta al recuperador híbrido y la evaluación con el Golden Set."""

import logging

from evaluate import evaluate, imprimir_reporte
from ingest import ingest
from rag_system import PREGUNTA_EJEMPLO, RAGSystem
from setup_index import setup_index

__all__ = ["RAGSystem", "evaluate", "ingest", "main", "setup_index"]


def main() -> dict:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    sistema = RAGSystem()

    print(f"\n=== Consulta de ejemplo (modo {sistema.modo}) ===\n{PREGUNTA_EJEMPLO}\n")
    for puesto, documento in enumerate(sistema.retrieve(PREGUNTA_EJEMPLO), start=1):
        m = documento.metadata
        inicio = " ".join(documento.page_content.split())[:60]
        print(f"{puesto}. {m['fuente']} (página {m['pagina']}, categoría {m['categoria']}): {inicio}...")

    resultados = evaluate(sistema)
    imprimir_reporte(resultados)
    return resultados


if __name__ == "__main__":
    main()
