"""Evaluación del recuperador con un Golden Set: Recall@5, Precision@5 y MRR.

Uso: python evaluate.py
"""

import json
import logging
from pathlib import Path

from langchain_core.documents import Document
from pydantic import TypeAdapter

from rag_system import TOP_K, RAGSystem
from schemas import GoldenItem

GOLDEN_SET = Path(__file__).parent / "golden_set.json"


def cargar_golden_set(ruta: Path = GOLDEN_SET) -> list[GoldenItem]:
    return TypeAdapter(list[GoldenItem]).validate_python(json.loads(Path(ruta).read_text(encoding="utf-8")))


def recall_at_k(recuperados: list[Document], documento_id_esperado: str, k: int = TOP_K) -> float:
    """1 si el documento correcto está entre los k recuperados, 0 si no."""
    return float(any(d.metadata["documento_id"] == documento_id_esperado for d in recuperados[:k]))


def precision_at_k(recuperados: list[Document], documento_id_esperado: str, k: int = TOP_K) -> float:
    """Proporción de los k recuperados que pertenecen al documento correcto."""
    utiles = sum(d.metadata["documento_id"] == documento_id_esperado for d in recuperados[:k])
    return utiles / k


def reciprocal_rank(recuperados: list[Document], documento_id_esperado: str, k: int = TOP_K) -> float:
    """1 / posición del primer fragmento del documento correcto (0 si no aparece)."""
    for posicion, documento in enumerate(recuperados[:k], start=1):
        if documento.metadata["documento_id"] == documento_id_esperado:
            return 1 / posicion
    return 0.0


def evaluate(sistema: RAGSystem | None = None, golden_set: list[GoldenItem] | None = None, k: int = TOP_K) -> dict:
    """Corre el benchmark con el recuperador híbrido y, como referencia, con cada parte por separado."""
    sistema = sistema or RAGSystem()
    golden_set = golden_set if golden_set is not None else cargar_golden_set()
    recuperadores = {
        "Híbrido (BM25 + vectores)": sistema.retriever,
        "Solo léxico (BM25)": sistema.bm25,
        "Solo semántico (vectores)": sistema.vectorial,
    }

    resultados: dict = {"k": k, "modo": sistema.modo, "preguntas": [], "resumen": {}}
    for nombre, recuperador in recuperadores.items():
        recalls, precisiones, rangos = [], [], []
        for item in golden_set:
            recuperados = recuperador.invoke(item.pregunta)[:k]
            recall = recall_at_k(recuperados, item.documento_id_esperado, k)
            precision = precision_at_k(recuperados, item.documento_id_esperado, k)
            recalls.append(recall)
            precisiones.append(precision)
            rangos.append(reciprocal_rank(recuperados, item.documento_id_esperado, k))
            if recuperador is sistema.retriever:
                resultados["preguntas"].append({
                    "pregunta": item.pregunta,
                    "documento_id_esperado": item.documento_id_esperado,
                    "recuperados": [d.metadata["documento_id"] for d in recuperados],
                    "recall": recall,
                    "precision": precision,
                })
        resultados["resumen"][nombre] = {
            "recall": sum(recalls) / len(recalls),
            "precision": sum(precisiones) / len(precisiones),
            "mrr": sum(rangos) / len(rangos),
        }
    return resultados


def imprimir_reporte(resultados: dict) -> None:
    k = resultados["k"]
    print(f"\n=== Evaluación del recuperador (k={k}, modo {resultados['modo']}) ===\n")
    for fila in resultados["preguntas"]:
        print(f"- {fila['pregunta']}")
        print(f"  esperado: {fila['documento_id_esperado']} | recuperados: {', '.join(fila['recuperados'])}")
        print(f"  Recall@{k}: {fila['recall']:.2f} | Precision@{k}: {fila['precision']:.2f}")
    print(f"\n{'Recuperador':<28}{f'Recall@{k}':>10}{f'Precision@{k}':>14}{'MRR':>8}")
    for nombre, metricas in resultados["resumen"].items():
        print(f"{nombre:<28}{metricas['recall']:>10.2f}{metricas['precision']:>14.2f}{metricas['mrr']:>8.2f}")


def main() -> dict:
    resultados = evaluate()
    imprimir_reporte(resultados)
    return resultados


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)-7s %(message)s")
    main()
