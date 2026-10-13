"""Esquemas Pydantic: la interfaz de metadatos de cada fragmento y el formato del Golden Set."""

from pydantic import BaseModel, ConfigDict, Field


class ChunkMetadata(BaseModel):
    """Metadatos que acompañan a cada vector. Un esquema único evita que cada proceso
    invente sus propios nombres de campo y los filtros dejen de funcionar."""

    model_config = ConfigDict(extra="forbid")

    documento_id: str = Field(min_length=1)
    fuente: str = Field(min_length=1)  # archivo de origen
    titulo: str
    pagina: int = Field(ge=1)  # posición del fragmento dentro del documento
    categoria: str = Field(min_length=1)
    etiquetas: list[str] = Field(default_factory=list)


class GoldenItem(BaseModel):
    """Un caso del benchmark: una pregunta y el documento que la responde."""

    model_config = ConfigDict(extra="forbid")

    pregunta: str = Field(min_length=1)
    documento_id_esperado: str = Field(min_length=1)
