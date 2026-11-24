"""Atajos para escribir esquemas JSON compactos (menos tokens, menos ruido en specs.py)."""


def text(description: str) -> dict:
    return {"type": "string", "description": description}


def num(description: str) -> dict:
    return {"type": "number", "description": description}


def obj(properties: dict, required: list | None = None) -> dict:
    schema = {"type": "object", "properties": properties}
    if required:
        schema["required"] = list(required)
    return schema
