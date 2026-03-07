# Contexto del Proyecto: DARDE/DARDO Automator para ONG (ACH)

## 1. Objetivo Principal
Somos un equipo participando en una hackatón. Estamos desarrollando un pipeline automatizado (backend) en Python para una ONG (ACH). El sistema procesa documentos de desempleo españoles (DARDE/Madrid, DARDO/Cataluña) subidos por usuarios vulnerables.
El objetivo es extraer fechas clave mediante IA (Vision Language Models) y evaluar si el documento es válido para entrar en un programa de empleabilidad, eliminando el cuello de botella de la revisión manual.

## 2. Restricciones Técnicas y de Negocio (¡MUY IMPORTANTE!)
- **Privacidad (RGPD):** CERO extracción de PII (Nombres, DNI, Direcciones). Solo nos interesan las fechas. Las imágenes se procesan en memoria y se descartan.
- **Formato de Salida:** Todas las extracciones de la IA deben ser parseadas a un JSON estricto.
- **Fail-Fast:** El sistema debe identificar rápido si la imagen es ilegible o no es un documento válido antes de intentar extraer fechas.

## 3. Lógica de Validación (Core Business Rule)
Para que un documento sea válido, debe cumplir la siguiente regla matemática con respecto a la fecha de inicio ficticia del programa (`2025-03-01`):
`Fecha de Inscripción Inicial < 2025-03-01 < Primera Fecha de Renovación`

## 4. Stack Tecnológico
- **Lenguaje:** Python 3.10+
- **IA/Inferencia:** Ollama local (ej. `qwen-vl` o `llama3.2-vision`) para extracción zero-shot, o fallback a APIs cloud ligeras.
- **Librerías clave esperadas:** `Pydantic` (para estructurar el output del LLM), `Pillow` (manejo de imágenes base64), `python-dateutil` (para normalizar fechas difusas).

## 5. Estructura del Pipeline que vamos a construir
1. `ingestion.py`: Recibe la imagen (simulada desde una carpeta local).
2. `classifier.py`: Verifica si la imagen tiene calidad suficiente y parece un documento oficial.
3. `extractor.py`: Envía la imagen + Prompt al VLM para extraer las dos fechas en JSON.
4. `validator.py`: Normaliza las fechas a formato ISO (YYYY-MM-DD) y aplica la regla del 2025-03-01. Devuelve un estado: VÁLIDO, NO VÁLIDO, o REQUIERE REVISIÓN HUMANA (HITL).
5. `export.py`: Guarda el resultado final en un archivo CSV/JSON con los campos: `[id_documento, fecha_inscripcion, fecha_renovacion, estado_validacion, motivo, nivel_confianza]`.

## 6. Instrucciones para el Copilot
- Actúa como un Arquitecto de Software y un Ingeniero de Datos Senior.
- Escribe código modular, con tipado estricto (Type Hints) y docstrings.
- Anticípate a los errores comunes de OCR (fechas mal leídas, formatos como "12 de mayo de 2024" vs "12/05/2024").
- Prioriza el manejo de excepciones (`try/except`) para que el pipeline no se rompa si un documento falla.