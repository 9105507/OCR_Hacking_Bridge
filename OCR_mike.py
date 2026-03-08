import cv2
import pytesseract
import os
import re
import unicodedata
from pathlib import Path
import numpy as np
from datetime import datetime


def normalizar_texto(texto: str) -> str:
    """
    Pasa el texto a minúsculas y elimina tildes para facilitar búsquedas.
    """
    texto = texto.lower()
    texto = unicodedata.normalize("NFD", texto)
    texto = "".join(c for c in texto if unicodedata.category(c) != "Mn")
    return texto


def extraer_campos(texto: str) -> dict:
    """
    Extrae del texto OCR:
      - fecha de inscripción
      - fecha de renovación (la siguiente a la fecha de inscripción)
      - fecha de emisión (formato: '18 de julio de 2025')

    Devuelve un diccionario con los campos encontrados.
    """
    texto_norm = normalizar_texto(texto)

    # Fechas tipo 12/03/2024
    patron_fecha_num = r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"

    # Fechas tipo 18 de julio de 2025
    patron_fecha_texto = (
        r"\b\d{1,2}\s+de\s+"
        r"(enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|setiembre|octubre|noviembre|diciembre)"
        r"\s+de\s+\d{4}\b"
    )

    resultado = {
        "fecha_inscripcion": None,
        "fecha_renovacion": None,
        "fecha_emision": None,
    }

    # 1) Buscar bloque desde "inscripcion" y sacar las 2 primeras fechas numéricas:
    #    primera = inscripción, segunda = renovación
    m_bloque = re.search(
        r"inscrip(?:cion)?(.{0,300})",
        texto_norm,
        re.IGNORECASE | re.DOTALL
    )
    if m_bloque:
        fechas = re.findall(patron_fecha_num, m_bloque.group(1), re.IGNORECASE)
        if len(fechas) >= 1:
            resultado["fecha_inscripcion"] = fechas[0]
        if len(fechas) >= 2:
            resultado["fecha_renovacion"] = fechas[1]

    # 2) Fecha de emisión en formato texto
    m_emi = re.search(
        rf"fecha\s+de\s+emision\D{{0,20}}({patron_fecha_texto})",
        texto_norm,
        re.IGNORECASE
    )
    if m_emi:
        resultado["fecha_emision"] = m_emi.group(1)

    return resultado


def parsear_fecha(fecha_str: str):
    """
    Convierte una fecha en formato dd/mm/yyyy o dd-mm-yyyy a datetime.
    Devuelve None si la fecha no es válida o viene vacía.
    """
    if not fecha_str:
        return None

    fecha_str = fecha_str.strip().replace("-", "/")

    for formato in ("%d/%m/%Y", "%d/%m/%y"):
        try:
            return datetime.strptime(fecha_str, formato)
        except ValueError:
            pass

    return None


def evaluar_validez_documento(fecha_inscripcion: str, fecha_renovacion: str) -> dict:
    """
    Evalúa si el documento es válido según la condición:
        inscripción < 01/03/2025 < renovación

    Devuelve un diccionario con:
      - valido: True / False / None
      - motivo: explicación del resultado
    """
    fecha_referencia = datetime.strptime("01/03/2025", "%d/%m/%Y")

    fi = parsear_fecha(fecha_inscripcion)
    fr = parsear_fecha(fecha_renovacion)

    if fi is None and fr is None:
        return {
            "valido": None,
            "motivo": "No se pudieron interpretar ni la fecha de inscripción ni la de renovación."
        }

    if fi is None:
        return {
            "valido": None,
            "motivo": "No se pudo interpretar la fecha de inscripción."
        }

    if fr is None:
        return {
            "valido": None,
            "motivo": "No se pudo interpretar la fecha de renovación."
        }

    valido = fi < fecha_referencia < fr

    return {
        "valido": valido,
        "motivo": (
            "El documento es válido."
            if valido
            else "El documento no es válido porque no cumple: inscripción < 01/03/2025 < renovación."
        )
    }


def aplicar_ocr(file_path, lang="spa"):
    """
    Lee una imagen desde file_path y devuelve el texto detectado por OCR.

    Parámetros:
        file_path (str): ruta de la imagen
        lang (str): idioma para OCR, por ejemplo "spa", "eng" o "spa+eng"

    Retorna:
        str: texto detectado
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No existe el archivo: {file_path}")

    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

    # Leer imagen
    imagen = cv2.imread(file_path)

    if imagen is None:
        raise ValueError(f"No se pudo leer la imagen: {file_path}")

    # Nuevo tamaño: ancho, alto
    img_resize = cv2.resize(imagen, (800, 600))
    cv2.imshow("Imagen", img_resize)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Preprocesado básico
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    img_resize = cv2.resize(gris, (800, 600))
    cv2.imshow("Gris", img_resize)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    gris = cv2.GaussianBlur(gris, (3, 3), 0)

    img_resize = cv2.resize(gris, (800, 600))
    cv2.imshow("Gris Blurred", img_resize)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    _, binaria = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    img_resize = cv2.resize(binaria, (800, 600))
    cv2.imshow("Binary", img_resize)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # OCR
    texto = pytesseract.image_to_string(binaria, lang=lang)

    return texto


if __name__ == "__main__":
    carpeta_imagenes = Path("Archivos/DARDE")
    carpeta_salida = Path("Archivos/SalidaTXT")
    carpeta_salida.mkdir(parents=True, exist_ok=True)

    extensiones_validas = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    imagenes = sorted(
        [f for f in carpeta_imagenes.iterdir() if f.is_file() and f.suffix.lower() in extensiones_validas]
    )

    # ruta_imagen = imagenes[0]
    ruta_imagen = imagenes[10]
    print(ruta_imagen)

    try:
        resultado = aplicar_ocr(ruta_imagen, lang="spa")

        # Guardar OCR completo en un .txt
        ruta_txt = carpeta_salida / f"{ruta_imagen.stem}.txt"
        with open(ruta_txt, "w", encoding="utf-8") as f:
            f.write(resultado)

        campos = extraer_campos(resultado)
        print("\nCampos extraídos:")
        print(campos)

        validacion = evaluar_validez_documento(
            campos["fecha_inscripcion"],
            campos["fecha_renovacion"]
        )
        print("\nValidación del documento:")
        print(validacion)

    except Exception as e:
        print(f"Error: {e}")