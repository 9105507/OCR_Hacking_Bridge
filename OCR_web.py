# uv run streamlit run OCR_web.py

import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytesseract
import streamlit as st
from PIL import Image


st.set_page_config(page_title="OCR con Streamlit", page_icon="🧾", layout="wide")


EXTENSIONES_VALIDAS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
IDIOMAS_OCR = {
    "Español": "spa",
    "Inglés": "eng",
    "Español + Inglés": "spa+eng",
}


def pil_a_bgr(imagen_pil: Image.Image) -> np.ndarray:
    """Convierte una imagen PIL a OpenCV BGR."""
    imagen_rgb = np.array(imagen_pil.convert("RGB"))
    return cv2.cvtColor(imagen_rgb, cv2.COLOR_RGB2BGR)


def bgr_a_rgb(imagen_bgr: np.ndarray) -> np.ndarray:
    """Convierte OpenCV BGR a RGB para mostrar en Streamlit."""
    return cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2RGB)


def aplicar_ocr_desde_array(
    imagen: np.ndarray,
    lang: str = "spa",
    blur_kernel: int = 3,
    usar_resize: bool = True,
    resize_ancho: int = 800,
    resize_alto: int = 600,
):
    """
    Aplica OCR sobre una imagen cargada en memoria y devuelve el texto
    junto con las imágenes intermedias del preprocesado.
    """
    if imagen is None:
        raise ValueError("La imagen cargada es inválida.")

    pasos = {}

    # Paso 1: imagen original
    pasos["Original"] = imagen.copy()

    # Paso 2: resize opcional para visualización/proceso
    imagen_proceso = imagen.copy()
    if usar_resize:
        imagen_proceso = cv2.resize(imagen_proceso, (resize_ancho, resize_alto))
    pasos["Redimensionada"] = imagen_proceso.copy()

    # Paso 3: escala de grises
    gris = cv2.cvtColor(imagen_proceso, cv2.COLOR_BGR2GRAY)
    pasos["Gris"] = gris.copy()

    # Paso 4: blur
    if blur_kernel % 2 == 0:
        blur_kernel += 1
    gris_blur = cv2.GaussianBlur(gris, (blur_kernel, blur_kernel), 0)
    pasos["Gris con blur"] = gris_blur.copy()

    # Paso 5: binarización Otsu
    _, binaria = cv2.threshold(gris_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    pasos["Binaria"] = binaria.copy()

    # OCR
    texto = pytesseract.image_to_string(binaria, lang=lang)

    return texto, pasos


def guardar_upload_temporal(uploaded_file) -> str:
    """Guarda el archivo subido temporalmente y devuelve la ruta."""
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix not in EXTENSIONES_VALIDAS:
        raise ValueError(f"Extensión no soportada: {suffix}")

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        return tmp.name


st.title("🧾 OCR de imágenes con Streamlit")
st.write("Sube una imagen, aplica preprocesado y extrae el texto con Tesseract OCR.")

with st.sidebar:
    st.header("Configuración")

    tesseract_path = st.text_input(
        "Ruta de Tesseract",
        value=r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        help="Déjalo así si tienes Tesseract instalado en esa ruta en Windows.",
    )

    idioma_label = st.selectbox("Idioma OCR", list(IDIOMAS_OCR.keys()), index=0)
    idioma_ocr = IDIOMAS_OCR[idioma_label]

    usar_resize = st.checkbox("Redimensionar antes del OCR", value=True)
    resize_ancho = st.number_input("Ancho resize", min_value=100, max_value=4000, value=800, step=50)
    resize_alto = st.number_input("Alto resize", min_value=100, max_value=4000, value=600, step=50)
    blur_kernel = st.slider("Kernel Gaussian Blur", min_value=1, max_value=11, value=3, step=2)

uploaded_file = st.file_uploader(
    "Sube una imagen",
    type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
)

if uploaded_file is not None:
    try:
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path

        imagen_pil = Image.open(uploaded_file)
        imagen_bgr = pil_a_bgr(imagen_pil)

        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader("Vista previa")
            st.image(imagen_pil, caption="Imagen subida", use_container_width=True)

        with col2:
            st.subheader("Datos del archivo")
            st.write(f"**Nombre:** {uploaded_file.name}")
            st.write(f"**Tipo:** {uploaded_file.type}")
            st.write(f"**Tamaño:** {uploaded_file.size} bytes")

        if st.button("Aplicar OCR", type="primary"):
            texto, pasos = aplicar_ocr_desde_array(
                imagen=imagen_bgr,
                lang=idioma_ocr,
                blur_kernel=blur_kernel,
                usar_resize=usar_resize,
                resize_ancho=int(resize_ancho),
                resize_alto=int(resize_alto),
            )

            st.success("OCR completado correctamente.")

            st.subheader("Resultado OCR")
            st.text_area("Texto detectado", texto, height=300)
            st.download_button(
                "Descargar texto (.txt)",
                data=texto,
                file_name="resultado_ocr.txt",
                mime="text/plain",
            )

            st.subheader("Pasos del preprocesado")
            nombres = list(pasos.keys())
            columnas = st.columns(2)

            for i, nombre in enumerate(nombres):
                with columnas[i % 2]:
                    imagen_paso = pasos[nombre]
                    if len(imagen_paso.shape) == 2:
                        st.image(imagen_paso, caption=nombre, use_container_width=True, clamp=True)
                    else:
                        st.image(bgr_a_rgb(imagen_paso), caption=nombre, use_container_width=True)

    except pytesseract.TesseractNotFoundError:
        st.error(
            "No se ha encontrado Tesseract. Revisa la ruta configurada en la barra lateral "
            "o instala Tesseract OCR en tu sistema."
        )
    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("Sube una imagen para empezar.")


st.markdown("---")
st.caption("App hecha con Streamlit + OpenCV + pytesseract")