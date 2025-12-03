# translator.py

from deep_translator import GoogleTranslator

LANGS = {
    "en": "English",
    "hi": "Hindi",
    "mr": "Marathi",
    "kn": "Kannada",
    "ta": "Tamil",
    "te": "Telugu"
}

def tr(text: str, lang: str) -> str:
    """Translate text into selected language using deep_translator."""
    if lang == "en":
        return text
    try:
        return GoogleTranslator(source="auto", target=lang).translate(text)
    except Exception as e:
        print("Translation error:", e)
        return text
