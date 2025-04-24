from deep_translator import GoogleTranslator
from langdetect import detect

def translate_to_french(text):
    try:
        detected_lang = detect(text)
        if detected_lang == 'fr':
            return text  # Already in French, no need to translate
        else:
            translated = GoogleTranslator(source='auto', target='fr').translate(text)
            return  translated

    except Exception as e:
        return f"Error during translation: {e}"

# Example usage
"""if __name__ == "__main__":
    text = input("Enter text in English or French: ")
    result = translate_to_french(text)
    print(f"\nResult:\n{result}")"""

