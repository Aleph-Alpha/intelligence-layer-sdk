from intelligence_layer.core.detect_language import Language

LANGUAGES_QA_INSTRUCTIONS = {
        Language("en"): """{{question}} If there's no answer, say {{no_answer_text}}. Only answer the question based on the text.""",
        Language("de"): """{{question}} Wenn es keine Antwort gibt, gib {{no_answer_text}} aus. Beantworte die Frage nur anhand des Textes.""",
        Language("fr"): """{{question}} S'il n'y a pas de réponse, dites {{no_answer_text}}. Ne répondez à la question qu'en vous basant sur le texte. """,
        Language("es"): """{{question}}Si no hay respuesta, di {{no_answer_text}}. Responde sólo a la pregunta basándote en el texto.""",
        Language("it"): """{{question}}Se non c'è risposta, dire {{no_answer_text}}. Rispondere alla domanda solo in base al testo.""",
    }
