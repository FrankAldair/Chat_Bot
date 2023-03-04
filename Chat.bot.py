# Importar las librerias
import random
import string
import warnings
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

# TF-IDF: Term Frequency – Inverse Document Frequency

# Descargar vocabulario auxiliar
# nltk.download() => Abre la interfaz de usuario para ver y descargar el vocabulario
nltk.download('popular', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

# Variables de chat bot
BOT_NAME = "Bot"

# Keyword Matching for Greeting
GREETING_INPUTS = ("hola", "oye", "saludos", "que tal",)
GREETING_RESPONSES = ["Hola", "Hola","Encantado que estes chateando conmigo", "Estoy contento de que estés chateando conmigo"]

# Lee el contenido de ChatBot del archivo
with open('banco.txt', 'r', encoding='utf8', errors='ignore') as fin:
    raw = fin.read().lower()

# Tokenizar contenido por oraciones
sent_tokens = nltk.sent_tokenize(raw)

# Fichas de lematización
def LemTokens(tokens):
    lemmer = WordNetLemmatizer()
    return [lemmer.lemmatize(token) for token in tokens]

# Lematizar y normalizar texto
def LemNormalize(text):
    remove_punct_dict = dict((ord(punct), None)
                             for punct in string.punctuation)
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

# retorna un Greeting si el usuario envio un Greeting
def greeting(sentence):
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)

# Procesar la entrada del usuario, obtener la respuesta y devolverla
def response(user_response):
    bot_response = ''

    # Procesar la entrada del usuario, obtener la respuesta y devolverla
    sent_tokens.append(user_response)

    # Crea y entrena un modelo de vectorizador Tf-Idf
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)

    # Obtiene los valores más similares utilizando el método de similitud de coseno ([-1]: Last Item)
    vals = cosine_similarity(tfidf[-1], tfidf)

    # Gets The Answer Index to Pick From the Array ([-2]: Last 2 Items)
    idx = vals.argsort()[0][-2]

    # Obtiene la respuesta del  index para elegir el array
    flat = vals.flatten()
    flat.sort()

    # Establece la respuesta del bot a la cadena de respuesta
    if(flat[-2] == 0):
        bot_response = bot_response + "Lo siento pero no entiendo tu pregunta"
    else:
        bot_response = bot_response+sent_tokens[idx]
    
    # Elimina la respuesta del usuario de la lista de tokens enviados
    sent_tokens.remove(user_response)

    # Devuelve la respuesta del bot
    return bot_response

# Escribir texto al usuario
def talk_to_client(message):
    print(f"{BOT_NAME}: " + message)

# Ejecuta el Chat Bot
if __name__ == '__main__':
    flag = True
    talk_to_client(f"Soy un {BOT_NAME}. Responderé a tus consultas sobre el ámbito financiero")
    while(flag == True):
        talk_to_client(
            "Escriba una pregunta sobre el ambito financiero. Si desea salir, escriba ¡Adiós!")
        user_response = input()
        if("Adiós" in user_response.lower()):
            flag = False
            talk_to_client("Adios! Cuidese...")
        elif ("gracias" in user_response.lower()):
            talk_to_client("De nada..")
        elif (greeting(user_response) != None):
            talk_to_client(greeting(user_response))
        else:
            talk_to_client(response(user_response))