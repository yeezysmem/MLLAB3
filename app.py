import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import ssl

import re
import csv

from gensim.corpora import Dictionary
from gensim.models import LdaModel

# Додаємо шлях до даних NLTK
nltk.data.path.append("/path/to/nltk_data")

# Обробка перевірки SSL для завантаження даних
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
    
reader = open("book.txt", "r")

# Читаємо текст книги
text = reader.read()

# Знаходимо позначки початку і кінця змісту книги (без коментарів)
start_pos = text.find("CHAPTER I.", text.find("CHAPTER I.") + 1)
end_pos = text.find("THE END")

# Виділяємо текст книги без коментарів
text = text[start_pos:end_pos + len("THE END")].strip()

# Завантажуємо дані NLTK punkt
nltk.download('punkt')

# Встановлюємо розташування SSL-сертифіката
ssl._create_default_https_context = ssl._create_unverified_context

# Завантажуємо стоп-слова NLTK 
nltk.download('stopwords')

# Отримуємо стоп-слова зі списку
stop_words = set(stopwords.words('english'))

# Виводимо стоп-слова
print("Stop words:\n")
print(stop_words)
print("\n")


# РЕДАГУЄМО ТЕКСТ
text = text.lower()

# Видаляємо цифри
text = re.sub(r'\d+', '', text)

# Видаляємо пунктуацію
text = re.sub(r'[^a-zA-Z\s]', '', text)

# Сплітимо текст на слова
tokens = nltk.word_tokenize(text, language = "english")

# Видаляємо стоп-слова
filtered_words = [word for word in tokens if word not in stop_words]

# Конвертуємо список слів у рядок
filtered_text = ' '.join(filtered_words)
chapters = re.split(r'chapter ', filtered_text)[1:]


# Визначаємо функцію для запису даних у файл CSV
def write_to_csv(name, array):
    with open(name, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=';')

         # Транспонуємо дані
        transposed_data = list(map(list, zip(*array)))

        # Записуємо заголовки
        headers = ['Chapter ' + str(i + 1) for i in range(len(array))]
        headers.insert(0, '')

        csv_writer.writerow(headers)

        # Записываем транспонированные данные в файл
        for i, row in enumerate(transposed_data):
            csv_writer.writerow([str(i + 1)] + row)

# .......... Початок обробки ..........          

def process_idf():
    print("TF-IDF:")

    vect = TfidfVectorizer()

    tf_idf_csv_array = []
    for index, chapter in enumerate(chapters):
        print(f"Chapter {index + 1}:")

        matrix = vect.fit_transform([chapter])
        
        names = vect.get_feature_names_out()

        all_words = [names[i] for i in  matrix.sum(axis=0).argsort()[0, ::-1][:20]]

        print(all_words[0][0][:20])
        tf_idf_csv_array.append(all_words[0][0][:20])

    write_to_csv('TF-IDF.csv', tf_idf_csv_array)

def process_lda():
    print("LDA:")

    lda_csv_array = []

    for index, chapter in enumerate(chapters):
        print(f"Chapter {index + 1}:")
        chapter_tokens = nltk.word_tokenize(chapter, language = "english")

        dictionary = Dictionary([chapter_tokens])

        corp = [dictionary.doc2bow(token) for token in [chapter_tokens]]

        model = LdaModel(corp, num_topics=1, id2word=dictionary)

        topics = model.show_topics(num_topics=1, num_words=20, formatted=False)

        top_words = [word[0] for word in topics[0][1]]

        print(top_words)

        lda_csv_array.append(top_words)

    write_to_csv('LDA.csv', lda_csv_array)
        

process_idf()
process_lda()


