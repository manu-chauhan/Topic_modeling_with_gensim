import os
import re
import sys
import csv
import nltk
import string
import pickle
import traceback
import collections
import pandas as pd
import psycopg2 as pg
from Utils import Utils
from gensim import models
from gensim import corpora
from Definitions import ROOT_DIR
from nltk.stem.wordnet import wordnet
from stop_words import get_stop_words
from nltk.stem.wordnet import WordNetLemmatizer

author = 'Manu Chauhan'

clean_text = []

raw_text_pkl_path = os.path.join(ROOT_DIR, "data/raw_text.pkl")


def store_raw_text(data):
    """
    used to store raw text as pickle file
    :param data: raw data as list
    :return: None
    """

    if os.path.exists(raw_text_pkl_path):
        os.remove(raw_text_pkl_path)

    with open(raw_text_pkl_path, 'wb') as f:
        pickle.dump(data, f)


def get_raw_text():
    """
    used to retrieve raw data which was stored as pickle file
    :return: raw data
    """

    if os.path.exists(raw_text_pkl_path):
        with open(raw_text_pkl_path, 'rb') as f:
            raw_data = pickle.load(f)

        return raw_data
    else:
        print('No raw_text.pkl file found')
        sys.exit()


def get_stop_words_set():
    """
    returns set of stop words for english language. Lookup in a set is faster than a list.
     Other libs can also be used such as nltk stop_words. But the one used here is larger than nltk stop words list
    :return: set of stop words
    """

    return set(get_stop_words(language='english'))


def get_punctuations_set():
    """
    return a set of punctuations from string module of python. Lookup in a set is faster than a list.
    :return: set of punctuations
    """

    return set(string.punctuation)


def get_wordnet_pos(tag):
    """
    Converts nltk.pos tag to WordNet part of speech name
    :param tag: nltk.pos tag
    :return: WordNet pos name that WordNetLemmatizer can use
    """

    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return False


def get_data():
    """
    Retrieves data from database. Uses psycopg2 and pandas.
     Uses database details from 'config.properties' file in conf dir by calling 'get_db_details' from Utils.py and
     connects to the database to retrieve data using 'get_data' sql provided in config.properties file.
    :return: list of texts
    """

    conn = None
    db = Utils.get_db_details()

    try:
        conn = pg.connect(database=db['db_name'], user=db['db_user'], password=db['db_password'], host=db['db_host'],
                          port=db['db_port'])

        sql = Utils.get_sql_dict()['get_data']

        df = pd.read_sql(sql, conn)
        data = df['text']

    except Exception:
        print('Exception in get_data()', traceback.format_exc())
        sys.exit()
    else:
        return data
    finally:
        conn.close()


def get_dictionary(data):
    """
    Used to form a dictionary from list of texts after performing part of speech tagging, cleaning text for stop words,
     punctuations and digits and then Lemmatizing words.
     The order of steps is important here.
     POS tagging should be performed before removing stop words and punctuations to get better pos tags for words.
     Words with length > 2 are considered.

     WordNetLemmatizer is used. Lemmatizer takes word and pos tag.

     Only Nouns, Verbs, Adverbs and Adjectives are considered.

     Dictionary is filtered for words which occur only in one document.
     Dictionary is saved for future reference under data dir to speed up processes next time (if data is not changed).

     Note: 'mydict.dict' must be removed along with all other files under data directory if data
      on which topic modeling is performed changes.

    :param data: list of texts(documents)
    :return: dictionary for the data provided as param
    """

    texts = []
    store_raw_text(data)

    lemmatizer = WordNetLemmatizer()

    stop_words = get_stop_words_set()
    punctuations = get_punctuations_set()

    for text in data:
        tags = nltk.pos_tag(nltk.word_tokenize(text.lower()))

        clean_txt = [(word, get_wordnet_pos(pos)) for word, pos in tags if
                     word not in stop_words and word not in punctuations and len(
                         word) > 2 and not word.isdigit() and get_wordnet_pos(pos) is not False]

        texts.append([lemmatizer.lemmatize(word=word, pos=pos) for word, pos in clean_txt])

    globals()['clean_text'] = texts

    dictionary = corpora.Dictionary(texts)

    once_ids = [tokenid for tokenid, docfreq in dictionary.dfs.items() if docfreq == 1]

    dictionary.filter_tokens(once_ids)
    dictionary.compactify()
    dictionary.save(os.path.join(ROOT_DIR, "data/mydict.dict"))

    return dictionary


def get_corpus(texts, dictionary):
    """
    Forms corpus for the data using the dictionary created before.
    Also serializes the corpus as 'corpus.mm' under data dir.
    Serialization is done to speed up processes next time.

    Note: 'corpus.mm' file must be deleted along with all other files under data directory if data
     on which topic modeling is performed changes.

    :param texts: cleaned texts(documents)
    :param dictionary: the dictionary for the data created earlier
    :return: corpus for the data
    """

    corpus = [dictionary.doc2bow(text) for text in texts if len(dictionary.doc2bow(text)) > 0]
    corpora.MmCorpus.serialize(os.path.join(ROOT_DIR, "data/corpus.mm"), corpus)

    return corpus


def convert_corpus_to_tfidf(corpus):
    """
    Transforms the corpus using 'TfidfModel' from gensim.models.
    Tfidf is important as it weights down common words across documents in a collection while
     weighing up important words.

    :param corpus: the corpus for the document collection created earlier
    :return: tfidf transformed corpus
    """

    model_tfidf = models.TfidfModel(corpus, normalize=True)
    corpus_tfidf = model_tfidf[corpus]

    return corpus_tfidf


def get_lda(dictionary, corpus):
    """
    Creates and returns the trained LDA model.
    'get_lda_params' from Utils.py gets parameters for LDA model from 'config.properties' file which are:
    :argument 'num_topics' to train for this number of topics, :argument 'passes' for number of passes over corpus.

    :param dictionary: the dictionary for the documents created earlier
    :param corpus: the corpus for the documents created earlier
    :return: LDA model
    """

    param_dict = Utils.get_lda_params()
    lda_model = models.LdaModel(corpus=corpus, num_topics=int(param_dict['train_topic_number']), id2word=dictionary,
                                passes=int(param_dict['passes_number']))

    return lda_model


def topic_words_dict(lda_model):
    """
    Finds 'num_topics' number of topics from the LDA model passed as param,
     where each topic has 'num_words' number of prominent words.

    The values for :argument 'num_topics' and :argument 'num_words' for show_topics method is retrieved
     from 'config.properties file' by 'get_lda_params' in Utils.py and is used here.

    This method creates a CSV file in 'OUTPUT' dir of the project as 'topic_dict.csv'
     which contains each topic id and corresponding 'num_words' prominent words for each topic id and their probability.

    :param lda_model: the LDA model returned from get_lda
    """

    param_dict = Utils.get_lda_params()
    l = lda_model.show_topics(num_topics=int(param_dict['get_topic_number']), num_words=int(param_dict['word_number']),
                              formatted=False)
    l.sort(key=lambda x: x[0])
    # print(l)

    topic_dict = collections.defaultdict(dict)

    for topic_id, word_list in l:
        word_percentage_dict = collections.defaultdict(float)

        for word, percentage in word_list:
            word_percentage_dict[word] = round(percentage, 4)

        sorted_list = sorted(word_percentage_dict.items(), key=lambda x: -x[1])
        ordr_dict = collections.OrderedDict(sorted_list)
        topic_dict[topic_id] = ordr_dict

    # print('topic dict ', topic_dict, '\n')

    with open(os.path.join(ROOT_DIR, 'output/topic_data.csv'), 'w', newline='') as file:
        writer = csv.writer(file, delimiter=',')

        writer.writerow(["Topic ID", "Word", "Percentage"])
        writer.writerow([])

        for topic_id, d in topic_dict.items():
            for word, perc in d.items():
                writer.writerow([topic_id, word, perc])
            writer.writerow([])


def text_to_topic(texts, model):
    """
    Finds topic distribution for each document or text in texts.
    Uses the trained LDA model to figure out which topic(s) a text belongs to with percentage for each topic.
    The same raw texts are used but new unseen documents can also be used on the trained LDA model.

    This method writes a CSV file 'text_to_topic.csv' under 'OUTPUT' dir of the project containing
     each raw text along with topic id(s) and percentage for that topic id ie. Topic distribution for each text.

    :param texts: the texts on which topic distribution needs to be retrieved
    :param model: Trained LDA model
    """

    data = []
    stop_words = get_stop_words_set()

    punctuations = get_punctuations_set()

    for text in texts:
        words = re.findall(r'\w+', text.lower(), flags=re.UNICODE | re.LOCALE)

        tmp = [word for word in words if
               word not in stop_words and word not in punctuations and len(word) > 1 and not word.isdigit()]

        bow = model.id2word.doc2bow(tmp)
        bow = model[bow]

        doc_topic, word_topics, phi_value = model.get_document_topics(bow, per_word_topics=True)

        doc_topic.sort(key=lambda x: -x[1])

        tmp = []
        # print('doc topic ', doc_topic)
        tmp.append(text)
        for i in range(len(doc_topic)):
            for j in range(0, 2):
                tmp.append(doc_topic[i][j])
        # tmp.append(' ')
        tmp = tuple(tmp)
        # print('tmp ', tmp)
        data.append(tmp)

    with open(os.path.join(ROOT_DIR, 'output/text_to_topic.csv'), 'w', newline='', encoding='UTF-8') as file:
        writer = csv.writer(file, delimiter=',')
        for item in data:
            writer.writerow([x for x in item])
            writer.writerow([])


if __name__ == "__main__":
    if os.path.isfile(os.path.join(ROOT_DIR, "data/mydict.dict")) and os.path.isfile(
            os.path.join(ROOT_DIR, "data/corpus.mm")):

        dictionary = corpora.Dictionary.load(os.path.join(ROOT_DIR, "data/mydict.dict"))
        corpus = corpora.MmCorpus(os.path.join(ROOT_DIR, "data/corpus.mm"))
    else:
        dictionary = get_dictionary(get_data())
        corpus = get_corpus(clean_text, dictionary)

    tfidf_corpus = convert_corpus_to_tfidf(corpus)
    lda_model = get_lda(dictionary, tfidf_corpus)
    topic_words_dict(lda_model)

    '''
    Here, in text_to_topic, same raw texts(documents) are provided which were used to train LDA .
     New or unseen documents can also be provided.
    '''

    text_to_topic(get_raw_text(), lda_model)
