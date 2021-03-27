import os
import re
import pandas as pd
import pickle
from stemming.porter2 import stem
from gensim import corpora, models


# tokenization
def preprocessing(text):
    words = []
    words.extend(re.findall(r'[\w]+', text))
    words = [word.lower() for word in words]
    words = [stem(word) for word in words]
    return words


class Indexer:

    def __init__(self):
        self.title, self.album, self.lyrics, self.dates, self.title_reverse_indexer, self.album_reverse_indexer, self.lyrics_reverse_indexer, self.title_word_counts, self.album_word_counts, self.lyrics_word_counts = self.build_index()

        self.topic_related_words = self.build_topic_list()

    @staticmethod
    def build_index():
        title, album, lyrics, dates = dict(), dict(), dict(), dict()
        title_reverse_indexer, album_reverse_indexer, lyrics_reverse_indexer = dict(), dict(), dict()
        title_word_counts, album_word_counts, lyrics_word_counts = dict(), dict(), dict()
        for root, dirs, files in os.walk('./data/csv'):
            for filename in files:
                songs = './data/csv/' + filename
                df = pd.read_csv(songs)
                artist = df['Artist'].tolist()[0]

                # process title
                for index, text in enumerate(df['Title'].tolist()):
                    title[artist, index] = str(text)
                    words = preprocessing(title[artist, index])
                    for i in range(len(words)):
                        word = words[i]
                        pos = i
                        if word in title_reverse_indexer:
                            if (artist, index) in title_reverse_indexer[word]:
                                title_reverse_indexer[word][artist, index].append(pos)
                            else:
                                title_reverse_indexer[word][artist, index] = [pos]
                        else:
                            title_reverse_indexer[word] = {(artist, index): [pos]}

                        if (artist, index) in title_word_counts:
                            if word in title_word_counts[artist, index]:
                                title_word_counts[artist, index][word] += 1
                            else:
                                title_word_counts[artist, index][word] = 1
                        else:
                            title_word_counts[artist, index] = {word: 1}

                # process album
                for index, text in enumerate(df['Album'].tolist()):
                    album[artist, index] = str(text)
                    if album[artist, index] == 'NaN' or album[artist, index] == 'nan':
                        album[artist, index] = ''
                    else:
                        words = preprocessing(album[artist, index])
                        for i in range(len(words)):
                            word = words[i]
                            pos = i
                            if word in album_reverse_indexer:
                                if (artist, index) in album_reverse_indexer[word]:
                                    album_reverse_indexer[word][artist, index].append(pos)
                                else:
                                    album_reverse_indexer[word][artist, index] = [pos]
                            else:
                                album_reverse_indexer[word] = {(artist, index): [pos]}

                            if (artist, index) in album_word_counts:
                                if word in album_word_counts[artist, index]:
                                    album_word_counts[artist, index][word] += 1
                                else:
                                    album_word_counts[artist, index][word] = 1
                            else:
                                album_word_counts[artist, index] = {word: 1}

                # process lyrics
                for index, text in enumerate(df['Lyric'].tolist()):
                    lyrics[artist, index] = str(text).replace(artist, '')
                    words = preprocessing(lyrics[artist, index])
                    for i in range(len(words)):
                        word = words[i]
                        pos = i
                        if word in lyrics_reverse_indexer:
                            if (artist, index) in lyrics_reverse_indexer[word]:
                                lyrics_reverse_indexer[word][artist, index].append(pos)
                            else:
                                lyrics_reverse_indexer[word][artist, index] = [pos]
                        else:
                            lyrics_reverse_indexer[word] = {(artist, index): [pos]}

                        if (artist, index) in lyrics_word_counts:
                            if word in lyrics_word_counts[artist, index]:
                                lyrics_word_counts[artist, index][word] += 1
                            else:
                                lyrics_word_counts[artist, index][word] = 1
                        else:
                            lyrics_word_counts[artist, index] = {word: 1}

                # process release dates
                for index, text in enumerate(df['Date'].tolist()):
                    dates[artist, index] = str(text)
                    if dates[artist, index] == 'NaN' or dates[artist, index] == 'nan' or dates[artist, index] == 'None':
                        dates[artist, index] = ''

        title_reverse_indexer = dict(sorted(title_reverse_indexer.items(), key=lambda x: x[0]))
        album_reverse_indexer = dict(sorted(album_reverse_indexer.items(), key=lambda x: x[0]))
        lyrics_reverse_indexer = dict(sorted(lyrics_reverse_indexer.items(), key=lambda x: x[0]))
        return title, album, lyrics, dates, title_reverse_indexer, album_reverse_indexer, lyrics_reverse_indexer, title_word_counts, album_word_counts, lyrics_word_counts

    def build_topic_list(self):
        common_texts = []
        topic_related_words = {}

        with open('./englishST.txt', encoding='utf-8') as f:
            stop_words = f.read().split()
        for (artist, index) in self.lyrics.keys():
            words = []
            words.extend(re.findall(r'[\w]+', self.lyrics[artist, index]))
            words = [word.lower() for word in words]
            words = [word for word in words if word not in stop_words]
            common_texts.append(words)

        common_dictionary = corpora.Dictionary(common_texts)
        common_dictionary.filter_extremes(no_below=10, no_above=0.5, keep_n=10000, keep_tokens=None)
        bow = [common_dictionary.doc2bow(words) for words in common_texts]
        lda = models.LdaModel(bow, num_topics=50, id2word=common_dictionary, random_state=1)
        topics = lda.show_topics(num_topics=50, num_words=5, formatted=False)
        topics_words = [[word[0] for word in topic[1]] for topic in topics]
        for words, (artist, index) in zip(common_texts, self.lyrics.keys()):
            doc_bow = common_dictionary.doc2bow(words)
            topic_related_words[artist, index] = topics_words[lda[doc_bow][0][0]]
        return topic_related_words

    def store_data(self):
        with open('./data/pickle/title.pickle', 'wb') as f:
            pickle.dump(self.title, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open('./data/pickle/title_reverse_indexer.pickle', 'wb') as f:
            pickle.dump(self.title_reverse_indexer, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open('./data/pickle/title_words_count.pickle', 'wb') as f:
            pickle.dump(self.title_word_counts, f, protocol=pickle.HIGHEST_PROTOCOL)

        with open('./data/pickle/album.pickle', 'wb') as f:
            pickle.dump(self.album, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open('./data/pickle/album_reverse_indexer.pickle', 'wb') as f:
            pickle.dump(self.album_reverse_indexer, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open('./data/pickle/album_words_count.pickle', 'wb') as f:
            pickle.dump(self.album_word_counts, f, protocol=pickle.HIGHEST_PROTOCOL)

        with open('./data/pickle/lyrics.pickle', 'wb') as f:
            pickle.dump(self.lyrics, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open('./data/pickle/lyrics_reverse_indexer.pickle', 'wb') as f:
            pickle.dump(self.lyrics_reverse_indexer, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open('./data/pickle/lyrics_words_count.pickle', 'wb') as f:
            pickle.dump(self.lyrics_word_counts, f, protocol=pickle.HIGHEST_PROTOCOL)

        with open('./data/pickle/topic_related_words.pickle', 'wb') as f:
            pickle.dump(self.topic_related_words, f, protocol=pickle.HIGHEST_PROTOCOL)

        with open('./data/pickle/dates.pickle', 'wb') as f:
            pickle.dump(self.dates, f, protocol=pickle.HIGHEST_PROTOCOL)


indexer = Indexer()
indexer.store_data()
