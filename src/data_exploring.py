'''
Extraction of features from text
'''

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itables import to_html_datatable

from rake_nltk import Rake
from sklearn.decomposition import PCA
from wordcloud import WordCloud, STOPWORDS
from .LexRank import degree_centrality_scores
from nltk.sentiment import SentimentIntensityAnalyzer
from sentence_transformers import SentenceTransformer, util

from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyError)
warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)

def export_tables(data:pd.DataFrame, outname:str):
    html = to_html_datatable(data, display_logo_when_loading=False,
                             classes="display nowrap compact", buttons=["columnsToggle"], 
                             fixedColumns={"start": 1, "end": 2}, scrollX=True, scrollY=True,
                             autoWidth=False, column_filters="footer")
    text_file = open(outname, "w")
    text_file.write(html)
    text_file.close()

def get_embeddings(list_sentences, model, reduce=None):
    if not isinstance(list_sentences, list):
        list_sentences = list(set(list_sentences))
    embeddings = model.encode(list_sentences, show_progress_bar=False)
    reduced = None
    if reduce:
        pca = PCA(n_components=reduce)
        reduced = pca.fit_transform(embeddings)
        print(f'Original shape: {embeddings.shape} > Reduced shape: {reduced.shape}')
    return embeddings, reduced

def get_text_description(data:pd.DataFrame, colname:str, reduced=None, cols2return:list=None, kw:int=3):
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    rake_nltk_model = Rake()

    ## clustering opinions
    total_clusters = []
    column_phrase = []
    column_key = []

    for text in data[colname].values:
        sentences = list(set(text))
        ## clustering
        embeddings, reduced = get_embeddings(sentences, embedding_model, reduce=reduced)
        if reduced:
            clusters = util.community_detection(reduced, min_community_size=1, threshold=0.75)
        else:
            clusters = util.community_detection(embeddings, min_community_size=1, threshold=0.75)
        
        total_clusters.append(len(clusters))
        
        ## summary phrase for each cluster
        cluster_phrases = []
        cluster_keys = []
        for idx in range(len(clusters)):
            cluster_emb = [embeddings[x] for x in clusters[idx]] if not reduced else [reduced[x] for x in clusters[idx]]
            # Compute the similarity scores
            similarity_scores = embedding_model.similarity(cluster_emb, cluster_emb).numpy()
            # Compute the centrality for each sentence
            centrality_scores = degree_centrality_scores(similarity_scores, threshold=0.1)
            # We argsort so that the first element is the sentence with the highest score
            most_central_sentence_indices = np.argsort(-centrality_scores)
            # Central phrase
            phrase = sentences[clusters[idx][most_central_sentence_indices[0]]]
            cluster_phrases.append(phrase)

            ## get 5 keywords from each cluster
            cluster_txt = '. '.join(sentences[x] for x in clusters[idx])
            rake_nltk_model.extract_keywords_from_text(cluster_txt)
            keywords = rake_nltk_model.get_ranked_phrases()
            cluster_keys.append(list(set(keywords[:kw])))

        column_phrase.append(list(set(cluster_phrases)))
        column_key.append(cluster_keys)
    data['totalClusters'] = total_clusters
    data['Phrase'] = column_phrase
    data['Keywords'] = column_key
    
    if cols2return:
        cols2return.append('totalClusters')
        cols2return.append('Phrase')
        cols2return.append('Keywords')
        return data[cols2return]
    else:
        return data
    
def get_wordcloud(text:str, plot:bool=True, save:str=None):
    stopwords = set(STOPWORDS)
    wordcloud = WordCloud(width = 800, height = 400, background_color ='white',
                          stopwords = stopwords,
                          min_font_size = 10).generate(text)
    if plot:
        # display the WordCloud image                       
        plt.figure(figsize = (8, 4), facecolor = None)
        plt.imshow(wordcloud)
        plt.axis("off")
        plt.tight_layout(pad = 0)
        plt.show()
    if save:
        # store to file
        wordcloud.to_file(save)
        
def get_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    mapping = {'neg':'negative', 'neu':'neutral', 'pos':'positive', 'compound':'compound'}
    anlz = analyzer.polarity_scores(text)
    anlz = dict((mapping[key], value) for (key, value) in anlz.items())
    class_ = 'positive' if anlz['compound'] > 0 else 'negative'
    sentiment = sorted([(v, k) for k,v in anlz.items() if k != 'compound'])[-1]
    return class_, sentiment

def text_summaryzing(text, size=10, OPENAI_KEY=None):
    llm_chat = ChatOpenAI(temperature=0.2, model_name="gpt-3.5-turbo-0125", openai_api_key=OPENAI_KEY)       
    prompt = """Please regard the following data: {context}. \
        Extract the sentiment in one word as positive, negative or neutral\
        and a short summary with {size} words as maximum lenght. Return a json."""
    prompt = ChatPromptTemplate.from_template(prompt)
    inputs = {'context':text, 'size':size}
    chain = prompt | llm_chat
    answer = chain.invoke(inputs)
    return eval(answer.content)

def get_similar_text(data, colname, query, topk):
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    corpus = [f'index {i} | ' + x  for i,x in enumerate(data[colname].values)]
    corpus_embeddings = embedder.encode(corpus)#, convert_to_tensor=True)
    query_embedding = embedder.encode(query)#, convert_to_tensor=True)
    
    hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=topk)
    hits = hits[0]
    indexes = []
    for hit in hits:
        located = corpus[hit['corpus_id']], "(Score: {:.4f})".format(hit['score'])
        indexes.append(int(located[0][5:located[0].find('|')].strip()))
    return data.iloc[indexes]