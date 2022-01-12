import os
import json
import time
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from random import randrange
import ipywidgets as widgets
from langdetect import detect
from yelpapi import YelpAPI
import cv2
import nltk
import spacy

from PIL import Image
from PIL import ImageFilter

from sklearn import cluster, metrics, manifold, decomposition
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.manifold import TSNE 
from sklearn.decomposition import NMF
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.utils import to_categorical

from gensim.models import TfidfModel
from gensim.corpora import Dictionary
from gensim.models import LsiModel
from gensim.models.ldamodel import LdaModel
from gensim.models.coherencemodel import CoherenceModel

import pyLDAvis.gensim_models
from cv2 import imread, GaussianBlur, BORDER_DEFAULT, equalizeHist, resize, INTER_AREA


def loading(n_fichhier=None, nrows=None):
    """
    nrows : Nombre de ligne à importer du jeu de données.
    """
    f = {
        1 : 'Identifiants business',
        2 : 'Reviews clients',
        3 : 'Publication photo & commentaires',
        4 : 'Commentaires clients par business',
        5 : 'Données de établissements',
        6 : 'Données des utilisateurs'   
    }
    d = {1:0,
     2:2,
     3:3,
     4:4,
     5:6,
     6:7
    }
    
    list_dir = os.listdir('my_folder/')
    list_dir.remove('.DS_Store')
    
    if n_fichhier != None:
        print(f'Import du fichier : "{f[n_fichhier]}" : "{list_dir[d[n_fichhier]]}"')
        return pd.read_json('my_folder/'+list_dir[d[n_fichhier]], lines=True, nrows=nrows)
        
    else :
        n = int(input("""Choose your file :
        1 : Identifiants business,
        2 : Reviews clients,
        3 : Publication photo & commentaires,
        4 : Commentaires clients par business,
        5 : Données de établissements,
        6 : Données des utilisateurs
        """))
        print(f'Import du fichier : "{f[n]}"')
        return pd.read_json('my_folder/'+list_dir[d[n]], lines=True, nrows=nrows)


def run2():
    button = widgets.Button(description='Display some image')
    out = widgets.Output()
    df = pd.read_csv('utile/data_photo.csv')

    def on_button_clicked(_):
        df = pd.read_csv('utile/data_photo.csv')
        with out:
            print('Something happens!')
            n = randrange(10000)
            affiche_random(df)
            img_resize = traitement(read(df.photo_id[n]),True)
            sift = cv2.SIFT.create()

            kp, des = sift.detectAndCompute(img_resize, None)
            img = cv2.drawKeypoints(img_resize,kp,img_resize)
            plt.figure(figsize=(10,10))
            plt.imshow(img)
            plt.show()
            
            with open('im_features', 'rb') as f1:im_features = pickle.load(f1)
            df_tsne, X_tsne = show_tsne(im_features, df)
            show_kmeans(df_tsne, X_tsne,df)
        
        
    button.on_click(on_button_clicked)
    display(widgets.VBox([button,out]))
    

def affiche_random(df):
    for u in range(len(df.label_name.value_counts().index)):
        index = df[df.label_name == df.label_name.value_counts().index[u]].index
        i = 1
        print(u,df.label_name.value_counts().index[u])
        plt.figure(figsize=(20,15))
        plt.subplot(1, 6, i)
        plt.imshow(Image.open("my_folder/photos/"+df.photo_id.iloc[index[randrange(2000)]]+'.jpg'))
        plt.axis("off")
        i+=1
        plt.subplot(1, 6, i)
        plt.imshow(Image.open("my_folder/photos/"+df.photo_id.iloc[index[randrange(2000)]]+'.jpg'))
        plt.axis("off")
        i+=1
        plt.subplot(1, 6, i)
        plt.imshow(Image.open("my_folder/photos/"+df.photo_id.iloc[index[randrange(2000)]]+'.jpg'))
        plt.axis("off")
        i+=1
        plt.subplot(1, 6, i)
        plt.imshow(Image.open("my_folder/photos/"+df.photo_id.iloc[index[randrange(2000)]]+'.jpg'))
        plt.axis("off")
        i+=1
        plt.subplot(1, 6, i)
        plt.imshow(Image.open("my_folder/photos/"+df.photo_id.iloc[index[randrange(2000)]]+'.jpg'))
        plt.axis("off")
        i+=1
        plt.subplot(1, 6, i)
        plt.imshow(Image.open("my_folder/photos/"+df.photo_id.iloc[index[randrange(2000)]]+'.jpg'))
        plt.axis("off")
        plt.show()



def data_collect():
    #Bouton de collecte des données
    button = widgets.Button(description='Lancer la collecte')
    out = widgets.Output()
    
    # Proposition de villes pour la collecte
    choose_city = widgets.SelectMultiple(options=['Paris', 'London', 'NYC', 'Madrid', 'Lisbon', 'Miami', 'Marseil'])
    # Sélection des villes par l'utilisateur
    add_city = widgets.Text(placeholder='Entrez le nom des villes, ex : Berlin Dublin')
    
    # Affichage de la sélection et de la zone de texte
    display(choose_city)
    display(add_city)


    def on_button_clicked(_):
        with out:
            
            # Définission de la liste des ville pour l'API Yelp
            print('Collecte des données...')
            selected_city = list(choose_city.value)
            if add_city.value != '':
                for city in add_city.value.split(' '):
                    selected_city.append(city)
            print('Voici les villes sélectionnées : ', selected_city)
            
            # Revoie de la liste des ville vers la fonction de collecte 
            df = generate_data_review(selected_city)
            
            # Sauvegarde du jeu de données collectées via l'API Yelp
            df.to_csv('data_colected.csv')
            
            # Traitement du text + Visualisation des topics en fonction du nombre de topic
            df, tokens = traitement_by_collect(df)
            @widgets.interact(topics = (2,5,1))
            def num_topics(topics):
                traitement_txt(df, tokens,topics)
    
    # Activation et affichage du bouton "Collecte des données"
    button.on_click(on_button_clicked)
    display(widgets.VBox([button,out]))


def build_histogram(kmeans, des, image_num):
    res = kmeans.predict(des)
    hist = np.zeros(len(kmeans.cluster_centers_))
    nb_des=len(des)
    if nb_des==0 : print("problème histogramme image  : ", image_num)
    for i in res:
        hist[i] += 1.0/nb_des
    return hist

def traitement_des_donnees(df,SIFT=False,Transfert=False):
    
    df = pd.concat([df[df.label == 'drink'][df[df.label == 'drink'].reset_index().index < 2000],
             df[df.label == 'food'][df[df.label == 'food'].reset_index().index < 2000],
              df[df.label == 'interior'][df[df.label == 'interior'].reset_index().index < 2000],
              df[df.label == 'outside'][df[df.label == 'outside'].reset_index().index < 2000],
              df[df.label == 'menu'][df[df.label == 'menu'].reset_index().index < 2000]])
    
    df = df.reset_index()

    le = LabelEncoder()
    df['label_name'] = df.label
    df['label'] = le.fit_transform(df.label)

 
    
    # Sift
    if SIST:
        df['image'] = df.photo_id.apply(read)
        df['image'] = df.image.apply(traitement)
        
        
        sift_keypoints = []
        sift = cv2.SIFT.create()

        for image_num in range(len(df)) :
            res = df.image[image_num] # convert in gray
            kp, des = sift.detectAndCompute(res, None)
            sift_keypoints.append(des)

        sift_keypoints_by_img = np.asarray(sift_keypoints)
        sift_keypoints_all    = np.concatenate(sift_keypoints_by_img, axis=0)
        np.array([len(sift_keypoints_by_img[n]) for n in range(len(df))]).mean()

        k = int(round(np.sqrt(len(sift_keypoints_all)),0))

        # Clustering
        kmeans = cluster.MiniBatchKMeans(n_clusters=k, init_size=3*k, random_state=0)
        kmeans.fit(sift_keypoints_all)


        # Creation of a matrix of histograms
        hist_vectors=[]

        for i, image_desc in enumerate(sift_keypoints_by_img) :
            hist = build_histogram(kmeans, image_desc, i) #calculates the histogram
            hist_vectors.append(hist) #histogram is the feature vector

        bag_of_img = np.asarray(hist_vectors)

        df_tsne, X_tsne = show_tsne(bag_of_img, df)
        show_kmeans(df_tsne, X_tsne)
        return bag_of_img
    
    # Transfert Learnin
    if Transfert:
        df['image_tensor'] = df.photo_id.apply(lambda x: read(x, False).resize((200,200)))
        df['image_tensor'] = df['image_tensor'].apply(np.array)
        
        
        model = VGG16(weights="imagenet", 
              include_top=False, 
              input_shape=(200, 200, 3))
        
        y = df.label.values.reshape(-1,1)
        X = df['image_tensor'].values

        y = to_categorical(y, 5)  
        X = np.array([X[u] for u in range(len(X))])


        y_pred = [model.predict(X[u:u+1]).flatten() for u in range(len(X))]
        y_pred = np.array(y_pred)
        
        df_tsne, X_tsne = show_tsne(y_pred, df)
        show_kmeans(df_tsne, X_tsne)
        
        return y_pred
    
    
    


def show_tsne(y, df):
    print("Dimensions dataset avant réduction PCA : ", y.shape)
    pca = decomposition.PCA(n_components=0.99)
    feat_pca = pca.fit_transform(y)
    print("Dimensions dataset après réduction PCA : ", feat_pca.shape)

    tsne = manifold.TSNE(n_components=2, 
                         perplexity=30, 
                         n_iter=2000, 
                         init='random', 
                         random_state=6)
    
    X_tsne = tsne.fit_transform(feat_pca)
    
    df_tsne = pd.DataFrame(X_tsne[:,0:2], columns=['tsne1', 'tsne2'])
    df_tsne["class"] = df.label_name.values
    
    plt.figure(figsize=(8,5))
    sns.scatterplot(x="tsne1", 
                    y="tsne2", 
                    hue="class", 
                    data=df_tsne)

    plt.title('TSNE selon les vraies classes', fontsize = 30, pad = 35, fontweight = 'bold')
    plt.xlabel('tsne1', fontsize = 26, fontweight = 'bold')
    plt.ylabel('tsne2', fontsize = 26, fontweight = 'bold')
    plt.legend(prop={'size': 14}) 

    plt.show()
    
    return df_tsne, X_tsne

def show_kmeans(df_tsne, X_tsne):
    cls = cluster.KMeans(n_clusters=5, random_state=6)
    cls.fit(X_tsne)

    df_tsne["cluster"] = cls.labels_
    print(df_tsne.shape)

    plt.figure(figsize=(10,6))
    sns.scatterplot(
        x="tsne1", y="tsne2",
        hue="cluster",
        palette=sns.color_palette('tab10', n_colors=5), s=50, alpha=0.6,
        data=df_tsne,
        legend="brief")

    plt.title('TSNE selon les clusters', fontsize = 30, pad = 35, fontweight = 'bold')
    plt.xlabel('tsne1', fontsize = 26, fontweight = 'bold')
    plt.ylabel('tsne2', fontsize = 26, fontweight = 'bold')
    plt.legend(prop={'size': 14}) 

    plt.show()

    labels = df.label_name
    print("ARI : ", metrics.adjusted_rand_score(labels, cls.labels_))
    
    
def traitement(img_):
    img_equlized = equalizeHist(img_)
    img_blur = GaussianBlur(equalizeHist(img_equlized), (3,3), BORDER_DEFAULT)
    img_resize = resize(img_blur, (200, 200), interpolation=INTER_AREA)
    return img_resize


def generate_data_review(selected_city):
    """Fonction de collecte des données via l'API Yelp.
    selected_city : List str des villes.
    """
    
    api_key = "PQg94HLFE1DrrNWvV4wqx2FbtCjPwAQtpG8h5tOJcA7p2t3tPmziYJi27gNXmdD5L8qp9WxAd83tvoRLtuggr3tzofjAz2GARhezUc1l_VN3Fv-x1bW9Q5KHCwWxYXYx"
    yelp_api = YelpAPI(api_key,timeout_s=10)
    
    
    city = [yelp_api.search_query(location=city)['businesses'] for city in selected_city]

    data = pd.DataFrame(columns=['name','business_id', 'id_commentaire','url','text', 'stars'])   

    for u in city:
        data = pd.concat([data,rewies_by_city(u,yelp_api)])
    print('Voici les données collectées')
    display(data.reset_index())
    return data.reset_index()


def rewies_by_city(city,yelp_api):
    reviews = []
    
    city_list = [u.values() for u in city]
    
    d = {u:list(city[0].keys())[u] for u in range(len(city[0].keys()))}
    
    for u in range(15,10,-1): d[u] = d.pop(u-1)
    
    df = pd.DataFrame(city_list).drop(10,axis=1).rename(columns=d)
    df = df[['id','name','rating','review_count']]
    id_commerce = df.id.values
    
    for u in id_commerce:
        for i in yelp_api.reviews_query(u)['reviews']:
            reviews.append([df[df.id == u].name.unique()[0],u,
                                i['id'],
                               i['url'],
                               i['text'],
                               i['rating']])
    
    return pd.DataFrame(reviews, columns=['name','business_id', 'id_commentaire','url','text', 'stars']).reset_index()

def traitement_txt(df, tokens, n):
    
    docs = []
    for doc in tokens:
        sentence = ''
        for token in doc:
            sentence+=token + " "
        docs.append(sentence)
        
    vectorizer = CountVectorizer(stop_words='english')
    BOW = vectorizer.fit_transform(docs).toarray()
    BOW = pd.DataFrame(data=BOW, columns=vectorizer.get_feature_names())
    
    vec = TfidfVectorizer(stop_words='english')
    tfidf = vec.fit_transform(docs).toarray()
    tfidf = pd.DataFrame(tfidf, columns=vec.get_feature_names())
    
    id2word = Dictionary(tokens)
    bow = [id2word.doc2bow(doc) for doc in tokens]

    model = TfidfModel(bow)
    tfidf_gensim = model[bow]

    lsa = LsiModel(corpus=tfidf_gensim, id2word=id2word,
                   num_topics= n)

    lda = LdaModel(corpus=tfidf_gensim, id2word=id2word,
               num_topics=n, passes=10)
    
    model_ = NMF(n_components=n, random_state=5)

    # Transform the TF-IDF: nmf_features
    nmf_features = model_.fit_transform(tfidf)
    components_df = pd.DataFrame(model_.components_, columns=vec.get_feature_names())


    for u in range(lsa.num_topics):
        d_lsa = {u:i for u,i in lsa.show_topic(u)}
        d_lda = {u:i for u,i in lda.show_topic(u)}
        
        plt.figure(figsize=(15,5))
        plt.subplot(1, 2, 1)
        sns.barplot(y = list(d_lsa.keys()), x = list(d_lsa.values()))
        
        
        plt.subplot(1, 2, 2)
        sns.barplot(y = list(d_lda.keys()), x = list(d_lda.values()))

        
        plt.show()
        
    label = [pd.DataFrame(nmf_features).loc[n].idxmax() for n in range(len(nmf_features))]
    X_embedded = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(nmf_features)
    X_embedded.shape

    plt.figure(figsize=(15,5))
    plt.subplot(1, 2, 1)
    sns.scatterplot(X_embedded[:,0],X_embedded[:,1], hue=label)
    
    plt.subplot(1, 2, 2)
    sns.barplot(y = pd.DataFrame(nmf_features).idxmax().values, x=list(pd.DataFrame(nmf_features).idxmax().index))
    plt.show()
    
    vis = pyLDAvis.gensim_models.prepare(lda, corpus=tfidf_gensim, 
                                                dictionary=id2word)
    display(vis)
    
def traitement(img_, show=False):
    """
    img_ : array img
    show : True or False, True : showing img, False: don't showing img
    """
    img_equlized = equalizeHist(img_)
    if show: show_img(img_, img_equlized)
    
    img_blur = GaussianBlur(equalizeHist(img_equlized), (3,3), BORDER_DEFAULT)
    if show: show_img(img_equlized, img_blur)
    
    img_resize = resize(img_blur, (200, 200), interpolation=INTER_AREA)
    if show: show_img(img_blur, img_resize)
    return img_resize


def read(df_photo_id, mode=True):
    if mode:
        return imread("my_folder/photos/" + df_photo_id + '.jpg', 0)
    else : return Image.open("my_folder/photos/" + df_photo_id + '.jpg')
    
    
def show_img(Img, Image):
    
    plt.figure(figsize = (20,10))
    plt.subplot(2, 2, 1)
    plt.imshow(Img)
    plt.axis("off") 

    plt.subplot(2, 2, 2)
    plt.imshow(Image)
    plt.axis("off") 

    plt.subplot(2, 2, 3)
    plt.title('Niveau de gris avant traitement')
    plt.xlabel('niveau de gris')
    plt.ylabel('nombre de pixel')
    patches = plt.hist(Img.flatten(), bins=range(256))

    plt.subplot(2, 2, 4)
    plt.title('Niveau de gris après traitement')
    patches = plt.hist(Image.flatten(), bins=range(256))
    plt.xlabel('niveau de gris')
    plt.ylabel('nombre de pixel')
    plt.show()

def traitement_by_collect(df):
    # Import spacy and load the English version of the model
    nlp = spacy.load("en_core_web_sm")
    stop_words = nltk.corpus.stopwords.words('english')
    stop_words = set(stop_words)   
    document = df.text.apply(nlp)

    tokens = [[u.lemma_ for u in doc if u.is_alpha and not u.is_punct and detect(u.text)=='en' and u.text not in stop_words] for doc in document]
   
    return df, tokens


def show_tsne(y, df):
    print("Dimensions dataset avant réduction PCA : ", y.shape)
    pca = decomposition.PCA(n_components=0.99)
    feat_pca = pca.fit_transform(y)
    print("Dimensions dataset après réduction PCA : ", feat_pca.shape)

    tsne = manifold.TSNE(n_components=2, 
                         perplexity=30, 
                         n_iter=2000, 
                         init='random', 
                         random_state=6)
    
    X_tsne = tsne.fit_transform(feat_pca)
    
    df_tsne = pd.DataFrame(X_tsne[:,0:2], columns=['tsne1', 'tsne2'])
    df_tsne["class"] = df.label_name.values
    
    plt.figure(figsize=(15,7))
    sns.scatterplot(x="tsne1", 
                    y="tsne2", 
                    hue="class", 
                    data=df_tsne)
    sns.color_palette("Set2")

    plt.title('TSNE selon les vraies classes', fontsize = 30, pad = 35, fontweight = 'bold')
    plt.xlabel('Projection tsne1', fontsize = 26, fontweight = 'bold')
    plt.ylabel('Projection tsne2', fontsize = 26, fontweight = 'bold')
    plt.legend(prop={'size': 14}) 

    plt.show()
    
    return df_tsne, X_tsne

def show_kmeans(df_tsne, X_tsne,df):
    cls = cluster.KMeans(n_clusters=5, random_state=6)
    cls.fit(X_tsne)

    df_tsne["cluster"] = cls.labels_
    print(df_tsne.shape)

    plt.figure(figsize=(15,7))
    sns.scatterplot(
        x="tsne1", y="tsne2",
        hue="cluster",
        palette=sns.color_palette('tab10', n_colors=5), s=50, alpha=0.6,
        data=df_tsne,
        legend="brief")
    sns.color_palette("Set2")

    plt.title('TSNE selon les clusters', fontsize = 30, pad = 35, fontweight = 'bold')
    plt.xlabel('Projection tsne1', fontsize = 26, fontweight = 'bold')
    plt.ylabel('Projection tsne2', fontsize = 26, fontweight = 'bold')
    plt.axis("off") 
    plt.legend(prop={'size': 14}) 

    plt.show()

    labels = df.label_name
    print("ARI : ", metrics.adjusted_rand_score(labels, cls.labels_))