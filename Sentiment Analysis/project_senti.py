def open_pickle(path_in, file_name):
    import pickle
    import pandas as pd
    tmp = pd.read_pickle(open(path_in + file_name, "rb"))
    return tmp

from sklearn import preprocessing
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

base_path = "C:/Users/fungk/Desktop/School/QMSS/project/"

cleaned = open_pickle(base_path, 'reviews_data_cleaned.pkl')

senti = open_pickle(base_path + 'output/', 'review_with_sent_score.pkl')

# ratings = senti.overall_ratings.values.reshape(-1,1)
# senti['ratings_adj'] = np.squeeze(preprocessing.minmax_scale(ratings,feature_range= (-1,1)))

sentiment = senti.sum_stem_sent.values.reshape(-1,1)
senti['sentiment_adj'] = np.squeeze(preprocessing.minmax_scale(sentiment,feature_range= (1,5)))

pro_sentiment = senti.pros_stem_sent.values.reshape(-1,1)
senti['pro_sentiment_adj'] = np.squeeze(preprocessing.minmax_scale(pro_sentiment,feature_range= (1,5)))

con_sentiment = senti.cons_stem_sent.values.reshape(-1,1)
senti['con_sentiment_adj'] = np.squeeze(preprocessing.minmax_scale(con_sentiment,feature_range= (1,5)))

plt.scatter(senti.sentiment_adj,senti.overall_ratings)
plt.xlabel('sentiment')
plt.ylabel('normalized ratings')
plt.savefig('senti_vs_ratings.png')
plt.show()

plt.scatter(senti.pro_sentiment_adj,senti.overall_ratings)
plt.xlabel('pros_sentiment')
plt.ylabel('normalized ratings')
plt.savefig('pro_senti_vs_ratings.png')
plt.show()

plt.scatter(senti.con_sentiment_adj,senti.overall_ratings)
plt.xlabel('cons_sentiment')
plt.ylabel('normalized ratings')
plt.savefig('cons_senti_vs_ratings.png')
plt.show()


#plot sentiment by company


senti_avg = np.average(senti.sentiment_adj)
google_senti = np.average(senti[senti.company == 'google'].sentiment_adj)
amazon_senti = np.average(senti[senti.company == 'amazon'].sentiment_adj)
netflix_senti = np.average(senti[senti.company == 'netflix'].sentiment_adj)
microsoft_senti = np.average(senti[senti.company == 'microsoft'].sentiment_adj)
senti_plot = plt.bar(['google','amazon','netflix','microsoft'],[google_senti,amazon_senti,netflix_senti,microsoft_senti],bottom = 1)
#rating by company
google_rating = np.average(senti[senti.company == 'google'].overall_ratings)
amazon_rating = np.average(senti[senti.company == 'amazon'].overall_ratings)
netflix_rating = np.average(senti[senti.company == 'netflix'].overall_ratings)
microsoft_rating = np.average(senti[senti.company == 'microsoft'].overall_ratings)
rating_plot = plt.bar(['google','amazon','netflix','microsoft'],[google_rating,amazon_rating,netflix_rating,microsoft_rating],bottom = 1)