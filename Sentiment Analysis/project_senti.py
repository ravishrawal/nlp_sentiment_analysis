def open_pickle(path_in, file_name):
    import pickle
    import pandas as pd
    tmp = pd.read_pickle(open(path_in + file_name, "rb"))
    return tmp

from sklearn import preprocessing
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import warnings


base_path = "C:/Users/fungk/Desktop/School/QMSS/project/"

cleaned = open_pickle(base_path, 'reviews_data_cleaned.pkl')

senti = open_pickle(base_path + 'output/', 'review_with_sent_score.pkl')
year = open_pickle(base_path + 'output/', 'review_w_sent_and_year.pkl')
# ratings = senti.overall_ratings.values.reshape(-1,1)
# senti['ratings_adj'] = np.squeeze(preprocessing.minmax_scale(ratings,feature_range= (-1,1)))

senti['year'] = year.year

sentiment = senti.sum_stem_sent.values.reshape(-1,1)
senti['sentiment_adj'] = np.squeeze(preprocessing.minmax_scale(sentiment,feature_range= (1,5)))

pro_sentiment = senti.pros_stem_sent.values.reshape(-1,1)
senti['pro_sentiment_adj'] = np.squeeze(preprocessing.minmax_scale(pro_sentiment,feature_range= (1,5)))

con_sentiment = senti.cons_stem_sent.values.reshape(-1,1)
senti['con_sentiment_adj'] = np.squeeze(preprocessing.minmax_scale(con_sentiment,feature_range= (1,5)))

warnings.filterwarnings('ignore')
senti_plot = sns.stripplot(x = senti.overall_ratings, y = senti.sentiment_adj,jitter = True, edgecolor = 'none', alpha = .40)
senti_fig = senti_plot.get_figure()
senti_fig.savefig('senti_vs_ratings.png')

pro_senti_plot = sns.stripplot(x = senti.overall_ratings, y = senti.pro_sentiment_adj,jitter = True, edgecolor = 'none', alpha = .40)
pro_fig = senti_plot.get_figure()
pro_fig.savefig('prosenti_vs_ratings.png')

con_senti_plot = sns.stripplot(x = senti.overall_ratings, y = senti.con_sentiment_adj,jitter = True, edgecolor = 'none', alpha = .40)
con_fig = senti_plot.get_figure()
con_fig.savefig('consenti_vs_ratings.png')



#data preprocessing for company ratings and company sentiments
senti_avg = np.average(senti.sentiment_adj)
google_senti = np.average(senti[senti.company == 'google'].sentiment_adj)
amazon_senti = np.average(senti[senti.company == 'amazon'].sentiment_adj)
netflix_senti = np.average(senti[senti.company == 'netflix'].sentiment_adj)
microsoft_senti = np.average(senti[senti.company == 'microsoft'].sentiment_adj)
google_rating = np.average(senti[senti.company == 'google'].overall_ratings)
amazon_rating = np.average(senti[senti.company == 'amazon'].overall_ratings)
netflix_rating = np.average(senti[senti.company == 'netflix'].overall_ratings)
microsoft_rating = np.average(senti[senti.company == 'microsoft'].overall_ratings)

x = np.arange(len(['google','amazon','netflix','microsoft']))  # the label locations
width = 0.35  # the width of the bars
fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, [google_senti,amazon_senti,netflix_senti,microsoft_senti], width, label='Sentiment')
rects2 = ax.bar(x + width/2, [google_rating,amazon_rating,netflix_rating,microsoft_rating], width, label='Rating')
ax.set_ylabel('Scores')
ax.set_xticks(x)
ax.set_xticklabels(['Google','Amazon','Netflix','Microsoft'])
ax.legend()
# data values on each bar
# ax.bar_label(rects1, padding=3)
# ax.bar_label(rects2, padding=3)
fig.tight_layout()
plt.savefig('rating_vs_senti.png')

#year
google = senti[senti.company == 'google']
google_senti_year = google[['sentiment_adj','year']]
google_senti_year['year'] = google_senti_year.year.apply(int)
google_year_plot = sns.stripplot(x = google_senti_year.year, y = google_senti_year.sentiment_adj,jitter = True, edgecolor = 'none', alpha = .40)
google_year_fig = google_year_plot.get_figure()
google_year_fig.savefig('google_senti_year.png')
