# The bert part of nlp_sentiment_analysis
First, place the pkl file ```review_with_sent_score.pkl``` under this directory.

Then, to further preprocess the data, use ```python bert_preprocess.py```.

Next, to train the model, use ```python bert_main.py | tee log.txt```. Ouput is saved in 'log.txt'

Finally, to evaluate your model, use ```python bert_test.py | tee evaluation.txt```. Output is saved in 'evaluation.txt'
