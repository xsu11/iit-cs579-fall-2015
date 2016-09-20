
# coding: utf-8

# In[1]:

from TwitterAPI import TwitterAPI
from collections import defaultdict, Counter
from scipy.sparse import lil_matrix
from StringIO import StringIO
from zipfile import ZipFile
from urllib import urlopen
from sklearn.cross_validation import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score #, confusion_matrix

import requests
import ConfigParser
import sys
import pickle
import re
import numpy as np


# In[2]:

"""
Get location info for Streaming API use from the txt file using the city's name as the filename

Paras:
    city_name: city's name
    
Returns:
    a string to record the location info of the city
"""
def get_locations(city_name):
    return open(city_name.lower() + '.txt').readlines()[0].strip().lower()

locations = get_locations('Chicago')
print locations


# In[3]:

"""
Get geo info for REST API use from the txt file using the city's name as the filename

Paras:
    city_name: city's name
    
Returns:
    a string to record the geo info of the city
"""
def get_geocode(city_name):
    return open(city_name.lower() + '.txt').readlines()[1].strip().lower()

geocode = get_geocode('Chicago')
print geocode


# In[4]:

"""
Get the sight names from the txt file using the city's name as the filename

Paras:
    city_name: city's name
    
Returns:
    a list of string to record the sight names of the city
"""
def get_sight_names(city_name):
    return [l.strip().lower() for l in open(city_name.lower() + '.txt').readlines()[2:]]

sight_names = get_sight_names('Chicago')
print 'INDEX\tNAME'
for i, n in enumerate(sight_names):
    print i, '\t', n


# In[5]:

"""
Get multiple names of the sights.
    E.g. use “art institute of chicago”, “artinstituteofchicago” and “art_institute_of_chicago”
    for searching the tweets for the Art Institute of Chicago

Paras:
    sight_names: a sight's name
    
Returns:
    a list of tuple to record the multiple names of the sights
"""
def get_sight_multi_names(sight_names):
    return [(n, ''.join(n.split()), n.replace(' ', '_')) for n in sight_names]

sight_multi_names = get_sight_multi_names(sight_names)
for n in zip(sight_names, sight_multi_names):
    print n[0], '\n\t', ' | '.join(n[1][i] for i in range(len(n[1]))), '\n'


# In[6]:

"""
Establish twitter connection

Paras:
    config_file: config file's name
    
Returns:
    twitter
"""
def get_twitter(config_file):
    config = ConfigParser.ConfigParser()
    config.read(config_file)
    twitter = TwitterAPI(
                   config.get('twitter', 'consumer_key'),
                   config.get('twitter', 'consumer_secret'),
                   config.get('twitter', 'access_token'),
                   config.get('twitter', 'access_token_secret'))
    return twitter

twitter = get_twitter('twitter.cfg')
print 'Established Twitter connection.'


# In[7]:

"""
Get tweets using Stream API 'statuses/filter' within the limit amount with two modes:
    mode = 0: use track: keywords as paras
    mode = 1: use location: get_location(city_name) as paras

Paras:
    twitter: twitter connection
    limit: limit amount of the retrieved tweets
    city_name: city's name
    mode: when 0 use keywords as paras to search for tweets, when 1 use location
    keywords: used to search tweets
    lang='en': language option, default English
    verbose=False: if True, print log
    n=100: frequency of printing log
    
Returns:
    a list of dictionary to store all the tweets retrieved
"""
def get_tweets(twitter,
               limit,
               city_name,
               mode,
               keywords,
               lang='en',
               verbose=False,
               n=100):
    tweets = []
    paras = {}

    # statuses/filter
    # track: keywords
    if mode == 0:
        if keywords:
            paras['track'] = keywords
    
    # statuses/filter
    # location: get_location(city_name)
    elif mode == 1:
        paras['locations'] = get_locations(city_name)
        
    print 'mode =', mode
    print 'paras=' + str(paras)
    
    while True:
        try:
            if len(paras) != 0:
                for response in twitter.request('statuses/filter', paras):
                    tweets.append(response)
                    if verbose:
                        if len(tweets) % n == 0:
                            print 'found %d tweets' % len(tweets)
                    if len(tweets) >= limit:
                        return tweets
        except:
            print "Unexpected error:", sys.exc_info()[0]
        
    return tweets


# # First approach - first try

# In[ ]:

"""
def get_tweets(twitter,
           limit,
           city_name,
           mode,
           keywords,
           lang='en',
           verbose=False,
           n=100)

Get tweets with Stream API using mode 0
"""
tweets = get_tweets(twitter,
                    50,
                    'Chicago',
                    0,
                    'art institute of chicagi',
                    'en',
                    True,
                    10)
print 'Get %d tweets.' % len(tweets)


# This approach doesn't work.
# 
# We implement a second approach for smaller amount of data.

# # First Approach - second try

# In[ ]:

"""
def get_tweets(twitter,
           limit,
           city_name,
           mode,
           keywords,
           lang='en',
           verbose=False,
           n=100)

Get tweets with Stream API using mode 0
"""
tweets = get_tweets(twitter,
                    50,
                    'Chicago',
                    1,
                    None,
                    'en',
                    True,
                    10)
print 'Get %d tweets.' % len(tweets)


# In[ ]:

"""
Dump the tweets into a file for archive
"""
pickle.dump(tweets, open('tweets.pkl', 'wb'))


# In[8]:

"""
Read tweets from the archived file
"""
tweets = pickle.load(open('tweets.pkl', 'rb'))
print 'Get %d tweets from archive file.' % len(tweets)


# In[9]:

"""
Print a tweet's content

Paras:
    tweets: list of tweets
    index: index of the tweet to be printed
    
Returns:
    N/A
"""
def print_tweet(tweets, index):
    test_tweet = tweets[index]
    print('tweet:\n\tscreen_name = %s\n\tname = %s\n\tdescr = %s\n\ttext = %s' %
          (test_tweet['user']['screen_name'],
           test_tweet['user']['name'],
           test_tweet['user']['description'],
           test_tweet['text']))

print_tweet(tweets, 3377)
print_tweet(tweets, 4366)


# In[10]:

"""
Get each sight's tweet indices that relates to this sight from the retrieved tweets list

Paras:
    tweets: list of tweets
    names: sights' names
    
Returns:
    a list of list of indices
"""
def get_sight_indices(tweets, names):
    indices = []
    for i, t in enumerate(tweets):
        for n in names:
            if i not in indices and n in t['text'].lower():
                indices.append(i)
    return indices

sight_indices = []
for names in sight_multi_names:
        sight_indices.append(get_sight_indices(tweets, names))

for i in sight_indices:
    print i,


# In[11]:

"""
Print the result from getting sights' indices
"""
print 'The tweets that has \'art institute of chicago\' or \'artinstituteofchicago\' or \'art_institute_of_chicago\':'
for i, t in enumerate(tweets):
    if 'art institute of chicago' in t['text'].lower():
        print i, ':', t['text']


# This approach doesn't work when the amount of tweets is relatively small.
# 
# We implement a second approach for small amount of data.

# # Second Approach

# In[14]:

"""
Get tweets using REST API 'search/tweets' within the limit amount with two modes:
    mode = 0: use q: keywords, lang: lang as paras
    mode = 1: use q: keywords, lang: lang, geocode: get_geocode(city_name) as paras

Paras:
    twitter: twitter connection
    mode: when 0 use keywords and language as paras to search for tweets, when 1 use keywords language and location
    city_name: city's name
    keywords: used to search tweets
    limit: limit amount of the retrieved tweets
    lang='en': language option, default English
    
Returns:
    a list of dictionary to store all the tweets retrieved
"""
def get_tweets2(twitter,
                mode,
                city_name,
                keywords,
                limit=100,
                lang='en'):
    tweets = []
    paras = {}
    
    # search/tweets
    # q: keywords
    # lang: lang
    # geocode: get_geocode(city_name)
    if keywords:
        paras['q'] = keywords
        paras['count'] = limit
        paras['lang'] = lang
        if mode == 1:
            paras['geocode'] = get_geocode(city_name)
    
    print 'paras=' + str(paras)
    for r in twitter.request('search/tweets', paras):
        tweets.append(r)
    print "found %d tweets" % len(tweets)
    
    return tweets


# In[15]:

"""
Get tweets with REST API using mode 1
"""
sight_tweets_geo = []
for n in sight_names:
    print "Retrieving tweets with geo info for:", n
    sight_tweets_geo.append(get_tweets2(twitter, 1, 'Chicago', n))


# This way return nealy nothing

# In[16]:

"""
Get tweets with REST API using mode 0
"""
sight_tweets = []
for n in sight_names:
    print "Retrieving tweets for:", n
    sight_tweets.append(get_tweets2(twitter, 0, 'Chicago', n))


# In[17]:

print 'Get tweets for %d sights' % len(sight_tweets)
print 'sight 0 has %d tweets' % len(sight_tweets[0])


# In[18]:

"""
Sort the number of tweets retrieved for every sight

Paras:
    sight_tweets: list of tweets group by sight
    
Returns:
    a list of tuple that records (index of the sight, mention times)
"""
def sort_mention_times(sight_tweets):
    mention_times = ((index, len(tweets)) for index, tweets in enumerate(sight_tweets))
    return sorted(mention_times, key=lambda x:x[1], reverse=True)

sorted_sight_tweets = sort_mention_times(sight_tweets)
for i in sorted_sight_tweets:
    print i, sight_names[i[0]]


# The mention contributes little to the result.

# In[19]:

"""
Get AFINN lexicon dataset
"""
url = urlopen('http://www2.compute.dtu.dk/~faan/data/AFINN.zip')
zipfile = ZipFile(StringIO(url.read()))
afinn_file = zipfile.open('AFINN/AFINN-111.txt')

afinn = dict()

for line in afinn_file:
    parts = line.strip().split()
    if len(parts) == 2:
        afinn[parts[0]] = int(parts[1])

print 'Read', len(afinn), 'AFINN terms'


# In[20]:

"""
Using AFINN to seperate positive and negtive terms

Paras:
    terms: terms of the tweets to be determined pos/neg
    afinn: AFINN lexicon dictionary
    verbose=False: print log if True
    
Returns:
    pos: score of total positive terms
    neg: score of total megative terms
"""
def afinn_sentiment(terms, afinn, verbose=False):
    pos = 0
    neg = 0
    for t in terms:
        if t in afinn:
            if verbose:
                print '\t%s=%d' % (t, afinn[t])
            if afinn[t] > 0:
                pos += afinn[t]
            else:
                neg += -1 * afinn[t]
    return pos, neg


# In[21]:

"""
Tokenize a tweet

Paras:
    string: the tweet to be tokenized
    lowercase: change the tweet content to lowercase if True
    keep_punctuation: keep all punctuations when tokenizing the tweet if True
    collapse_urls: collapse all urls when tokenizing the tweet if True
    collapse_mentions: collapse all mentions when tokenizing the tweet if True
    
Returns:
    a list of tokens
"""
def tokenize(tweet,
             lowercase,
             keep_punctuation,
             collapse_urls,
             collapse_mentions):
    """ Split a tweet into tokens."""
    if not tweet:
        return []
    
    if lowercase:
        tweet = tweet.lower()
        
    tokens = []
    
    if collapse_urls:
        tweet = re.sub('http\S+', 'THIS_IS_A_URL', tweet)
        
    if collapse_mentions:
        tweet = re.sub('@\S+', 'THIS_IS_A_MENTION', tweet)
        
    if keep_punctuation:
        tokens = tweet.split()
    else:
        tokens = re.sub('\W+', ' ', tweet).split()
        
    return tokens


# In[22]:

"""
Get tokens list for all sights
"""
sight_tokens_list = []
for ts in sight_tweets:
    sight_tokens_list.append([tokenize(t['text'],
                                       lowercase=True,
                                       keep_punctuation=False,
                                       collapse_urls=True,
                                       collapse_mentions=False)
                  for t in ts])
print 'Get tokens list for %d sights' % len(sight_tokens_list)
print 'Print an example of the tokens:'
print sight_tokens_list[0][4]


# In[23]:

"""
Using AFINN lexicon dict to seperate the tweets to three different sets of tweets
"""
sight_positives = []
sight_negatives = []
sight_neutrals = []
for tokens in sight_tokens_list:
    positives = []
    negatives = []
    neutrals = []
    for tweet in tokens:
        pos, neg = afinn_sentiment(tweet, afinn)
        if pos > neg:
            positives.append((pos, neg, ' '.join(tweet)))
        elif neg > pos:
            negatives.append((pos, neg, ' '.join(tweet)))
        else:
            neutrals.append((pos, neg, ' '.join(tweet)))
    sight_positives.append(positives)
    sight_negatives.append(negatives)
    sight_neutrals.append(neutrals)

print len(sight_positives)
print len(sight_positives)
print len(sight_neutrals)


# In[24]:

"""
Print all positive tweets in order
"""
for index, positives in enumerate(sight_positives):
    print '\n---Positive tweets of:', sight_names[index]
    for pos, neg, tweet in sorted(positives, key=lambda x: x[1], reverse=True):
        print pos, neg, tweet


# In[25]:

"""
Print all negative tweets in order
"""
for index, negatives in enumerate(sight_negatives):
    print '\n---Negative tweets of:', sight_names[index]
    for pos, neg, tweet in sorted(negatives, key=lambda x: x[1], reverse=True):
        print pos, neg, tweet


# In[26]:

"""
Print all neutral tweets in order
"""
for index, neutrals in enumerate(sight_neutrals):
    print '\n---Neutral tweets of:', sight_names[index]
    for pos, neg, tweet in sorted(neutrals, key=lambda x: x[1], reverse=True):
        print pos, neg, tweet


# In[27]:

"""
Calculate the number of tweets in each sets and the total number of tweets
"""
total_pos = 0
total_neg = 0
total_neu = 0
for pos_ts in sight_positives:
    total_pos += len(pos_ts)
for neg_ts in sight_negatives:
    total_neg += len(neg_ts)
for neu_ts in sight_neutrals:
    total_neu += len(neu_ts)
print 'Have %d tweets in total:' % (total_pos + total_neg + total_neu)
print 'positives=%d\nnegtives=%d\nneutral=%d' % (total_pos, total_neg, total_neu)


# In[28]:

"""
A second way using AFINN to seperate positive and negtive terms and calculate the total score of the terms

Paras:
    terms: terms of the tweets to be determined pos/neg
    afinn: AFINN lexicon dictionary
    
Returns:
    total: the total score of the terms
"""
def afinn_sentiment2(terms, afinn):
    total = 0.
    for t in terms:
        if t in afinn:
            total += afinn[t]
    return total


# In[29]:

"""
Get the total score for each tweets of each sights
"""
sight_scores = []
for tokens in sight_tokens_list:
    scores= []
    for tweet in tokens:
        scores.append(afinn_sentiment2(tweet, afinn))
    sight_scores.append(scores)

print len(sight_scores)
print len(sight_scores[0])
print sight_scores[0]


# In[30]:

"""
Calculate the mean score for each sight
"""
sight_mean_scores = [(i, sum(s) / len(s)) for i, s in enumerate(sight_scores)]
print sight_mean_scores


# In[31]:

"""
Print the first 10 mean scores and sights' names
"""
top_pos = sorted(sight_mean_scores, key=lambda x:x[1], reverse=True)[:10]
for t in top_pos:
    print '%.5f %s' % (t[1], sight_names[t[0]])


# In[32]:

"""
Print the last 10 mean scores and sights' names
"""
top_neg = sorted(sight_mean_scores, key=lambda x:x[1])[:10]
for t in top_neg:
    print '%.5f %s' % (t[1], sight_names[t[0]])


# In[33]:

"""
Get the most frequently-used terms a sight's tweets use

Paras:
    tokens_list: the tokens list of the tweets of a sight
    n=20: top n frequently-used terms to be returned
    
Returns:
    a Counter stores the top n frequentlt-used terms
"""
def get_most_frequent_terms(tokens_list, n=20):
    counts = Counter()
    for tokens in tokens_list:
        str_tokens = [str(t) for t in tokens]
        counts.update(str_tokens)
    return counts


# In[34]:

"""
Get the top 30 frequently-used terms for each sight
"""
sight_term_counts = []
for tl in sight_tokens_list:
    sight_term_counts.append(get_most_frequent_terms(tl, 30))


# In[35]:

"""
Print the top 30 frequently-used terms for the first 10 sight
"""
for p in top_pos:
    print 'The top 30 frequent terms for \'%s\':' % sight_names[p[0]]
    print sorted(sight_term_counts[p[0]].items(), key=lambda x:x[1], reverse=True)[:30], '\n'


# In[36]:

"""
Print the top 30 frequently-used terms for the last 10 sight
"""
for p in top_neg:
    print 'The top 30 frequent terms for \'%s\':' % sight_names[p[0]]
    print sorted(sight_term_counts[p[0]].items(), key=lambda x:x[1], reverse=True)[:30], '\n'


# In[37]:

"""
Store the tweets into files to archive
"""
f = open('pos.txt', 'w')
for positives in sight_positives:
    for pos, neg, tweet in positives:
        content = '\t'.join([str(pos), str(neg), tweet]) + '\n'
        f.write(content)
f.close()

f = open('neg.txt', 'w')
for negatives in sight_negatives:
    for pos, neg, tweet in negatives:
        content = '\t'.join([str(pos), str(neg), tweet]) + '\n'
        f.write(content)
f.close()

f = open('neu.txt', 'w')
for neutrals in sight_neutrals:
    for pos, neg, tweet in neutrals:
        content = '\t'.join([str(pos), str(neg), tweet]) + '\n'
        f.write(content)
f.close()


# In[38]:

"""
Get the labeled tweets from the archive files

Paras:
    filenames: the filenames that are used to retrieve labeled tweets from
    
Returns:
    labels: a list of all the labels
    tweets: a list of all the relative tweets
"""
def get_labeled_tweets(filenames):
    labels = []
    tweets = []
    length = 0
    for f in filenames:
        for l in open(f).readlines():
            terms = l.strip().lower().split('\t')
            labels.append(terms[0])
            tweets.append(terms[3])
        print 'Get %d labels and tweets from the archived file: %s' % (len(tweets) - length, f)
        length = len(tweets)
    return labels, tweets


# In[39]:

"""
Get the labels and relative labeled tweets
"""
labels, labeled_tweets = get_labeled_tweets(['pos_labeled.txt', 'neg_labeled.txt', 'neu_labeled.txt'])
print len(labels)
print len(labeled_tweets)


# In[40]:

"""
Get tokens list for all labeled tweets
"""
tokens_list = [tokenize(t,
                        lowercase=True,
                        keep_punctuation=False,
                        collapse_urls=True,
                        collapse_mentions=False)
              for t in labeled_tweets]


# In[41]:

"""
Make vocabulary using the tokens list

Paras:
    tokens_list: the tokens list to be used to make the vocabulary
    
Returns:
    a dictionary of thee vocabulary
"""
def make_vocabulary(tokens_list):
    vocabulary = defaultdict(lambda: len(vocabulary))  # If term not present, assign next int.
    for tokens in tokens_list:
        for token in tokens:
            vocabulary[token]  # looking up a key; defaultdict takes care of assigning it a value.
    print '%d unique terms in vocabulary' % len(vocabulary)
    return vocabulary

vocabulary = make_vocabulary(tokens_list)


# In[42]:

"""
Make feature matrix using tweets, tokens list and vocabulary

Paras:
    tweets
    tokens_list
    vocabulary
    
Returns:
    a csr matrix of the features
"""
def make_feature_matrix(tweets, tokens_list, vocabulary):
    X = lil_matrix((len(tweets), len(vocabulary)))
    for i, tokens in enumerate(tokens_list):
        for token in tokens:
            j = vocabulary[token]
            X[i,j] += 1
    return X.tocsr()  # convert to CSR for more efficient random access.


# In[43]:

"""
Get the feature matrix and label array
"""
X = make_feature_matrix(labeled_tweets, tokens_list, vocabulary)
y = np.array(labels)


# In[44]:

"""
Using logistic regression to do n-fold cross validation

Paras:
    X: csr matrix of feature
    y: list of labels
    nfolds: n-fold
    
Returns:
    avg: the average accuracy of the cross validation
"""
def do_cross_val(X, y, nfolds):
    """ Compute average cross-validation acccuracy."""
    cv = KFold(len(y), nfolds)
    accuracies = []
    for train_idx, test_idx in cv:
        clf = LogisticRegression()
        clf.fit(X[train_idx], y[train_idx])
        predicted = clf.predict(X[test_idx])
        acc = accuracy_score(y[test_idx], predicted)
        accuracies.append(acc)
    avg = np.mean(accuracies)
    return avg

print 'avg accuracy', do_cross_val(X, y, 10)

