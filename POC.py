#!/usr/bin/env python
# coding: utf-8

# # TO DO
# - add most popular comments

# # Imports

# In[1]:

print("===Downloading dependencies===")

import tweepy, datetime, time, nltk, re, emoji
import pandas as pd
import numpy as np
from textblob import TextBlob
from wordcloud import WordCloud, STOPWORDS
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from langdetect import detect
from nltk.stem import SnowballStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import dominate
from dominate import document
from dominate.tags import *
import pdfkit 
nltk.download('vader_lexicon')
nltk.download('punkt')
import sys
# # Connect to Twitter API

# In[18]:

print("===Connecting to Twitter API===")

days = int(sys.argv[1])
user = sys.argv[2]
filename = sys.argv[3]
online = False
if sys.argv[4] == "True":
	online = True
pdf = False
if sys.argv[5] == "True":
	pdf = True
replies_limit = 1000
api_key = "3863vc7cW3VMcfDCPT9QiSJZj"
api_key_secret = "AM80Jc6LImm2taGtubmUi1uwZ90K5orzsx4d7puFfoLAYbg0AG"
access_token = "2165317132-UJo9jDBVLd1oZY8Evnq3omd4ouHdKgt2MdsA7do"
access_token_secret = "COFGN6BmqVsSrJj6EAJ9bK3FBegnrb9WDSJLwB76UHhBV"
auth = tweepy.OAuthHandler(api_key, api_key_secret)
auth.set_access_token(access_token, access_token_secret)
auth.secure=True
api = tweepy.API(auth,wait_on_rate_limit=True, wait_on_rate_limit_notify=True)


# # Auxiliary Functions

# In[3]:


def get_tweets(api, username):
    page = 1
    deadend = False
    tweets_list = []
    while True:
        tweets = api.user_timeline(username, page = page) #include_rts=False, exclude_replies=True)
        for tweet in tweets:
            if (datetime.datetime.now() - tweet.created_at).days <= days:
                tweets_list.append([tweet.created_at, tweet.id, tweet.text])
            else:
                deadend = True
                return tweets_list
        if not deadend:
            page+=1
            #time.sleep()


# In[4]:


def get_replies(tweet_id, tweet_txt):
    replies=[]
    for tweet in tweepy.Cursor(api.search,q='to:{}'.format(user), result_type='recent', since_id=tweet_id, is_async=True, timeout=999999).items(replies_limit):
        if hasattr(tweet, 'in_reply_to_status_id_str'):
            if (tweet.in_reply_to_status_id_str==tweet_id):
                replies.append([tweet_id, tweet_txt, tweet.id, tweet.text, tweet.author.name, tweet.favorite_count, tweet.retweet_count])
    return replies


# def get_replies(tweet_id, tweet_txt):
#     replies=[]
#     reply = tweepy.Cursor(api.search, q='to:{}'.format(user), since_id=tweet_id, is_async=True,timeout=999999).items(replies_limit)
#     while True:
#         try:
#             rep = reply.next()
#             if not hasattr(rep, 'in_reply_to_status_id_str'):
#                 continue
#             if rep.in_reply_to_status_id == tweet_id:
#                 replies.append([tweet_id, tweet_txt, rep.id, rep.text, rep.author.name, rep.favorite_count, rep.retweet_count])
#                 print(tweet_id, tweet_txt, rep.id, rep.text, rep.author.name, rep.favorite_count, rep.retweet_count)
#         except tweepy.RateLimitError as e:
#             print("Twitter api rate limit reached".format(e))
#             time.sleep(60)
#             continue
#         except tweepy.TweepError as e:
#             print("Tweepy error occured:{}".format(e))
#             break
#         except StopIteration:
#             break
#         except Exception as e:
#             print("Failed while fetching replies {}".format(e))
#             break
#     return replies

# In[5]:


def cleanUpTweet(txt):
    # Remove mentions
    txt = re.sub(r'@[A-Za-z0-9_]+', '', txt)
    # Remove hashtags
    txt = re.sub(r'#', '', txt)
    # Remove retweets:
    txt = re.sub(r'RT : ', '', txt)
    # Remove urls
    txt = re.sub(r'https?:\/\/[A-Za-z0-9\.\/]+', '', txt)
    txt = emoji.demojize(txt).replace(":"," ")
    return ' '.join(txt.split())


# # Set Column Names

# In[6]:


tweets_df = pd.DataFrame(get_tweets(api, user))
tweets_df.columns = ["date_time", "post_id", "post_text"]


if online:
	print("===Getting Data from Twitter(May take several hours===")
	print("Sleeping for x, x are seconds")
	# GET DATA FROM TWITTER (TAKES LOOOONG!)
	tweets_w_replies = [get_replies(str(row['post_id']), str(row["post_text"])) for index, row in tweets_df.iterrows()]

	# POPULATE DATAFRAME
	my_list = []
	for el in tweets_w_replies:
    		for el2 in el:
    			my_list.append(el2)
	tweets_w_replies_df = pd.DataFrame(my_list, columns=["post_id", "post_text", "comment_id", "comment_text", "author_name", "favorite_count", "retweet_count"])
	print("===Saving data to file===")
    	# SAVE DATAFRAME TO FILE
	tweets_w_replies_df.to_csv(filename, sep='|')
else:
	None

# # Read DataFrame from File

# In[20]:

print("===Reading data from file===")
my_df = pd.read_csv(filename, sep='|')


# # Clean the Tweets

# In[21]:


my_df["comment_text"] = my_df["comment_text"].apply(cleanUpTweet)


# In[22]:


my_df["polarity"] = ""


# # Classify the Tweets

# In[23]:

print("===Classifying the Tweets===")
for index, row in my_df.iterrows():
    ss = SentimentIntensityAnalyzer().polarity_scores(row["comment_text"])
    if ss["compound"] == 0.0:
        my_df.at[index,"polarity"] = "neutral"
    elif ss["compound"] > 0.0:
        my_df.at[index,"polarity"] = "positive"
    else:
        my_df.at[index,"polarity"] = "negative"


# # Remove empty Tweets

# In[24]:


my_df = my_df[my_df.comment_text != '']


# # Group DataFrame by Post

# In[25]:


post_texts = my_df.post_text.unique()
grouped = my_df.groupby(my_df.post_text)


# # Create a Pie Chart for each Post

# In[26]:


colours = {'positive': '#AAFF85', 'neutral': '#00CCFF', 'negative': '#FF5033'}


# In[28]:
fig, axs = plt.subplots(len(post_texts),figsize=(10,20))
i = 0
neg = []
neu = []
pos = []
comments = []
for el in post_texts:
    if(len(post_texts) > 1):
            axs[i].set_title(el)
    else:
        axs.set_title(el)
    counts = grouped.get_group(el).polarity.value_counts()
    if hasattr(counts, "negative"):
        neg.append(counts.negative)
    else:
        neg.append(0)
    if hasattr(counts, "neutral"):
        neu.append(counts.neutral)
    else:
        neu.append(0)
    if hasattr(counts, "positive"):
        pos.append(counts.positive)
    else:
        pos.append(0)
    #comments.append([el, counts.negative, counts.neutral, counts.positive])
    if(len(post_texts) > 1):
        axs[i].pie(counts, labels=counts.index,colors=[colours[key] for key in counts.index], autopct='%1.1f%%')
    else:
        axs.pie(counts, labels=counts.index,colors=[colours[key] for key in counts.index], autopct='%1.1f%%')
    i += 1
fig.tight_layout()
plt.savefig("res.png")

# # Create a WordCloud for all comments

# In[29]:

print("===Creating wordcloud===")

wordcloud = WordCloud(background_color='white', margin=0).generate(" ".join(my_df.comment_text.unique()))


# In[30]:


plt.figure(figsize=(15,15))
plt.title("Wordcloud for the comments")
#plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
#plt.show()


# ## Save wordcloud to file

# In[31]:


wordcloud.to_file("wordcloud.png")


# # Identify the posts with the most positive/negative reactions

# In[32]:

print("===Identifying the posts with the most positive/negative reactions===")
posts_pos = []
posts_neg = []
posts_df = pd.DataFrame(post_texts)
posts_df["neg"] = neg
posts_df["neu"] = neu
posts_df["pos"] = pos
posts_df["sum"] = posts_df["neg"] + posts_df["neu"] + posts_df["pos"]
posts_df["neg"] = posts_df["neg"] / posts_df["sum"]
posts_df["neu"] = posts_df["neu"] / posts_df["sum"]
posts_df["pos"] = posts_df["pos"] / posts_df["sum"]
neg_max = posts_df["neg"].max()
pos_max = posts_df["pos"].max()
for index, row in posts_df.iterrows():
    if(row["neg"] == neg_max):
        posts_pos.append(row[0])
    if(row["pos"] == pos_max):
        posts_neg.append(row[0])




# # Comment with most retweets

# In[34]:


most_retweets = my_df[my_df["retweet_count"] == my_df["retweet_count"].max()]


# # Comment with most likes

# In[35]:


most_likes = my_df[my_df["favorite_count"] == my_df["favorite_count"].max()]

output_df = my_df[["post_text", "comment_text", "author_name", "favorite_count", "retweet_count", "polarity"]]
output_df["rt_fv"] = output_df["favorite_count"] + output_df["retweet_count"]
output_df_new = output_df.sort_values(by="rt_fv", ascending=False).head(10).drop(['rt_fv'], axis = 1) 

# # Create HTML Report

# In[36]:
logo = "hwanalytics.png"
wordcloud = "wordcloud.png"
res = "res.png"
doc = dominate.document(title='Report')
with doc.head:
    link(rel='stylesheet', href='stylesheet.css')
    meta(charset="UTF-8")
with doc:
    div(img(src=logo, cls="logo"))
    h1(f"Report of {user}'s Twitter page for the last {days} days")
    hr()
    h2(f"Wordcloud for the reactions within the last {days} days")
    div(img(src=wordcloud, cls="wordcloud"))
    hr()
    h2("Posts with most positive reactions:\n")
    for el in posts_pos:
        span(el)
        br()
    h2("\nPosts with most negative reactions:\n")
    for el in posts_neg:
        span(el)
        br()
    hr()
    span(h2("Posts with classified reactions"),img(src=res))
    hr()
    if(my_df["retweet_count"].max() != 0):
        h2("Comment(s) with most retweets")
        for index, row in most_retweets.iterrows():
            p(b("Comment: "), row["comment_text"])
            p(b("on Post: "), row["post_text"])
            p(b("by Author: "), row["author_name"])
            p(b("Retweets: "), row["retweet_count"])
        br()
    if(my_df["favorite_count"].max() != 0):
        h2("Comment(s) with most likes")
        for index, row in most_likes.iterrows():
            p(b("Comment: "), row["comment_text"])
            p(b("on Post: "), row["post_text"])
            p(b("by Author: "), row["author_name"])
            p(b("Likes: "), row["favorite_count"])
        br()
    hr()
    h2(f"Number of comments: ({len(output_df)})")
    h2("Top 10 comments with most likes and retweets")
    dominate.util.raw(output_df_new.to_html())
#   for path in photos:
#        div(img(src=path), _class='photo')


with open('report.html', 'w', encoding="utf-8") as f:
    f.write(doc.render())

# # Export PDF Report

# In[37]:
print("===Report stored as report.html===")

if pdf:
	options = {'encoding': "unicode"}
	pdfkit.from_file('report.html', 'report.pdf', options=options)

	print("===Report stored as report.pdf===")


# In[ ]:





# In[ ]:




