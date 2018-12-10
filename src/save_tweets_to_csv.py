###############################################################################

import pandas as pd
from os.path import isfile
from utils import make_new_dir, save_to_csv
from html import unescape



def _is_retweet(text):
	return len(text) >= 3 and text[0:3]=="RT " 

def _read_tweets_from_author(filepath, drop_rt):
	data = None

	with open(filepath, "r", encoding="utf-8") as f:
		data = [unescape(tweet.rstrip("\n")) for tweet in f]
		data = [tweet for tweet in data if tweet and (not drop_rt or 
				not _is_retweet(tweet))]

	return data

def _read_tweets_to_dataframe(path, drop_rt, min_nb_tweets):

	data = pd.DataFrame(columns=["text", "author_id"])

	path_template = path + "tweet_data_{}.txt"

	for author_id in range(1, 101):
		cur_path = path_template.format(author_id)
		if isfile(cur_path):
			author_tweets = _read_tweets_from_author(cur_path, drop_rt)
			author_data = pd.DataFrame(author_tweets, columns=["text"])
			author_data.drop_duplicates(subset="text", inplace=True)
			if len(author_data) >= min_nb_tweets:
				author_data["author_id"] = author_id
				data = data.append(author_data, ignore_index=True, sort=True)

	return data



def main():
	tweets = _read_tweets_to_dataframe("data/tweet_data/", True, 2000)
	make_new_dir("data/datasets")
	save_to_csv(tweets, "data/datasets/individual_tweets.csv", "tweet_id")

if __name__ == "__main__":
	main()