###############################################################################

import pandas as pd
from os.path import isfile
from utils import make_new_dir, save_to_csv



def _read_tweets_from_author(filepath):
	data = None

	with open(filepath, "r", encoding="utf-8") as f:
		data = [line.rstrip("\n") for line in f if line]

	return data

def _read_tweets_to_dataframe(path):

	data = pd.DataFrame(columns=["text", "author_id"])

	path_template = path + "tweet_data_{:02}.txt"

	for author_id in range(1, 101):
		cur_path = path_template.format(author_id)
		if isfile(cur_path):
			author_tweets = _read_tweets_from_author(cur_path)
			author_data = pd.DataFrame(author_tweets, columns=["text"])
			author_data["author_id"] = author_id
			data = data.append(author_data, ignore_index=True, sort=True)

	data.drop_duplicates(subset="text", inplace=True)

	return data



def main():
	tweets = _read_tweets_to_dataframe("data/tweet_data/")
	make_new_dir("data/datasets")
	save_to_csv(tweets, "data/datasets/individual_tweets.csv", "tweet_id")

if __name__ == "__main__":
	main()