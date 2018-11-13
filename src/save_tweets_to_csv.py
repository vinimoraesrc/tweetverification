###############################################################################

import pandas as pd
from os.path import isfile, isdir
from os import mkdir



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

def _save_to_csv(dataframe, dataset_dir):

	if isdir(dataset_dir) == False:
		mkdir(dataset_dir)

	filepath = dataset_dir + "individual_tweets.csv"

	dataframe.to_csv(filepath, index_label="tweet_id", encoding="utf-8")

def main():
	tweets = _read_tweets_to_dataframe("data/tweet_data/")
	_save_to_csv(tweets, "data/datasets/")


if __name__ == "__main__":
	main()