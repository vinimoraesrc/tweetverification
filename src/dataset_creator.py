###############################################################################

import pandas as pd
from utils import save_to_csv
from collections import Counter



def _read_tweets_without_text(filepath):
	dataframe = pd.read_csv(filepath, index_col="tweet_id", encoding="utf-8")
	dataframe.drop(columns=["text"], inplace=True)
	return dataframe

def _select_authors(tweets, nb_authors, previous_authors, mode):
	authors = pd.Series(tweets["author_id"].unique())

	if mode == "same":
		authors = authors.loc[authors.isin(previous_authors)]
	elif mode == "disjoint":
		authors = authors.loc[~authors.isin(previous_authors)]

	return authors.sample(n=nb_authors, replace=False)

def _select_tweets_per_author(tweets, authors, nb_per_author, previous_tweets,
							  mode):

	if mode=="same":
		tweets = tweets.loc[previous_tweets,:]
	elif mode=="disjoint":
		tweets = tweets.drop(index=previous_tweets, inplace=False)

	sample = pd.DataFrame(columns=["author_id"])

	for author_id in authors:
		author_data = tweets[tweets["author_id"]==author_id]
		author_sample = author_data.sample(n=nb_per_author, replace=False)
		sample = sample.append(author_sample, ignore_index=False, sort=True)
	
	return sample

def _try_remove_from_count(idx, count, max_nb):
	if count[idx]==max_nb:
		del count[idx]

def _try_add_pos_pair(id1, id2, count, pairs, max_nb):
	if id1 != id2 and id1 in count and id2 in count:
		pair = (id1, id2)
		pair_rev = (id2, id1)
		if pair not in pairs and pair_rev not in pairs:
			count[id1]+=1
			count[id2]+=1
			pairs.add(pair)
			_try_remove_from_count(id1, count, max_nb)
			_try_remove_from_count(id2, count, max_nb)

def _create_author_positive_examples(df, nb_ocurrences, author_id):

	ids = df.index.tolist()
	count = Counter({tweet_id: 0 for tweet_id in ids})

	pairs = set()

	for jump in range(1, len(ids)):
		for start in range(len(ids)):
			id_1 = ids[start]
			id_2 = ids[(start+jump)%len(ids)]
			_try_add_pos_pair(id_1, id_2, count, pairs, nb_ocurrences)
			if len(count) <= 1:
				break
		if len(count) <= 1:
			break

	pos = pd.DataFrame(list(pairs), columns=["tweet_id1", "tweet_id2"])
	pos["author_id1"] = author_id
	pos["author_id2"] = author_id
	return pos

def _create_positive_examples(df, nb_ocurrences):
	df = df.sample(frac=1)

	pos_df = pd.DataFrame(columns=["tweet_id1", "tweet_id2", "author_id1",
								   "author_id2"])

	for author_id in df["author_id"].unique():
		author_data = df[df["author_id"]==author_id]
		pos_examples = _create_author_positive_examples(author_data,
														nb_ocurrences,
														author_id)
		pos_df = pos_df.append(pos_examples, ignore_index=True)

	pos_df["label"] = 1
	return pos_df

def _create_negative_examples(df, nb_ocurrences):
	df = df.sample(frac=1)

	neg_df = pd.DataFrame(columns=["tweet_id1", "tweet_id2", "author_id1",
								   "author_id2"])

	author_ids =  df["author_id"].unique().tolist()
	count = {author_id: Counter() for author_id in author_ids}

	quads = set()

	#for author_id in author_ids:
	#	neg_examples = _create_author_negative_examples(df, quads, count,
	#													nb_ocurrences,
	#													author_id)

	#	neg_df = neg_df.append(neg_examples, ignore_index=True)

	neg_df["label"] = 0
	return neg_df

def _pair_tweets(tweets, nb_authors, previous_authors, author_mode, 
				 nb_tweets_per_author, nb_pos_pairs_per_tweet, 
				 nb_neg_pairs_per_tweet, previous_tweets, tweet_mode):

	selected_authors = _select_authors(tweets, nb_authors, previous_authors,
									   author_mode)
	selected_tweets = _select_tweets_per_author(tweets, selected_authors,
												nb_tweets_per_author,
												previous_tweets, tweet_mode)
	positive = _create_positive_examples(selected_tweets,
										 nb_pos_pairs_per_tweet)
	negative = _create_negative_examples(selected_tweets,
										 nb_neg_pairs_per_tweet)

	return (selected_authors.tolist(), selected_tweets.index.tolist(), 
		positive.append(negative, ignore_index=True, sort=True))

def _create_set(tweets, set_name, index_name, nb_authors, previous_authors,
				author_mode, nb_tweets_per_author, nb_pos_pairs_per_tweet,
				nb_neg_pairs_per_tweet, previous_tweets, tweet_mode):

	authors, tweet_ids, pairs = _pair_tweets(tweets, nb_authors,
								  previous_authors,
								  author_mode, nb_tweets_per_author,
								  nb_pos_pairs_per_tweet,
								  nb_neg_pairs_per_tweet,
								  previous_tweets, tweet_mode)
	save_to_csv(pairs, "data/datasets/"+set_name+".csv", index_name)

	return authors, tweet_ids


def main():
	index_name = "pair_id"

	tweets = _read_tweets_without_text("data/datasets/individual_tweets.csv")

	train_authors, train_tweet_ids = _create_set(tweets, "train", index_name,
												 60, None, None,
												 500, 10, 10,
												 None, None)
	val_authors, val_tweet_ids = _create_set(tweets, "val", index_name,
												 10, train_authors, "disjoint",
												 500, 10, 10,
												 train_tweet_ids, "disjoint")

	train_authors = set(train_authors)
	train_authors.update(val_authors)
	train_authors = list(train_authors)

	train_tweet_ids = set(train_tweet_ids)
	train_tweet_ids.update(val_tweet_ids)
	train_tweet_ids = list(train_tweet_ids)

	_, _ = _create_set(tweets, "test", index_name,
					   20, train_authors, "disjoint",
					   500, 10, 10,
					   train_tweet_ids, "disjoint")


if __name__ == "__main__":
	main()