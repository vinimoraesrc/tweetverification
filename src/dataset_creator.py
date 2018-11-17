###############################################################################

import pandas as pd
from utils import save_to_csv
from collections import Counter
import time



def _read_tweets(filepath):
	dataframe = pd.read_csv(filepath, index_col="tweet_id", encoding="utf-8")
	return dataframe

def _extract_texts(dataframe):
	texts = dataframe["text"]
	dataframe.drop(columns=["text"], inplace=True)
	return texts

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

def _swap_half_ids(dataframe):
	half = int(len(dataframe)/2)

	upper_half = dataframe.iloc[:half,:]
	bottom_half = dataframe.iloc[half:,:]
	swapped = bottom_half.loc[:,["tweet_id2","tweet_id1"]].values
	bottom_half.loc[:,["tweet_id1","tweet_id2"]] = swapped

	return upper_half.append(bottom_half, verify_integrity=True, sort=True)

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
	tweet_count = Counter({tweet_id: 0 for tweet_id in ids})

	pairs = set()

	for jump in range(1, len(ids)):
		for start in range(len(ids)):
			id_1 = ids[start]
			id_2 = ids[(start+jump)%len(ids)]
			_try_add_pos_pair(id_1, id_2, tweet_count, pairs, nb_ocurrences)
			if len(tweet_count) <= 1:
				break
		if len(tweet_count) <= 1:
			break

	pos = pd.DataFrame(list(pairs), columns=["tweet_id1", "tweet_id2"])

	return pos

def _create_positive_examples(df, nb_ocurrences):
	df = df.sample(frac=1)

	pos_df = pd.DataFrame(columns=["tweet_id1", "tweet_id2"])

	for author_id in df["author_id"].unique():
		author_data = df[df["author_id"]==author_id]
		pos_examples = _create_author_positive_examples(author_data,
														nb_ocurrences,
														author_id)
		pos_df = pos_df.append(pos_examples, ignore_index=True)

	pos_df = pos_df.sample(frac=1)
	pos_df["label"] = True

	print("Positive")
	print(len(pos_df))

	return pos_df

def _try_add_neg_pair(author_id1, tweet_id1, author_id2, pairs, rem_data,
					  counts, max_nb, nb_authors):
	added = False
	tweet_ids2 = rem_data[rem_data["author_id"]==author_id2]
	has_tweets = len(tweet_ids2) > 0

	if has_tweets:
		tweet_id2 = tweet_ids2.sample(n=1).index[0]

		if counts[author_id2][tweet_id2] < max_nb:
			pair = (tweet_id1, tweet_id2)
			pair_rev = (tweet_id2, tweet_id1)

			if pair not in pairs and pair_rev not in pairs:
				pairs.add(pair)
				counts[author_id2][tweet_id2]+=1
				added = True
			elif nb_authors==1 and len(tweet_ids2) <= 5:
				has_tweets = False

		if counts[author_id2][tweet_id2]==max_nb:
			rem_data.drop(index=tweet_id2, inplace=True)

	return added, has_tweets

def _create_author_negative_examples(df, pairs, tweets_counts, nb_ocurrences,
									 author_id1):

	tweet_count = tweets_counts.pop(author_id1)
	tweet_ids1 = df[df["author_id"]==author_id1].index
	rem_data = df[df["author_id"].isin(tweets_counts)]
	rem_authors = rem_data["author_id"].unique().tolist()

	i = 0
	for tweet_id1 in tweet_ids1:
		if len(rem_authors)==0:
				break

		nb = 0 if tweet_id1 not in tweet_count else tweet_count[tweet_id1]

		while nb < nb_ocurrences:
			author_id2 = rem_authors[i]
			added, has_tweets = _try_add_neg_pair(author_id1, tweet_id1,
												  author_id2, pairs, 
												  rem_data, tweets_counts, 
										  		  nb_ocurrences, 
										  		  len(rem_authors))
			if not has_tweets:
				rem_authors.remove(author_id2)
				if len(rem_authors)==0:
					break
				else:
					i = i%len(rem_authors)
			elif added:
				nb+=1
				i = (i+1)%len(rem_authors)

	df.drop(index=tweet_ids1, inplace=True)

def _create_negative_examples(df, nb_ocurrences):
	df = df.sample(frac=1)

	author_ids =  df["author_id"].unique().tolist()
	counts = {author_id: Counter() for author_id in author_ids}

	pairs = set()

	[_create_author_negative_examples(df, pairs, counts, nb_ocurrences,
		author_id) for author_id in author_ids]

	neg_df = pd.DataFrame(list(pairs), columns=["tweet_id1", "tweet_id2"])

	neg_df = neg_df.sample(frac=1)
	neg_df["label"] = False

	print("Negative")
	print(len(neg_df))

	return _swap_half_ids(neg_df)

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

	all_pairs = positive.append(negative, ignore_index=True, sort=True)
	all_pairs = all_pairs.sample(frac=1)

	print("Total")
	print(len(all_pairs))

	return (selected_authors.tolist(), selected_tweets.index.tolist(), 
			all_pairs)

def _convert_format(pairs, texts):

	tweets_a = pairs["tweet_id1"].tolist()
	texts_a = [texts.loc[i] for i in tweets_a]
	tweets_b = pairs["tweet_id2"].tolist()
	texts_b = [texts.loc[i] for i in tweets_b]

	labels = pairs["label"].tolist()

	converted = pd.DataFrame(columns=["tweets_a", "tweets_b", "label"])
	converted.loc[0] = [texts_a, texts_b, labels]
	return converted


def _create_set(tweets, set_name, index_name, nb_authors, previous_authors,
				author_mode, nb_tweets_per_author, nb_pos_pairs_per_tweet,
				nb_neg_pairs_per_tweet, previous_tweets, tweet_mode, texts):

	authors, tweet_ids, pairs = _pair_tweets(tweets, nb_authors,
								  previous_authors,
								  author_mode, nb_tweets_per_author,
								  nb_pos_pairs_per_tweet,
								  nb_neg_pairs_per_tweet,
								  previous_tweets, tweet_mode)

	dataset = _convert_format(pairs, texts)
	save_to_csv(dataset, "data/datasets/"+set_name+".csv", index_name)

	return authors, tweet_ids


def main():
	index_name = None

	tweets = _read_tweets("data/datasets/individual_tweets.csv")
	texts = _extract_texts(tweets)

	train_authors, train_tweet_ids = _create_set(tweets, "same_authors_train", index_name,
												 90, None, None,
												 700, 5, 5,
												 None, None,
												 texts)

	val_authors, val_tweet_ids = _create_set(tweets, "same_authors_val", index_name,
											 90, train_authors, "same",
											 100, 5, 5,
											 train_tweet_ids, "disjoint",
											 texts)

	train_authors = set(train_authors)
	train_authors.update(val_authors)
	train_authors = list(train_authors)

	train_tweet_ids = set(train_tweet_ids)
	train_tweet_ids.update(val_tweet_ids)
	train_tweet_ids = list(train_tweet_ids)

	_, _ = _create_set(tweets, "same_authors_test", index_name,
					   90, train_authors, "same",
					   200, 5, 5,
					   train_tweet_ids, "disjoint",
					   texts)


if __name__ == "__main__":
	main()