import twitter
import errno
from os import mkdir

api = twitter.Api(consumer_key='<your consumer key>',
                consumer_secret='<your consumer secret>',
                access_token_key='<your access token key>',
                access_token_secret='<your access token secret>',
                sleep_on_rate_limit = True)

try:
    mkdir("tweet_data")
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

def read_tweet_ids:
    path_template = "tweetverification/data/{}.txt"
    all_authors = []
    for i in range(1, 101):
        with open(path_template.format(i), "r") as f:
            data = f.readlines()
        all_authors.append([x.strip() for x in data] )
    return all_authors

def fetch_tweets(all_authors):
    all_tweets = []
    i = 0
    for tweet_ids in all_authors:
        print("fetching tweets from author {}...".format(i))
        statuses = api.GetStatuses(tweet_ids)
        tweets = [s.text for s in statuses]
        all_tweets.append(tweets)
        write_tweets(tweets, "tweet_data/tweet_data_{}.txt".format(i))
        i += 1
    return all_tweets

def write_tweets(tweets, path):
    print("writing tweets from author {} to the file system...".format(path))
    result_file = open(path, "w")
    for tweet in tweets:
        result_file.write(tweet)
    result_file.close()

def main():
    all_authors = read_tweet_ids()
    fetch_tweets(all_authors)

if __name__ == "__main__":
    main()