# Tweet Verification

A deep learning approach to Tweet authorship verification, based on siamese networks. Project currently under development for a deep learning MSc class @ [CIn-UFPE](https://www2.cin.ufpe.br/en/).

# Datasets

The Tweet dataset used in this project is openly provided by the [University of Victoria](https://www.uvic.ca/engineering/ece/isot/assets/stylometry/twitterdataset.pdf), and you may read more about it in the following reference.

> Marcelo Luiz Brocardo, Issa Traore. “Continuous Authentication using Micro-Messages”, Twelfth Annual International Conference on Privacy, Security and Trust (PST 2014), Toronto, Canada, July 23-24, 2014.

# Executing the Code

First, you should have a working Tweet Dev account. Then, substitute the placeholders for your credentials in the `tweet_fetcher.py` file and execute it.
This file will fetch tweets from the aforementioned dataset and store them in an appropriate directory.
You must also download the GloVe Twitter 27B 50d embeddings:

`$ wget http://nlp.stanford.edu/data/glove.twitter.27B.zip && unzip glove.twitter.27B.zip glove.twitter.27B.50d.txt -d embeddings/`

Next, execute the file `save_tweets_to_csv.py`, which will unify the downloaded tweets into a single csv file.

Afterwards, execute the `dataset_creator.py` file. You may modify its parameters, such as the number of authors and tweets per author, in the `main` function.

Finally, you may execute the `main.py` file from the root of this repository, which will load the data, perform every pre-processing step and then train our model together with a comparable one. e.g.

`$ python src/main.py tweet_data/ embeddings/glove.twitter.27B.50d.txt data/datasets/same_authors_train.csv data/datasets/same_authors_val.csv data/datasets/same_authors_test.csv --variation=none`

For more information on the accepted parameters and possible variations, execute it with the `-h` parameter.


Please note that while we offer support for our baseline models through the `svm.py` file, we do not train them in the main pipeline.

# Experiments

All experiments for these models have been conducted using Google Collab.

# Authors

[João Pedro Magalhães](https://github.com/jpedrocm)

[Vinícius Cousseau](https://github.com/vinimoraesrc)