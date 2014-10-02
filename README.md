# Naive Bayes Algorithm
I wrote my own Naive Bayes' algorithm! Thanks to [write-up](https://github.com/toshiakit/NaiveBayes) by @toshiakit, [Andrew Ng's videos](http://openclassroom.stanford.edu/MainFolder/VideoPage.php?course=MachineLearning&video=06.1-NaiveBayes-GenerativeLearningAlgorithms&speed=100) [and notes](http://cs229.stanford.edu/notes/cs229-notes2.pdf), and [this video from Berkeley's class](https://www.youtube.com/watch?v=DNvwfNEiKvw)

## How to use this

Make a [term-document matrix](http://en.wikipedia.org/wiki/Document-term_matrix), like a NumPy array with rows as each email, and columns as each word in the master dictionary. If a word appears in the email, have the entry be 1, and 0 otherwise. I haven't tested this out on inputting the number of times a word appears in each email.

Numbering for the emails should start from 0 (i.e. there should be a 0th email, then a 1st email, etc.).

Then, make another array with 1 column where each row is an email, and 

Import `main.py`, and use `fit()` to train the algorithm. Then, load the test dataset, and use `predict()` on that. Use `score()` to find out how good your algorithm did.