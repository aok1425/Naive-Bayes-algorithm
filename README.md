# Naive Bayes Algorithm
I wrote my own Naive Bayes' algorithm! Thanks to [write-up](https://github.com/toshiakit/NaiveBayes) by @toshiakit, [Andrew Ng's videos](http://openclassroom.stanford.edu/MainFolder/VideoPage.php?course=MachineLearning&video=06.1-NaiveBayes-GenerativeLearningAlgorithms&speed=100) [and notes](http://cs229.stanford.edu/notes/cs229-notes2.pdf), and [this video from Berkeley's class](https://www.youtube.com/watch?v=DNvwfNEiKvw).

## Naive Bayes explained at a high level
The Naive Bayes machine learning algorithm is used to classify things based on their attributes. The classic case is determining whether an email is spam or not.

First, you need many, many emails, some of which are spam, and the others not. You need to have already marked which ones are spam. The fact that you already know the 'right answer' on at least some of these emails makes this a [supervised learning](http://en.wikipedia.org/wiki/Supervised_learning) problem. We feed these emails into the algorithm.

Now, we have received an email, and we don't know whether it's spam or not. Let's look at each word one by one. Better yet, let's convert multiple forms of a word into one form. For example, 'played' and 'playing' both become 'play'. This is called [lemmatization](http://en.wikipedia.org/wiki/Lemmatisation). We'll ignore punctuation marks and capitalization.

The first word in the email is 'viagra'. From the emails we've fed into the algorithm, we know that 'viagra' shows up in 80% of the spam emails and 2% of the non-spam emails. The probability that the email is spam is now 80%. The probability that it's not spam is 2%.

Let's look at the second word, 'rich'. This word shows up in 75% of the spam emails we fed to the algorithm, and 50% of non-spam emails. The probability that the email is spam is now 80% x 75% = 60%. The probability that the email is now not spam is 2% x 50% = 1%.

After we go through all the words in the email, we check which is greater: the probability that the email is spam, or the probablity that it's not. That's it!

## Further explanation
##### Why did we multiply the probabilities?

The way to find out if one thing occured, then another thing occured is to multiply them. For example, what's the probability of getting 2 heads in a row? It's the probability of getting one head in a flip, times the probability of getting another head, 1/2 x 1/2 = 1/4.

Importantly, in the coin flip example, the two flips are independent. Flipping a coin and getting a certain result doesn't affect the probability of getting a heads or a tails during the next flip.

In Naive Bayes, and in this spam example, we are assuming independence. That's what makes it 'naive'. We are assuming that a word appearing does not affect the probability of a certain word appearing after. This assumption is incorrect. But still, Naive Bayes gets good results even after making things imprecise by assuming so.

##### What if, in a new email, we see a word we haven't seen before?

We would be multiplying by 0/0, which is impossible. To prevent this, we will add 1 to the numerator of every word the algorithm knows, and also 1 to the denominator.

So when we fed emails to the algorithm, if 'nutshell' showed up in 10 of the 100 spam emails, we will save its probability as 11/101 = 10.9%. For a new word that hasn't been seen before, the probability that it has shown up in spam emails will be 1, and the probability that it has shown up in non-spam emails will also be 1.

This is called Laplace smoothing.

## How to use

Make a [term-document matrix](http://en.wikipedia.org/wiki/Document-term_matrix), like a NumPy array with rows as each email, and columns as each word in the master dictionary. If a word appears in the email, have the entry be 1, and 0 otherwise. I haven't tested the performance if the entry is the number of times a word appears in each email.

Numbering for the categories for `y` should start from 0. That is to say, if the two categories are spam and not spam, they should be labeled 0 and 1, respectively, not 1 and 2. You can use more than 2 categories!

Import `main.py`, and use `fit()` to train the algorithm. Then, load the test dataset, and use `predict()` on that. Use `score()` to find out how good your algorithm did.