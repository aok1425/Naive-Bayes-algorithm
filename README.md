# Naive Bayes Algorithm
I wrote my own Naive Bayes' algorithm! Thanks to [write-up](https://github.com/toshiakit/NaiveBayes) by @toshiakit, [Andrew Ng's videos](http://openclassroom.stanford.edu/MainFolder/VideoPage.php?course=MachineLearning&video=06.1-NaiveBayes-GenerativeLearningAlgorithms&speed=100) [and notes](http://cs229.stanford.edu/notes/cs229-notes2.pdf), and [this video from Berkeley's class](https://www.youtube.com/watch?v=DNvwfNEiKvw)

## A condundrum
I have two different versions of the algorithm. Same formula, but implemented differently. I get different results between the two. Why?

```
Old:
305 54
3 638
out of all predicted spam emails, 85% were actually spam (precision)
out of all the actual spam emails, 99% we predicted as being spam (recall)
f1 score is 0.914542728636

New:
305 51
3 641
out of all predicted spam emails, 86% were actually spam (precision)
out of all the actual spam emails, 99% we predicted as being spam (recall)
f1 score is 0.918674698795
```