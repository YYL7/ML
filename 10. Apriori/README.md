# Association Rule
Association analysis is the task of finding relationships in large datasets. These relationships can take two forms: frequent item sets or association rules. Frequent item sets are a collection of items that frequently occur together. The second one is Association rules suggest that a strong relationship exists between two items.

The support and confidence are ways we can quantify the success of our association analysis.

--Support of an itemset is defined as the percentage of the dataset that contains this itemset.

--Confidence is defined for an association rule like {diapers} ➞ {wine}. Confidence=support ({diapers, wine})/support({diapers}), the probability of buying wine given that diapers already in the cart. This explains how likely wine is purchased when diapers are purchased. 


# Apriori
Apriori is an algorithm for frequent item set mining and association rule learning over transactional databases. It proceeds by identifying the frequent individual items in the database and extending them to larger and larger item sets if those item sets appear sufficiently often in the database.

A data set that contains N possible items can generate 2N-1 possible item sets. The Apriori principle helps us reduce the number of possible interesting itemsets. The Apriori principle says that if an itemset is frequent, then all its subsets are frequent.

Summary:

Finding different combinations of items can be a time-consuming and expensive in terms of computing power. One good approach is the Apriori algorithm, which uses the Apriori principle to reduce the number of sets. If an item is infrequent, then supersets containing that item will also be infrequent. The Apriori algorithm starts from single itemsets and creates larger sets by combining sets that meet the minimum support measure. Support is used to measure how often a set appears in the original data

The Apriori algorithm scans over the dataset each time you increase the length of your frequent itemsets. 

FPgrowth algorithm, which only needs to go over the dataset twice and thus can lead to a significant increase in speed.

# FP-growth (Frequent Pattern)
Finding frequent itemsets or pairs, sets of things that commonly occur together, by storing the dataset in a special structure called an FP-tree. 

This algorithm does a better job of finding frequent itemsets (This results in faster execution times than Apriori), but it doesn’t find association rules

The FP-growth algorithm is faster than Apriori because it requires only two scans of the database, whereas Apriori will scan the dataset to find if a given pattern is frequent or not for every potential frequent item.

The FP-growth algorithm scans the dataset only twice. The basic approach to finding frequent itemsets using the FP-growth:

--1 Build the FP-tree. 

--2 Mine frequent itemsets from the FP-tree.

The FP-growth algorithm stores data in a compact data structure called an FP-tree, which has links connecting similar items. The linked items can be thought of as a linked list.

An item can appear multiple times in the same tree. The FPtree is used to store the frequency of occurrence for sets of items. Sets are stored as paths in the tree. Sets with similar items will share part of the tree. Only when they differ will the tree split. A node identifies a single item from the set and the number of times it occurred in this sequence. A path will tell you how many times a sequence occurred. 







