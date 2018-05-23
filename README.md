# Political Ideology Detection

## Abstract
Political ideologies are diverse and complex, but can be approximated through ones choice of language. Our paper explores existing research that predicts political ideology through text or speech, but uses these tools to understand political ideologies of United States presidential candidates over time. We take current research that typically classifies political ideology and instead predict where politicians fall on a liberal to conservative scale. We take speech from the Congressional Record to train our model, then predict political ideology of all presidential candidates using their language during presidential debates. This provides a method to computationally measure how polarized or similar candidates are.

## Introduction
The United States functions under a two-party political system, which means there are typically two main parties competing for votes. Over time, when enough Americans are unsatisfied with a party, one of two things happens: a third party emerges or an existing party changes. Over time, we get political parties with varying views and ideologies. An interesting aspect of these changing parties is the partisanship that emerges, whether it is a new party voting together as a block or the minority party using Congressional tools to prevent bills whose purpose with which they disagree. Our goal is to look at the ideological polarization of the presidential candidates for the United States and understand how that polarization has changed over the last fifty years.

The datasets we used for this research are the Congressional Record, transcripts of presidential debates, and DW-NOMINATE scores (Lewis et al. 2017). Because language changes over time, and these scores are ideology scores relative to the rest of the members of Congress, we included year as a feature in our models.

Much of the research on political ideology focuses on classification tasks: liberal or conservative, Republican or Democrat. Instead, we investigated predicting political ideology for the purpose of identifying relative ideology of a group of people for which not much data exists: presidential candidates, particularly those who did not become president. The way in which candidates speak to American citizens is an indication of how American citizens feel during that election cycle. If candidates hope to become president, their campaign platform needs to reflect the desires and needs of their voting base, and so these candidates need to speak in a manner that resonates with them. We believe that, with the political ideologies of presidential candidates, we can better understand the political divides (or lack thereof) of the United States.

## Data
In order to predict idology from text transcripts we needed political speech transcripts from various ideologies over a long time period. We found these qualities in the Congressional Record (The Congressional Record), which contains the transcripts of every session of Congress in both the House and the Senate, whether it is about political debates, speeches, or votes. The Congressional Record is stored online as PDF documents. We used two separate sites to find the PDFs for all records, up to and including 1999 (Congressional Record (Bound Edition)) and post-1999 (FDsys). The first site provided PDFs of week/month long periods of the record, so we randomly downloaded sections from each year and took random pages from those sections. The second site provided PDFs for each day, so we randomly downloaded days throughout the year and took random pages from those days. Once we had all the PDFs we used pdfminer (Shinyama, 2014 ), a python library for parsing PDF documents, to obtain the transcripts of each page of text.

Once we collected the training data, we needed labels to determine how liberal or conservative each sentence is. We combined our data from the Congressional Record with DW-NOMINATE scores from voteview.com (Lewis et al. 2017). This score uses the voting records from members of congress to score their political ideology on a scale from -1 to 1. These are normalized scales that compares each member of Congress to every other member of their Congress based on voting records. -1 means the member is the most liberal member of Congress, and 1 means the most conservative.

Finally, we scraped transcripts of presidential debates from The Presidency Project website (Peters and Woolley 2018), transforming the results into the same format as the training data. This data, unlike the congressional records, is stored as plain text on the website itself. This allowed us to directly scrape the debate text. We collected transcripts from every general election debate since 1976, as well as 1960. The site also includes primary debates beginning in 2000 so we also collected those transcripts as well. General election debates typically features one Democrat and one Republican, while primary debates are intra-party debates, e.g. a group of Republicans vying for the Republican nomination. In addition to Presidential debates, this also contains Vice-Presidential debates.

## Methodology
We formulated the problem as a supervised learning exercise and tried to predict the political ideology associated with each sentence spoken by a speaker. We later aggregated them to give a final predicted score for each speaker. For modeling the ideology, two distinct modeling approaches were used:

1. Bag of Words based models
2. Recurrent Neural Network (RNN) based models

### Bag of Words based models
All the sentences were converted into their bag-ofwords representation to indicate word counts. The overall vocabulary of the corpus is over 200,000 words but most words are not frequent. We eliminated all stop-words and selected the top 8000 words for the modeling. These bag of words were weighted using Term Frequency Inverse Document Frequency (TF-IDF).

We used two linear models using this TFIDF representation: Ridge Regression (L2 regularization) and Lasso (L1 regularization). We also tried Support Vector Machines (SVM) and Neural Networks but their results were not consistent. Hence we report only the results from the linear models.

### Long Short Term Memory (LSTM) model
An LSTM (Hochreiter, Sepp, et al) is a type of recurrent neural network (RNN) (Jain, L. C., et al). An RNN is useful because it can be trained to understand sequential information. At any point in a sequence they can remember all the information they have seen so far. This is often useful in text analysis, as sentences have a structure where the order of words is important. A problem with a generic RNN is the vanishing gradient problem, where the network has trouble remembering long-term dependencies in the data. If two words are too far apart, then the network will not recognize their relationship. This is where an LSTM network comes into play.

Long Short Term Memory neural networks are excellent when used in language modeling, translation, speech recognition, and many other tasks. As a type of RNN, they recognize the order of data, but unlike RNNs, they are designed to avoid a long-term dependency problem. This is especially helpful in recognizing text, as the order of words in a sentence can be important. For our task, we decided an LSTM network would be a good candidate to learn ideology from fairly long sentences.

We built an LSTM Neural Network using the keras (Chollet 2015) library. This library allows us to create a network layer by layer. The first layer is an embedding layer. Instead of using a sparse representation of the words in our sentences, words are represented by dense vectors, where each vector represents the projection of a word into a continuous vector space. The position of a word within the vector space is based on the words that most often surround the word when it is used. The following layer is an LSTM layer, the main part of this network, followed by a dense layer, which provides the prediction.

### Baseline
The baseline for the model is an all-zeros prediction, i.e. we predict all sentences to be of a neutral ideology.

### Model Evaluation
Since this is a regression problem, all models were evaluated using the Mean Squared Error (MSE) metric. The model selection and the reported MSE scores for the models are based on 5-fold cross validation.

## Results
![Model Results](results/model_scores.png)

The Bag of words models barely beat the baseline score of 0.158. The LSTM for all sentences does relatively better at 0.149. The LSTMs trained and tested on sentences with specific keywords related to taxes, drugs, or political terms tend to perform poorly relative to the baseline. This is likely due to the relatively smaller data sizes.

## Conclusion
We attempted to model political ideology of speech based on the known ideology scores of politicians. We used Congressional records data to train and evaluate our models. We used web scraping and pdf mining to obtain our datasets and modeling approaches like bag of words based linear models as well as deep learning to predict ideologies of sentences.

Ultimately, we lacked a sufficient target variable to accurately train our models. The DW-NOMINATE score is averaged for each politician over their tenure in Congress, while our observations were at the sentence level. Previous work in this field tended to use manual annotation to achieve an ideology rating for individual sentences. While there are political debates in Congress, there is also a great deal of procedural speech that is recorded in the Congressional Record. Procedural speeches are unrelated to the political ideology of the speaker. These noisy observations make it difficult to train a model, as the label we used will equate this as the same ideology as speech that is heavily polarized.

However, we were able to create a program that can scrape the Congressional Record and clean up the data. With this program, we can enable other researchers to conduct their own work on this dataset, which can hopefully bring greater understanding to language use in both the House of Representatives and the Senate.

## References
* Chollet, Francois. 2015 Keras: The Python Deep Learning library
* * Hochreiter, Sepp Schmidhuber, Jurgen 2013. Long Short-term Memory Neural computation. 9. 173580. 10.1162/neco.1997.9.8.1735.
* Jain, L. C. and Medsker, L. R.. 1999 Recurrent Neural Networks: Design and Applications
* Lewis, Jeffrey B., Keith Poole, Howard Rosenthal, Adam Boche, Aaron Rudkin, and Luke Sonnet. 2017. Voteview: Congressional Roll-Call Votes Database.
* Shinyama, Yusuke.. 2014 Python pdfminer
* The Congressional Record. 1972-2018. The Congressional Record, Congressional Research Service, Library of Congress, 1972-2018, pp. All 118-163 Congress, all sessions.