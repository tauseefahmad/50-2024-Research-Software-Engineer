#Project
Two Heuristic based approches were used to classify the papers. One uses Bert and the second one uses Deberta.
# Bert
# Brief Explanation
## Filtering: 
Embedding-based filtering is used to capture semantic context beyond keyword matching.
## Classification: 
Heuristic-based classification identifies if the paper focuses on text mining, computer vision, or both.
## Method Extraction: 
Specific methods like CNN and transformers are extracted using regular expressions.
# Results Summary
The filtered_and_classified_papers.csv has the results.
#Statistics:
Total Relevant Papers: 11221

Classification Counts:
CNN                       490
LSTM                      150
Transformer               114
RNN                        68
RNN, LSTM                  22
CNN, LSTM                  15
CNN, RNN                    7
CNN, Transformer            7
CNN, RNN, LSTM              6
RNN, Transformer            3
CNN, Transformer, LSTM      1
Transformer, LSTM           1
CNN, RNN, Transformer       1
Name: Category, dtype: int64

Method Counts:
other              7392
computer vision    2801
text mining         990
both                 38
Name: Methods Used, dtype: int64


# Deberta






