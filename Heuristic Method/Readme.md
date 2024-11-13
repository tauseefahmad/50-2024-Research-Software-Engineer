# Project
The specific goal is to identify papers that implement deep learning neural network-based solutions in the fields of virology and epidemiology.
As the LLM require many hours of computation, so after trying LLM based approach, i implemented the heuristic based approch. Two heuristic based approches were used to classify the papers. One uses Bert and the second one uses Deberta.

## Brief Explanation of Steps 
### Filtering: 
Embedding-based filtering is used to capture semantic context beyond keyword matching.
### Classification: 
Heuristic-based classification identifies if the paper focuses on text mining, computer vision, or both.
### Method Extraction: 
Specific methods like CNN and transformers are extracted using regular expressions.

# Bert
## Results Summary
The results have been saved in filtered_and_classified_papers.csv.
## Statistics:
 Research Paper Classification Summary

 Total Relevant Papers
- **11221**

 Classification Counts
| Category                   | Count |
|----------------------------|-------|
| CNN                        | 490   |
| LSTM                       | 150   |
| Transformer                | 114   |
| RNN                        | 68    |
| RNN, LSTM                  | 22    |
| CNN, LSTM                  | 15    |
| CNN, RNN                   | 7     |
| CNN, Transformer           | 7     |
| CNN, RNN, LSTM             | 6     |
| RNN, Transformer           | 3     |
| CNN, Transformer, LSTM     | 1     |
| Transformer, LSTM          | 1     |
| CNN, RNN, Transformer      | 1     |

 Method Counts
| Methods Used    | Count |
|-----------------|-------|
| other           | 7392  |
| computer vision | 2801  |
| text mining     | 990   |
| both            | 38    |








# Deberta
## Results Summary
The results have been saved in filtered_and_classified_papers.csv.
## Statistics:






