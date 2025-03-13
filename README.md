
# Project
The specific goal is to identify papers that implement deep learning neural network-based solutions in the fields of virology and epidemiology.
As the LLM require many hours of computation, so after trying LLM based approach, i implemented the heuristic based approach. Two heuristic based approches were used to classify the papers. One uses Bert and the second one uses Deberta.

## Brief Explanation of Steps 
### Filtering: 
Embedding-based filtering is used to capture semantic context beyond keyword matching.
### Classification: 
Heuristic-based classification identifies if the paper focuses on text mining, computer vision, or both.
### Method Extraction: 
Specific methods like CNN and transformers are extracted using regular expressions.

Code, Statistics and Results are in the Heuristic Method Folder. The code was made using Jupyter Notebook. For ease of executing code, separate files for python are also provided. 
