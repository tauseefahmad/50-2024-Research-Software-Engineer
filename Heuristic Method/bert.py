#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import necessary libraries

import pandas as pd
import re
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import torch

# Load dataset
def load_data(file_path):
    return pd.read_csv(file_path)

# Text Preprocessing
def preprocess_text(text):
    if isinstance(text, str):
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())
        return text
    return ""

# Initialize BERT model and tokenizer for embedding generation
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModel.from_pretrained("distilbert-base-uncased")

# Generate embedding for a single sentence
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach()

# Filter relevant papers using embeddings and cosine similarity
def filter_relevant_papers(df, threshold=0.7):
    target_phrases = ["deep learning in virology", "neural networks for epidemiology"]
    target_embedding = sum([get_embedding(phrase) for phrase in target_phrases]) / len(target_phrases)

    relevant_papers = []
    for idx, row in df.iterrows():
        title_embedding = get_embedding(preprocess_text(row["Title"]))
        abstract_embedding = get_embedding(preprocess_text(row["Abstract"]))
        combined_embedding = (title_embedding + abstract_embedding) / 2
        similarity = cosine_similarity(combined_embedding, target_embedding)
        if similarity >= threshold:
            relevant_papers.append(row)
    return pd.DataFrame(relevant_papers)

# Classify papers into text mining, computer vision, both, or other
def classify_methodology(row):
    text_mining_keywords = ["nlp", "text mining", "topic modeling", "language model", "bert"]
    computer_vision_keywords = ["image", "segmentation", "cnn", "convolutional neural network"]

    text_matches = any(word in row["Abstract"].lower() or word in row["Title"].lower() for word in text_mining_keywords)
    vision_matches = any(word in row["Abstract"].lower() or word in row["Title"].lower() for word in computer_vision_keywords)

    if text_matches and vision_matches:
        return "both"
    elif text_matches:
        return "text mining"
    elif vision_matches:
        return "computer vision"
    else:
        return "other"

def classify_papers(df):
    df["Methods Used"] = df.apply(classify_methodology, axis=1)
    return df

# Extract specific deep learning method names used in each paper
def extract_methods(row):
    methods = []
    method_patterns = {
        "CNN": r"\b(CNN|convolutional neural network)\b",
        "RNN": r"\b(RNN|recurrent neural network)\b",
        "Transformer": r"\b(transformer|attention-based model)\b",
        "LSTM": r"\b(LSTM|long short-term memory)\b"
    }
    for method, pattern in method_patterns.items():
        if re.search(pattern, row["Abstract"], re.IGNORECASE) or re.search(pattern, row["Title"], re.IGNORECASE):
            methods.append(method)
    return ", ".join(methods)

def extract_methods_from_papers(df):
    df["Category"] = df.apply(extract_methods, axis=1)
    return df

# Main processing function that runs the pipeline
def process_papers(file_path, output_path):
    # Load data
    df = load_data(file_path)
    
    # Preprocess text fields
    df["Title"] = df["Title"].apply(preprocess_text)
    df["Abstract"] = df["Abstract"].apply(preprocess_text)

    # Filter relevant papers
    df = filter_relevant_papers(df)
    
    # Classify methods
    df = classify_papers(df)

    # Extract methods used
    df = extract_methods_from_papers(df)

    # Save the result
    df.to_csv(output_path, index=False)
    print("Processing complete. Results saved to", output_path)

# Summary statistics function to provide an overview of the filtered data
def dataset_statistics(df):
    print("Total Relevant Papers:", len(df))
    print("\nClassification Counts:")
    print(df["Category"].value_counts())
    print("\nMethod Counts:")
    print(df["Methods Used"].value_counts())

# Run the full pipeline and print statistics
def main():
    # Specify input and output file paths
    file_path = "collection_with_abstracts.csv"  # Replace with the path to your input dataset
    output_path = "filtered_and_classified_papers.csv"

    # Run the processing function
    process_papers(file_path, output_path)
    
    # Load the results and display statistics
    df_result = pd.read_csv(output_path)
    dataset_statistics(df_result)

# Execute main function
if __name__ == "__main__":
    main()


# In[ ]:




