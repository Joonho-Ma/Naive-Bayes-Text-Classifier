
"""
UIC CS 411 - Project 03: Naive Bayes Text Classifier from Scratch
"""

import os
import csv
import math
import re

"""
train_model(train_file="data/train.csv")
1. tokenization
2. bag-of-words
3. vocabs
4. class-priors(positive or negative)
5. laplace smoothing
6. store the result to the train_result dictionary
"""
train_result = {} # empty dictionary to store trained result


def train_model(train_file="data/train.csv"):
    """Train the Naive Bayes classifier on the training dataset."""
    # Initialize counts and vocabulary
    class_counts = {"positive": 0, "negative": 0}
    word_counts = {"positive": {}, "negative": {}}
    total_words = {"positive": 0, "negative": 0}
    vocabulary = set() # store all unique words found in the training dataset (duplicates removed)

    # read training data
    with open(train_file, "r", encoding="utf-8-sig", newline="") as file:
        csv_reader = csv.DictReader(file)

        for row in csv_reader:
            # read each row, find the label, and make it lowercase
            label = row["label"].strip()
            text = row["text"].lower()

            # remove URLs
            text = re.sub(r"https?://\S+|www\.\S+", " ", text)
            # Handle negative contractions
            text = re.sub(r"\bcan't\b", "can not", text)
            text = re.sub(r"\bwon't\b", "will not", text)
            text = re.sub(r"n't\b", " not", text)

            # remove possessive "'s", apostrophes
            text = re.sub(r"'s\b", "", text)
            text = text.replace("'", "")

            # Then, extract word tokens
            tokens = re.findall(r"[a-z0-9]+", text)

            class_counts[label] += 1 # count sentences and word occurrences

            # Bag-of-Words: ignore order, count repeated words
            for word in tokens:
                vocabulary.add(word)
                word_counts[label][word] = word_counts[label].get(word, 0) + 1
                total_words[label] += 1

    # Check that both classes and vocabulary are not empty
    if not all(class_counts.values()) or not vocabulary:
        raise ValueError("Training data must contain both classes and words.")

    total_sentences = sum(class_counts.values())
    priors = {}
    word_probabilities = {"positive": {}, "negative": {}}

    for label in class_counts:
        priors[label] = class_counts[label] / total_sentences # calculate class prior
        denominator = total_words[label] + len(vocabulary) # Laplace smoothing
        for word in vocabulary:
            # P(w | c) = (count(w, c) + 1) / (N_c + |V|)
            word_probabilities[label][word] = (word_counts[label].get(word, 0) + 1) / denominator

    # Replace previous training data
    train_result.clear()
    train_result.update({
        "vocabulary": vocabulary,
        "class_counts": class_counts,
        "word_counts": word_counts,
        "total_words": total_words,
        "priors": priors,
        "word_probabilities": word_probabilities,
    })
    return train_result


def predict(text, show_scores=False):
    """
    Predict sentiment class for an input sentence.
    Returns: 'positive' or 'negative'
    Tie-breaking: If scores are tied, select the class with the higher prior P(c),
                  or default to 'positive' if priors are equal.
    """
    # 0. Check if the train_result exists
    if not train_result:
        raise ValueError("Model not trained yet. Or error in train_model()")

    vocabulary = train_result["vocabulary"]
    class_priors = train_result["priors"]
    word_probabilities = train_result["word_probabilities"]

    # 1. Tokenization (same as train_model)
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\bcan't\b", "can not", text)
    text = re.sub(r"\bwon't\b", "will not", text)
    text = re.sub(r"n't\b", " not", text)
    text = re.sub(r"'s\b", "", text)
    text = text.replace("'", "")
    tokens = re.findall(r"[a-z0-9]+", text)

    # 2. Initialize class scores using log priors
    scores = {"positive": math.log(class_priors["positive"]),
              "negative": math.log(class_priors["negative"])
              }

    # 3. Add log likelihood for each known token
    for each_word in tokens:
        if each_word in vocabulary:
            for label in scores:
                scores[label] += math.log(word_probabilities[label][each_word])

    # Presentation: display tokens and log scores
    if show_scores:
        print("Tokens:", tokens)
        print("Ignored words:", [word for word in tokens if word not in vocabulary])
        print(f"Positive log score: {scores['positive']:.6f}")
        print(f"Negative log score: {scores['negative']:.6f}")

    # 4. Compare scores and return the higher-scoring class
    if scores["positive"] > scores["negative"]:
        return "positive"
    elif scores["negative"] > scores["positive"]:
        return "negative"

    # (Tie-breaking): when scores are the same, then compare class priors, then default to positive
    if class_priors["positive"] >= class_priors["negative"]:
        return "positive"
    return "negative"


def evaluate_model(test_file="data/test.csv"):
    """
    Evaluate classifier performance on a test dataset.
    Returns:
        dict: A dictionary containing at least:
            {
                "accuracy": float,
                "confusion_matrix": {"TP": int, "TN": int, "FP": int, "FN": int}
            }
    """
    # 0. Initialize variables
    result_dict = {
        "accuracy": 0.0,
        "confusion_matrix": {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
    }
    total_count = 0

    # 1. Read test data file
    with open(test_file, "r", encoding="utf-8-sig", newline="") as file:
        csv_reader = csv.DictReader(file)

        for row in csv_reader:
            actual = row["label"].strip()
            predicted = predict(row["text"])

            # Compare actual and predicted labels
            if actual == "positive":
                if predicted == "positive":
                    result_dict["confusion_matrix"]["TP"] += 1 # True Positive
                else:
                    result_dict["confusion_matrix"]["FN"] += 1 # False Negative
            elif actual == "negative":
                if predicted == "negative":
                    result_dict["confusion_matrix"]["TN"] += 1 # True Negative
                else:
                    result_dict["confusion_matrix"]["FP"] += 1 # False Positive
            else:
                raise ValueError("Invalid Label")

            total_count += 1
    # Error-Checking: when total_count is 0, then raise an error
    if total_count == 0:
        raise ValueError("Total count 0, **empty dateset**")

    # Calculate accuracy
    result_dict["accuracy"] = (result_dict["confusion_matrix"]["TP"] + result_dict["confusion_matrix"]["TN"]) / total_count

    return result_dict


if __name__ == "__main__":
    train_model()
    # Evaluation call added
    result = evaluate_model()

    print("Accuracy:", result["accuracy"])
    print("Confusion Matrix:", result["confusion_matrix"])

    # training statistics and word likelihood calculations for presentation
    print("\n--- Training Statistics ---")
    print("Vocabulary size:", len(train_result["vocabulary"]))
    for label in ("positive", "negative"):
        print(f"{label}: sentences={train_result['class_counts'][label]}, "
              f"total words={train_result['total_words'][label]}, "
              f"prior={train_result['priors'][label]:.6f}")

    for word in ("great", "not"):
        print(f"\nWord: {word}")
        for label in ("positive", "negative"):
            count = train_result["word_counts"][label].get(word, 0)
            denominator = train_result["total_words"][label] + len(train_result["vocabulary"])
            probability = train_result["word_probabilities"][label].get(word)
            print(f"{label}: count={count}, P=({count}+1)/{denominator}={probability:.6f}")

    # Interactive prediction loop
    print("Type 'exit' to quit.")
    while True:
        try:
            text = input("\nTest Your Sentence: ")
        except EOFError:
            break

        if text.strip().lower() == "exit":
            break

        if not text.strip():
            continue

        prediction = predict(text, show_scores=True)
        print("Prediction:", prediction)
