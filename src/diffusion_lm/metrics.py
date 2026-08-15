import re


def tokenize_for_metrics(text):
    return re.findall(r"\w+|[^\w\s]", text.lower())


def distinct_n(text, n):
    tokens = tokenize_for_metrics(text)
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    return len(set(ngrams)) / len(ngrams)


def repetition_rate(text, n=2):
    return 1.0 - distinct_n(text, n)


def calculate_metrics(text):
    return {
        "Distinct-1": distinct_n(text, 1),
        "Distinct-2": distinct_n(text, 2),
        "Repetition-2": repetition_rate(text, 2),
        "Distinct-3": distinct_n(text, 3),
        "Repetition-3": repetition_rate(text, 3),
    }
