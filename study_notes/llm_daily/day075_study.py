"""Day 75: Learn task metrics such as accuracy, F1, BLEU, or ROUGE.

A dependency-light, local demonstration for the Day 75 LLM roadmap topic.
It uses a toy-sized example and explicitly labels where real systems need more rigor.
"""

def main():
    # The four confusion-matrix counts distinguish correct positives, false alarms, misses, and correct negatives.
    tp, fp, fn, tn = 8, 2, 1, 9
    # Precision measures trust in positive predictions, recall measures coverage of actual positives, and F1 balances both.
    precision, recall = tp/(tp+fp), tp/(tp+fn)
    # The harmonic mean falls when either precision or recall is low, rather than allowing one to dominate.
    f1 = 2*precision*recall/(precision+recall)
    print({"accuracy": (tp+tn)/(tp+fp+fn+tn), "precision": precision, "recall": recall, "F1": f1})
    print("BLEU/ROUGE are overlap metrics; use task-appropriate human judgment too.")

# Metrics compress quality into a number, but each one captures a different failure mode and should match the task.
if __name__ == "__main__":
    main()
