"""Day 75: Learn task metrics such as accuracy, F1, BLEU, or ROUGE.

A dependency-light, local demonstration for the Day 75 LLM roadmap topic.
It uses a toy-sized example and explicitly labels where real systems need more rigor.
"""

def main():
    tp, fp, fn, tn = 8, 2, 1, 9
    precision, recall = tp/(tp+fp), tp/(tp+fn)
    f1 = 2*precision*recall/(precision+recall)
    print({"accuracy": (tp+tn)/(tp+fp+fn+tn), "precision": precision, "recall": recall, "F1": f1})
    print("BLEU/ROUGE are overlap metrics; use task-appropriate human judgment too.")

if __name__ == "__main__":
    main()
