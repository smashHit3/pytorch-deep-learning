"""Day 63: Summarize how real GPT-style pretraining differs from toy training.

A dependency-light, local demonstration for the Day 63 LLM roadmap topic.
It uses a toy-sized example and explicitly labels where real systems need more rigor.
"""

def main():
    # Matching keys make each row a like-for-like contrast between a local lesson and production pretraining.
    toy = {"data": "one short local corpus", "workers": 1, "evaluation": "printed example", "safety": "manual review"}
    # Production values use the identical categories so differences cannot be attributed to a changed rubric.
    production = {"data": "curated large-scale pipeline", "workers": "many synchronized devices", "evaluation": "held-out suites", "safety": "layered process"}
    # Iterating the toy keys keeps the comparison order and coverage identical for both dictionaries.
    for key in toy: print(f"{key}: toy={toy[key]!r}; GPT-style pretraining={production[key]!r}")

# The guarded comparison highlights that production pretraining adds scale, data pipelines, and reliability engineering.
if __name__ == "__main__":
    main()
