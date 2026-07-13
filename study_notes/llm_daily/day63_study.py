"""Day 63: Summarize how real GPT-style pretraining differs from toy training.

A dependency-light, local demonstration for the Day 63 LLM roadmap topic.
It uses a toy-sized example and explicitly labels where real systems need more rigor.
"""

def main():
    toy = {"data": "one short local corpus", "workers": 1, "evaluation": "printed example", "safety": "manual review"}
    production = {"data": "curated large-scale pipeline", "workers": "many synchronized devices", "evaluation": "held-out suites", "safety": "layered process"}
    for key in toy: print(f"{key}: toy={toy[key]!r}; GPT-style pretraining={production[key]!r}")

if __name__ == "__main__":
    main()
