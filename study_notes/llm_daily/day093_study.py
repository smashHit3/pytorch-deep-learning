"""Day 93: Study DPO at a conceptual level.

A dependency-light, local demonstration for the Day 93 LLM roadmap topic.
It uses a toy-sized example and explicitly labels where real systems need more rigor.
"""

import math

def dpo_style_margin(chosen, rejected): return -math.log(1 / (1 + math.exp(-(chosen - rejected))))

def main():
    print("toy chosen/rejected loss:", round(dpo_style_margin(1.2, 0.4), 4))
    print("The pair says which response is preferred; real DPO also uses a reference policy and batched training.")

if __name__ == "__main__":
    main()
