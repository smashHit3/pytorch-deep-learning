"""Day 93: Study DPO at a conceptual level.

A dependency-light, local demonstration for the Day 93 LLM roadmap topic.
It uses a toy-sized example and explicitly labels where real systems need more rigor.
"""

import math

# This is negative log-sigmoid of the chosen-minus-rejected score, so larger preferred margins reduce loss.
def dpo_style_margin(chosen, rejected): return -math.log(1 / (1 + math.exp(-(chosen - rejected))))

def main():
    # The chosen score exceeds the rejected score, representing one annotated preference pair in toy form.
    # The positive score margin makes the negative log-sigmoid small because the preferred response ranks higher.
    print("toy chosen/rejected loss:", round(dpo_style_margin(1.2, 0.4), 4))
    print("The pair says which response is preferred; real DPO also uses a reference policy and batched training.")

# DPO compares preferred and rejected responses directly, avoiding the separate reward-model optimization used in RLHF.
if __name__ == "__main__":
    main()
