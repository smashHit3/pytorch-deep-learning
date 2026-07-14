"""Day 95: Apply retrieval, citation, verification, and abstention mitigations."""


def mitigate(question, evidence):
    """Use a small evidence policy instead of inventing an unsupported answer."""
    # Missing evidence is an explicit abstention path rather than an invitation to synthesize a plausible answer.
    if not evidence:
        return {"answer": "I cannot verify that from the supplied evidence.", "strategy": "abstain"}
    # Supported responses copy only supplied text and retain source attribution for later verification.
    return {
        "answer": evidence["text"],
        "citation": evidence["source"],
        "strategy": "retrieve -> answer only from evidence -> cite -> verify",
    }


def main():
    # The two calls contrast an evidence-backed question with one that has no matching local source.
    evidence = {"source": "local-note-3", "text": "Attention combines queries, keys, and values."}
    print("supported question:", mitigate("What does attention combine?", evidence))
    print("unsupported question:", mitigate("Who invented a new database?", None))
    print("Mitigations reduce risk; they need coverage and factuality evaluation on realistic cases.")

# Citations, verification, and abstention give an answer explicit ways to signal uncertainty instead of inventing support.
if __name__ == "__main__":
    main()
