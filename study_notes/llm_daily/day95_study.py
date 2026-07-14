"""Day 95: Apply retrieval, citation, verification, and abstention mitigations."""


def mitigate(question, evidence):
    """Use a small evidence policy instead of inventing an unsupported answer."""
    if not evidence:
        return {"answer": "I cannot verify that from the supplied evidence.", "strategy": "abstain"}
    return {
        "answer": evidence["text"],
        "citation": evidence["source"],
        "strategy": "retrieve -> answer only from evidence -> cite -> verify",
    }


def main():
    evidence = {"source": "local-note-3", "text": "Attention combines queries, keys, and values."}
    print("supported question:", mitigate("What does attention combine?", evidence))
    print("unsupported question:", mitigate("Who invented a new database?", None))
    print("Mitigations reduce risk; they need coverage and factuality evaluation on realistic cases.")

if __name__ == "__main__":
    main()
