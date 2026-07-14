"""Day 100: Relate latency, throughput, batching, and a KV cache."""


def decode_cost(generated_tokens, cached_prefix):
    """Toy attention work: cached keys/values avoid recomputing the prefix."""
    # Without cached projections, each new token attends over a growing prefix; the sum models that repeated work.
    return generated_tokens if cached_prefix else sum(range(1, generated_tokens + 1))


def main():
    # The simple latency formula increases per batch, while throughput divides all requests by shared elapsed time.
    for batch_size in (1, 4, 8):
        latency_ms = 8 + batch_size * 2
        throughput = batch_size * 1000 / latency_ms
        print(f"batch={batch_size}: latency={latency_ms}ms, estimated throughput={throughput:.1f} requests/s")
    # Comparing equal generated lengths isolates the cache assumption rather than a difference in output length.
    print("attention work for 4 generated tokens: no cache=", decode_cost(4, False), "with KV cache=", decode_cost(4, True))
    print("Batching can improve throughput while increasing waiting time; KV cache trades memory for faster decoding.")

# A KV cache reuses prior attention projections, trading memory for lower repeated work during autoregressive decoding.
if __name__ == "__main__":
    main()
