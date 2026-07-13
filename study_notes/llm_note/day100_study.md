# Day 100: Study latency, throughput, batching, and KV cache.

## Goal
Build a concrete mental model of **Study latency, throughput, batching, and KV cache** and connect it to the larger LLM training or application pipeline.

## Today's Focus
Study latency, throughput, batching, and KV cache. Use the small local exercise to make the idea observable before generalizing it to production-scale LLM work.

## Study Tasks / Hands-on Exercise
1. Read one primary explanation or tutorial section for this topic and write down one assumption it makes.
2. Run `python study_notes/llm_daily/day100_study.py`, inspect every printed value, and change one input or constant.
3. Estimate simple prefill/decode timing for different batch sizes and cache assumptions.
4. Record one observation, one limitation, and one question for the next study session.

## Key Concepts
latency, throughput, batching, prefill, KV cache.

## Mini Challenges
- Explain this day’s idea in three sentences without using jargon.
- Predict the script’s output before running it, then reconcile any difference.
- State one way the toy demonstration differs from a real LLM system.

## Completion Checklist
- [ ] I can define the main concept in my own words.
- [ ] I ran and modified the companion local script.
- [ ] I checked shapes, units, assumptions, or evaluation criteria as applicable.
- [ ] I wrote one limitation and one follow-up question.
