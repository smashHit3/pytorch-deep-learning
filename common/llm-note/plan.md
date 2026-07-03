# LLM Study Roadmap

## Goal
Understand LLM fundamentals, build small working systems, and progress toward training, evaluation, and deployment.

## 16-Week Roadmap

### Weeks 1-2: Core Prerequisites
- Learn Python and PyTorch basics.
- Review linear algebra, probability, gradients, and optimization.
- Focus on tensors, backpropagation, softmax, and model training loops.

### Weeks 3-4: NLP and Transformer Basics
- Study tokenization, embeddings, positional encoding, and self-attention.
- Understand encoder-decoder structure and language modeling objectives.
- Read *Attention Is All You Need* and at least one annotated Transformer tutorial.

### Weeks 5-6: Implement from Scratch
- Build a mini Transformer in PyTorch.
- Implement tokenizer, single-head attention, multi-head attention, and a tiny language model.
- Train on a small text dataset to understand the full training loop.

### Weeks 7-8: Pretraining and Scaling
- Learn causal language modeling, masked language modeling, and loss behavior.
- Study batch size, learning rate schedules, data quality, and scaling laws.
- Understand how GPT-style pretraining works in practice.

### Weeks 9-10: Fine-Tuning and Adaptation
- Study supervised fine-tuning, instruction tuning, LoRA, and QLoRA.
- Learn prompt engineering and task-specific evaluation.
- Fine-tune a small open model for Q&A or summarization.

### Weeks 11-12: Retrieval and Agents
- Learn embeddings, vector databases, RAG pipelines, tool use, and function calling.
- Understand memory and orchestration patterns for LLM apps.
- Build a document chatbot with retrieval.

### Weeks 13-14: Alignment and Evaluation
- Study RLHF, DPO, safety, hallucinations, benchmarks, and human evaluation.
- Practice structured error analysis instead of relying on intuition alone.
- Compare models or prompts using a repeatable evaluation setup.

### Weeks 15-16: Inference and Deployment
- Learn quantization, batching, caching, latency, throughput, and cost tradeoffs.
- Explore deployment tools such as vLLM, TGI, or Ollama.
- Deploy a local or cloud LLM application.

## Daily Study Plan

### Week 1: Python and PyTorch Basics
- Day 1: Install or verify Python, PyTorch, Jupyter, and VS Code; review variables, loops, functions, and write one small tensor script.
- Day 2: Learn Python lists, dicts, classes, and file I/O; rewrite yesterday's script more cleanly.
- Day 3: Study NumPy arrays and tensor shapes; practice reshape, transpose, slice, and broadcasting.
- Day 4: Learn PyTorch tensors, dtype, device, and basic math operations.
- Day 5: Study autograd; compute gradients for a tiny linear model.
- Day 6: Build a minimal training loop with dummy data.
- Day 7: Review the week and write short notes on tensors, gradients, and training loops.

### Week 2: Math and Optimization
- Day 8: Review vectors, matrices, dot products, and matrix multiplication.
- Day 9: Study derivatives, partial derivatives, and gradient descent intuition.
- Day 10: Learn probability basics: distributions, expectation, and variance.
- Day 11: Study softmax, cross-entropy, and why classification uses them.
- Day 12: Learn optimization concepts: learning rate, batch size, and epochs.
- Day 13: Compare SGD and Adam in a toy example.
- Day 14: Summarize backpropagation and optimization in your own words.

### Week 3: Deep Learning Building Blocks
- Day 15: Study linear regression and implement it in PyTorch.
- Day 16: Study logistic regression and binary classification.
- Day 17: Learn multilayer perceptrons and activation functions.
- Day 18: Train a small MLP on a simple dataset.
- Day 19: Study overfitting, validation, and train/test split.
- Day 20: Learn regularization, dropout, and weight decay.
- Day 21: Review and document the full supervised learning pipeline.

### Week 4: Sequence Modeling Foundations
- Day 22: Learn what makes text sequential data different from tabular data.
- Day 23: Study tokenization basics: word, subword, and character tokenization.
- Day 24: Learn embeddings and why token IDs need vector representations.
- Day 25: Study RNN intuition and limitations.
- Day 26: Learn LSTM and GRU at a high level.
- Day 27: Build a tiny character-level sequence model or inspect one tutorial.
- Day 28: Write notes on why Transformers replaced recurrent models.

### Week 5: Attention and Transformer Intuition
- Day 29: Study the problem attention is trying to solve.
- Day 30: Learn query, key, and value intuition with a tiny example.
- Day 31: Implement single-head self-attention with small tensors.
- Day 32: Study scaled dot-product attention step by step.
- Day 33: Learn multi-head attention and why multiple heads help.
- Day 34: Study residual connections and layer normalization.
- Day 35: Review all Transformer submodules in one diagram.

### Week 6: Transformer Architecture
- Day 36: Study positional encoding and why order information is needed.
- Day 37: Read the encoder-decoder overview in *Attention Is All You Need*.
- Day 38: Learn feed-forward blocks and attention masks.
- Day 39: Study causal masking for autoregressive generation.
- Day 40: Re-implement a Transformer block in PyTorch.
- Day 41: Trace tensor shapes through the full block.
- Day 42: Summarize the full forward pass of a Transformer.

### Week 7: Language Modeling
- Day 43: Learn next-token prediction and autoregressive modeling.
- Day 44: Prepare a tiny text dataset for language modeling.
- Day 45: Build a vocabulary or simple tokenizer.
- Day 46: Train a tiny language model on a small corpus.
- Day 47: Learn greedy decoding, sampling, and temperature.
- Day 48: Generate text and compare decoding strategies.
- Day 49: Write notes on why loss goes down but generation can still be bad.

### Week 8: Mini GPT Project
- Day 50: Study a small implementation such as nanoGPT or minGPT.
- Day 51: Implement embeddings, positional embeddings, and input pipeline.
- Day 52: Add attention and feed-forward blocks.
- Day 53: Add training loop, optimizer, and evaluation.
- Day 54: Train the mini GPT on a tiny dataset.
- Day 55: Debug shape issues, masking, and sampling behavior.
- Day 56: Document the architecture and training observations.

### Week 9: Pretraining Concepts
- Day 57: Study causal LM versus masked LM objectives.
- Day 58: Learn why dataset quality matters more than raw size in many cases.
- Day 59: Study scaling laws and model/data/compute tradeoffs.
- Day 60: Learn learning rate warmup, decay, and gradient clipping.
- Day 61: Study checkpoints, validation curves, and training instability.
- Day 62: Read about distributed training at a high level.
- Day 63: Summarize how real GPT-style pretraining differs from toy training.

### Week 10: Fine-Tuning
- Day 64: Learn the difference between base models and instruction-tuned models.
- Day 65: Study supervised fine-tuning datasets and formatting.
- Day 66: Learn LoRA and parameter-efficient fine-tuning.
- Day 67: Study QLoRA and quantized fine-tuning concepts.
- Day 68: Run or inspect a small Hugging Face fine-tuning example.
- Day 69: Fine-tune a small model for summarization or Q&A.
- Day 70: Evaluate outputs and write down common failure cases.

### Week 11: Prompting and Evaluation
- Day 71: Study zero-shot, few-shot, and chain-of-thought prompting.
- Day 72: Learn structured outputs and output constraints.
- Day 73: Build a small prompt comparison set for one task.
- Day 74: Measure prompt quality using consistency and error analysis.
- Day 75: Learn task metrics such as accuracy, F1, BLEU, or ROUGE.
- Day 76: Study hallucination patterns and when prompts fail.
- Day 77: Write a prompt engineering checklist for future projects.

### Week 12: Embeddings and Retrieval
- Day 78: Learn what embeddings are and how semantic similarity works.
- Day 79: Generate embeddings for a small document set.
- Day 80: Study cosine similarity and nearest-neighbor search.
- Day 81: Learn vector database basics and retrieval workflow.
- Day 82: Build a minimal retrieval pipeline over local notes or documents.
- Day 83: Connect retrieval with an LLM for question answering.
- Day 84: Evaluate retrieval quality and improve chunking strategy.

### Week 13: RAG and Tool Use
- Day 85: Learn the architecture of retrieval-augmented generation.
- Day 86: Study chunking, metadata, and context packing.
- Day 87: Improve the retriever or prompt template in your RAG app.
- Day 88: Learn tool calling and function calling basics.
- Day 89: Add one tool or external function to your app.
- Day 90: Study simple agent loops and orchestration patterns.
- Day 91: Write notes on when to use plain prompting, RAG, or tools.

### Week 14: Alignment and Safety
- Day 92: Learn the goal of RLHF and preference optimization.
- Day 93: Study DPO at a conceptual level.
- Day 94: Learn safety categories: toxicity, prompt injection, privacy, and misuse.
- Day 95: Study hallucination mitigation strategies.
- Day 96: Build a small human evaluation sheet for your project outputs.
- Day 97: Compare several outputs and rank them with clear criteria.
- Day 98: Summarize alignment versus capability in simple language.

### Week 15: Inference and Serving
- Day 99: Learn the inference pipeline from tokens to generated text.
- Day 100: Study latency, throughput, batching, and KV cache.
- Day 101: Learn quantization basics and memory tradeoffs.
- Day 102: Run a small open model locally with Ollama, vLLM, or another tool.
- Day 103: Compare CPU versus GPU serving tradeoffs.
- Day 104: Study API serving patterns and request limits.
- Day 105: Document the serving stack you would choose for a small app.

### Week 16: Deployment and Portfolio Project
- Day 106: Pick one final project: chatbot, summarizer, tutor, or QA system.
- Day 107: Define scope, dataset, evaluation method, and demo goal.
- Day 108: Build the first end-to-end version.
- Day 109: Improve reliability, prompt quality, or retrieval quality.
- Day 110: Add logging, simple evaluation, and usage instructions.
- Day 111: Deploy locally or to a small cloud target.
- Day 112: Write a final summary of what you learned and what to study next.


## Milestone Projects
1. Tiny GPT from scratch
2. Fine-tuned domain assistant
3. RAG chatbot
4. Evaluation dashboard
5. Deployed LLM service

## Study Loop
1. Read one concept
2. Reproduce one tutorial or paper idea
3. Build one small project
4. Write a short summary of what worked and what failed

## Recommended Resources
- Dive into Deep Learning
- CS224n
- The Illustrated Transformer
- Hugging Face Course
- Andrej Karpathy videos
- nanoGPT / minGPT
- Papers With Code

## Target Outcome
After about 4 months, you should be able to explain Transformer internals, fine-tune a small model, build a RAG application, and deploy an LLM-backed product.
