# PagedAttention: Deep Dive & Simulator

This repository contains an interactive HTML simulator demonstrating the **PagedAttention** algorithm, the core innovation behind the **vLLM** inference engine. 

Below is a deep, conceptual breakdown of how PagedAttention works, why it is necessary, and how it is visualized in the simulator.

---

## 1. The Problem: Memory Waste in LLM Serving

When generating text, Large Language Models (LLMs) cache the Key and Value (KV) states of past tokens to avoid recomputing them. This is known as the **KV Cache**. 

In traditional LLM serving systems, memory for the KV cache is allocated **contiguously**. Because the system cannot predict exactly how many tokens a user will generate, it must pre-allocate enough memory to fit the *maximum possible sequence length* (e.g., 2048 or 4096 tokens). 

If a user generates a short response of only 100 tokens, the remaining pre-allocated memory (e.g., 1948 tokens worth of space) is completely wasted. 
* **Internal Fragmentation**: Wasted space inside pre-allocated contiguous chunks.
* **External Fragmentation**: Memory becomes scattered, leaving free chunks that are too small for new contiguous requests.

In traditional systems, **60% to 80% of GPU memory is wasted**. Because GPU memory is the bottleneck for LLM inference, this waste severely limits the batch size (the number of users you can serve simultaneously).

---

## 2. The Solution: PagedAttention

PagedAttention solves this by borrowing a classic concept from Operating Systems: **Virtual Memory and Paging**.

Instead of allocating one giant contiguous chunk of memory per sequence, PagedAttention divides the KV cache into fixed-size chunks called **Blocks** (e.g., a block size of `B = 4` tokens). 

### The Core Architecture

1. **Logical Blocks**: From the perspective of the LLM generation sequence, the tokens exist in perfectly sequential, logical blocks (`L0, L1, L2...`).
2. **Physical Blocks**: The actual GPU memory is divided into a pool of non-contiguous physical blocks (`P0, P1, P2...`). 
3. **Block Table**: A mapping table tracks which Logical Block maps to which Physical Block. 

By decoupling logical memory from physical memory, **the KV cache no longer needs to be contiguous**.

---

## 3. How it Works (Step-by-Step)

The provided HTML simulator demonstrates three critical capabilities of PagedAttention.

### A. Single Generation (Dynamic Allocation)
In standard contiguous allocation, memory is blocked off upfront. In PagedAttention, memory is allocated **dynamically on demand**.
* **Prefill**: The prompt ("Four score and seven years") is broken into blocks of 4 tokens. 
* **Generation**: As the LLM generates tokens one by one ("ago", "our", "fathers"), it fills up the current physical block.
* **New Allocation**: Only when a block is completely full does the system allocate a *new* physical block from the GPU pool for the next token.
* **Result**: Memory waste is limited to, at most, the unused slots in the *final* block (a maximum of `B - 1` tokens). Effective memory utilization jumps from ~25% to **~96%**.

### B. Parallel Sampling (Copy-on-Write)
Often, LLMs generate multiple parallel responses for the same prompt (e.g., generating 3 different drafts). 
* **The Old Way**: The system would copy the entire KV cache of the prompt for all 3 drafts, wasting massive amounts of memory.
* **PagedAttention (CoW)**: Multiple logical sequences map to the *exact same* physical blocks for the shared prompt. The physical blocks simply maintain a **Reference Count** (e.g., `Ref: 3`).
* When the generation diverges (one draft outputs "A", another outputs "B"), the system notices the block is shared, duplicates it into a new physical block, and safely appends the new token. This is called **Copy-on-Write (CoW)**, saving massive amounts of GPU memory.

### C. Beam Search (Dynamic Memory Freeing)
Beam search generates multiple sequences simultaneously but prunes (deletes) the worst-performing branches as it goes.
* As sequences share a history, they share physical blocks (high Reference Counts).
* When a branch is pruned, the Block Table removes its logical-to-physical mappings, and the Reference Count of those physical blocks drops.
* If a Physical Block's Reference Count hits `0`, it is instantly marked as **FREED** and returned to the GPU DRAM pool to be reused immediately by other users. 

### D. CPU Swapping (Preemption)
When the GPU DRAM pool gets entirely full, the system cannot process new requests. 
* Instead of dropping active generations, PagedAttention **preempts** them. 
* It swaps the physical blocks out of expensive GPU DRAM and into cheaper **CPU Swap Space**.
* When space frees up on the GPU, the blocks are seamlessly swapped back into newly available physical blocks, the Block Table is updated, and generation resumes without the LLM ever knowing its memory was moved.

---

## 4. Why This Matters
By completely eliminating internal fragmentation and safely sharing memory across sequences via Copy-on-Write, PagedAttention effectively **doubles or triples the batching capacity** of a single GPU. It is the fundamental reason engines like vLLM can serve open-source models so cheaply and efficiently.
