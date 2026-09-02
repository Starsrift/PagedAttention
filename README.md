# PagedAttention: The Engine Behind vLLM

> *How OS-inspired memory paging eliminated 60-80% of GPU memory waste in LLM serving.*

This repository provides an educational deep-dive into **PagedAttention**, the core memory-management algorithm that powers high-throughput LLM inference engines like [vLLM](https://github.com/vllm-project/vllm).

---

## 🗂️ Repository Structure

```
PagedAttention/
├── README.md                          ← You are here
├── PagedAttention_Tutorial.ipynb      ← PyTorch implementation (code walkthrough)
├── simulator/
│   ├── index.html                     ← Interactive HTML/JS visual simulator
│   └── README.md                      ← Deep technical explanation of the simulator
└── recordings/
    ├── Gifs/                          ← Drop your GIF assets here
    ├── Single Generation.mp4
    ├── Parallel Sampling.mp4
    └── Beam Search.mp4
```

---

## 🧠 Background: The KV Cache Problem

When an LLM generates text token-by-token, it must attend to every previously generated token. To avoid recomputing the **Key (K)** and **Value (V)** vectors for all past tokens at every step, these vectors are cached in GPU memory. This is called the **KV Cache**.

Without the KV Cache: Generating 100 tokens requires 100 full forward passes, each reprocessing all previous tokens. Computationally catastrophic.

With the KV Cache: Each new token does one small forward pass and reads the cached K, V vectors of the past.

**The catch?** The KV Cache is enormous. For a model like LLaMA-3 70B with long context lengths, a single sequence's KV cache can consume gigabytes of VRAM.

---

## 🔴 The Problem: Contiguous Memory Allocation

Traditional serving systems (like early HuggingFace or FasterTransformer) allocate a single **contiguous** block of GPU memory per request, sized to the maximum possible sequence length.

```
Request A: [████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░] ← used 12 / 40 slots
Request B: [████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░] ← used 4  / 40 slots
Request C: [█████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░] ← used 9  / 40 slots
           ▲          ▲
      Filled      WASTED (pre-allocated, never used)
```

This causes two types of waste:

| Problem | Description |
|---|---|
| **Internal Fragmentation** | Memory is pre-allocated for 2048 tokens, but the sequence only generates 100. The remaining 1948 slots sit idle but locked. |
| **External Fragmentation** | As requests finish, they leave behind gaps of freed memory. New requests cannot use these scattered gaps if they need a larger contiguous block. |

**Result:** In production systems, **60–80% of GPU VRAM is wasted**. This directly caps the number of concurrent users you can serve.

---

## 🟢 The Solution: PagedAttention

PagedAttention, introduced by [Kwon et al. (2023)](https://arxiv.org/abs/2309.06180), borrows a foundational idea from Operating Systems: **Virtual Memory with Paging**.

Instead of one giant contiguous chunk, memory is divided into small, fixed-size **Blocks**. A block holds the KV vectors for a fixed number of tokens (the block size, e.g. B = 4 or B = 16).

### The Three Core Components

| Component | Description |
|---|---|
| **Logical Blocks** | The sequence's tokens are grouped into sequential, logical blocks (L0, L1, L2…). This is the sequence's "view" of memory. |
| **Physical Blocks** | The actual GPU DRAM is partitioned into a pool of physical blocks (P0…P15). These do NOT need to be contiguous. |
| **Block Table** | A per-request lookup table that maps each Logical Block ID → Physical Block ID. This is the bridge between the logical and physical worlds. |

```
Sequence: "Four score and seven years ago our fathers..."
          └─ L0 [Four, score, and, seven] ──→ Block Table ──→ P7 (anywhere in GPU RAM)
          └─ L1 [years, ago, our, fathers] ─→ Block Table ──→ P3 (anywhere in GPU RAM)
          └─ L2 [brought, forth, a, new] ──→ Block Table ──→ P11 (anywhere in GPU RAM)
```

By decoupling logical from physical memory, **the KV Cache no longer needs to be contiguous**.

---

## ⚙️ How PagedAttention Works — Step by Step

### Step 1: Prefill (Processing the Prompt)

When a new request arrives with a prompt, the vLLM **scheduler** processes all prompt tokens at once (the "prefill" phase):

1. The prompt is tokenized: `["Four", "score", "and", "seven", "years"]` → 5 tokens.
2. Block size is B = 4, so 5 tokens require **2 logical blocks**.
3. The scheduler requests 2 physical blocks from the **free block pool** → gets P7 and P3.
4. The Block Table is initialized: `{ L0 → P7, L1 → P3 }`.
5. The 4 prompt tokens fill P7 completely. The 5th token occupies slot 0 of P3 (1/4 filled).

```
After Prefill:
  L0 → P7: [Four | score | and | seven]   ← Full (4/4)
  L1 → P3: [years | ░░░ | ░░░ | ░░░  ]   ← Partially filled (1/4)
```

---

### Step 2: Decode (Autoregressive Token Generation)

The model now generates tokens one by one. For each new token:

1. The model does a **forward pass**, computing the new Query (Q) vector.
2. The **PagedAttention kernel** uses the Block Table to fetch K, V vectors from all allocated physical blocks — even though they are scattered in GPU RAM.
3. Attention scores are computed: `Softmax(Q · K^T / √d) · V`.
4. The newly generated token's K and V vectors are **appended** to the current partially-filled physical block.

```
Decode Step 1: Generate "ago"
  L1 → P3: [years | ago | ░░░ | ░░░]   ← 2/4 filled

Decode Step 2: Generate "our"
  L1 → P3: [years | ago | our | ░░░]   ← 3/4 filled

Decode Step 3: Generate "fathers"
  L1 → P3: [years | ago | our | fathers]  ← Full (4/4)
```

---

### Step 3: Dynamic Block Allocation (On-Demand Growth)

When the current physical block fills up, the system does NOT pre-allocate anything. It waits until needed:

1. Decode Step 4: Generate `"brought"`. Block P3 is full.
2. Scheduler requests a **new** physical block from the free pool → gets P11.
3. Block Table is updated: `{ L0 → P7, L1 → P3, L2 → P11 }`.
4. `"brought"` is written to slot 0 of P11.

```
Dynamic Allocation:
  L0 → P7 : [Four | score | and | seven]           ← Full
  L1 → P3 : [years | ago | our | fathers]           ← Full
  L2 → P11: [brought | ░░░ | ░░░ | ░░░]            ← New block, 1/4 filled
```

> 💡 **Key Insight:** Memory is never wasted on slots that haven't been generated yet. The maximum waste at any moment is `B - 1` slots in the *final* block only.

---

### Step 4: Parallel Sampling with Copy-on-Write (CoW)

When the user requests multiple parallel outputs from the same prompt (e.g., sampling 3 drafts), copying the entire KV cache for each draft would be hugely wasteful.

**PagedAttention uses Block Sharing:**

1. All sampled sequences point their Block Tables to the **same** physical blocks.
2. Each shared physical block maintains a **Reference Count** (Ref Count).
3. After the prompt is processed, the shared blocks have Ref Count = 3.

```
Before Branching:
  Seq-A: L0 → P7 (Ref: 3), L1 → P3 (Ref: 3)
  Seq-B: L0 → P7 (Ref: 3), L1 → P3 (Ref: 3)
  Seq-C: L0 → P7 (Ref: 3), L1 → P3 (Ref: 3)
```

When Seq-A tries to write its unique token to a shared block (Ref Count > 1), **Copy-on-Write** triggers:

1. A **new** physical block is allocated (e.g., P12).
2. The content of the shared block (P3) is **copied** into P12.
3. Seq-A's Block Table is updated: `L1 → P12`.
4. P3's Ref Count drops from 3 to 2.
5. Seq-A's unique token is written to P12 safely.

```
After Branching:
  Seq-A: L0 → P7 (Ref: 2), L1 → P12 [unique token] ← Private copy
  Seq-B: L0 → P7 (Ref: 2), L1 → P3  (Ref: 2)       ← Still shared
  Seq-C: L0 → P7 (Ref: 2), L1 → P3  (Ref: 2)       ← Still shared
```

---

### Step 5: Beam Search with Garbage Collection

In Beam Search, multiple candidate sequences compete, and low-probability branches are periodically **pruned**:

1. All beam candidates share historical blocks via high Ref Counts.
2. When a beam is pruned (evicted by the scheduler):
   - Its Block Table entries are removed.
   - The Ref Count of each of its physical blocks is **decremented**.
3. When a physical block's Ref Count drops to **0**, it is immediately returned to the **free block pool** — instantly usable by other requests.

```
Before Pruning:
  Beam-1: L0 → P7, L1 → P12   (Ref Counts: P7=4, P12=1)
  Beam-2: L0 → P7, L1 → P14   (Ref Counts: P7=4, P14=1)
  Beam-3: L0 → P7, L1 → P5    (Ref Counts: P7=4, P5=1)
  Beam-4: L0 → P7, L1 → P3    (Ref Counts: P7=4, P3=1)

After Pruning Beam-3 and Beam-4:
  P5 Ref Count → 0: FREED ♻️
  P3 Ref Count → 0: FREED ♻️
  P7 Ref Count → 2 (still alive, shared by Beam-1 and Beam-2)
```

---

### Step 6: Preemption (CPU Swapping)

When GPU DRAM is completely exhausted and no blocks can be freed:

1. The vLLM scheduler identifies low-priority active sequences.
2. Their physical KV blocks are **swapped out** from GPU DRAM to CPU RAM.
3. The GPU blocks are freed and allocated to higher-priority or incoming requests.
4. When the preempted sequences are resumed, their blocks are **swapped back in** to newly freed GPU blocks and the Block Table is updated transparently.

The LLM model never "knows" its memory was moved — only the scheduler and Cache Engine are aware.

---

## 📊 Impact at a Glance

| Metric | Contiguous Allocation | PagedAttention |
|---|---|---|
| Memory Utilization | ~25% (75% wasted) | ~96% |
| Internal Fragmentation | High (entire max-length blocks wasted) | Near zero (max B-1 slots wasted) |
| External Fragmentation | High | Eliminated (fixed-size blocks always fit) |
| Memory Sharing | Not possible | Native (via Ref Counts + CoW) |
| Throughput (req/s) | Baseline | **2-4x higher** |

---

## 🎮 Interactive Simulator

Open [`simulator/index.html`](./simulator/index.html) directly in your browser (no server needed).

### Simulator Modes

| Mode | What it shows |
|---|---|
| **Single Generation** | Dynamic block allocation as tokens are generated one by one |
| **Parallel Sampling** | Block sharing with Ref Counts, and the Copy-on-Write event when sequences diverge |
| **Beam Search** | Shared block trees, pruning events, and live memory garbage collection |
| **Preempt Button** | Simulates CPU swapping, showing blocks migrating from GPU DRAM to CPU Swap Space |

### Demo Animations

**Single Generation** — Dynamic block allocation as tokens fill physical blocks:

![Single Generation](./recordings/Gifs/Single_Generation.gif)
> *GIF placeholder — add `Single_Generation.gif` to `recordings/Gifs/`*

**Parallel Sampling** — Block sharing and Copy-on-Write in action:

![Parallel Sampling](./recordings/Gifs/Parallel_Sampling.gif)
> *GIF placeholder — add `Parallel_Sampling.gif` to `recordings/Gifs/`*

**Beam Search** — Candidate pruning and instant memory reclamation:

![Beam Search](./recordings/Gifs/Beam_Search.gif)
> *GIF placeholder — add `Beam_Search.gif` to `recordings/Gifs/`*

---

## 💻 Python Implementation

[`PagedAttention_Tutorial.ipynb`](./PagedAttention_Tutorial.ipynb) walks through the algorithm in PyTorch:

- **`KVCacheAllocator`** — Pre-allocated physical block pool with `allocate_block()` / `free_block()` methods.
- **`SequenceState`** — Per-request block table with `append_token()` that auto-allocates new blocks on overflow.
- **`paged_attention()`** — Fetches scattered KV blocks using the block table and computes Scaled Dot-Product Attention.

```bash
# Run the notebook
jupyter notebook PagedAttention_Tutorial.ipynb
```

---

## 📚 References

- [Efficient Memory Management for Large Language Model Serving with PagedAttention (Kwon et al., 2023)](https://arxiv.org/abs/2309.06180)
- [vLLM Project](https://github.com/vllm-project/vllm)
- [vLLM Blog Post](https://blog.vllm.ai/2023/06/20/vllm.html)

---

*Built as an educational companion for a Substack/Medium technical post on LLM serving infrastructure.*
