# PagedAttention: The Engine Behind vLLM

This repository provides an educational deep-dive into **PagedAttention**, the core memory-management algorithm that powers high-throughput LLM inference engines like [vLLM](https://github.com/vllm-project/vllm).

## 🗂️ Repository Structure

- 🎮 **`simulator/index.html`**: A highly interactive, single-file HTML/Vanilla JS visualizer demonstrating PagedAttention in real-time.
- 🐍 **`PagedAttention_Tutorial.ipynb`**: A step-by-step PyTorch implementation reconstructing the Block Table, Memory Pool, and PagedAttention kernel from scratch.
- 🎥 **`recordings/`**: Contains MP4s and GIFs capturing the simulator's animations for use in articles or blogs.

---

## 📖 Visualizing PagedAttention

Traditional LLM serving systems suffer from **60-80% memory waste** due to pre-allocating contiguous memory for the Key-Value (KV) cache. 

PagedAttention borrows from Operating System **Virtual Memory and Paging**. It divides the KV cache into fixed-size **Blocks** and maps logical sequence blocks to scattered physical GPU memory blocks using a **Block Table**.

Here is how it works in practice, as shown in the simulator:

### 1. Single Generation (Dynamic Allocation)
Instead of pre-allocating max sequence lengths, PagedAttention allocates physical blocks dynamically on-demand. Memory waste is limited to the unused slots in the *final* block, boosting utilization to ~96%.

![Single Generation Demo](./recordings/Gifs/Single_Generation.gif)
> *Placeholder for Single Generation GIF*

### 2. Parallel Sampling (Copy-on-Write)
When generating multiple responses from a single prompt, sequences share the prompt's physical blocks. The Cache Engine uses **Copy-on-Write (CoW)** to safely duplicate shared blocks only when the generations diverge, saving massive amounts of memory.

![Parallel Sampling Demo](./recordings/Gifs/Parallel_Sampling.gif)
> *Placeholder for Parallel Sampling GIF*

### 3. Beam Search (Dynamic Memory Freeing)
As beam search prunes low-probability branches, the reference counts of shared physical blocks drop. When a block's reference count hits `0`, it is instantly freed back to the active GPU DRAM pool.

![Beam Search Demo](./recordings/Gifs/Beam_Search.gif)
> *Placeholder for Beam Search GIF*

---

## 💻 Code-Level Implementation (PyTorch)

Check out [`PagedAttention_Tutorial.ipynb`](./PagedAttention_Tutorial.ipynb) for a simplified PyTorch implementation demonstrating:
1. `KVCacheAllocator`: The global GPU physical memory pool.
2. `SequenceState`: Block Tables maintaining logical-to-physical mappings.
3. `paged_attention()`: Reconstructing contiguous K/V tensors from scattered physical blocks to compute standard Scaled Dot-Product Attention.

---

*This educational companion was built to demystify the memory management breakthroughs driving modern AI infrastructure.*
