#  Engineering Portfolio: System Optimization & AI Agents

##  Project 1. E.M.Pilot - On-Device AI Email Client
> **1st Place Winner (Grand Prize)** @ Qualcomm Edge AI Developer Hackathon (2025)

**"Maximizing NPU efficiency for privacy-focused AI on the edge."**

- **Tech Stack:** `Python`, `Flask`, `Qualcomm QNN SDK`, `Qualcomm AI Hub`, `MySQL`
- **Key Achievements:**
  - **On-Device Optimization:** Quantized and optimized 4 AI models (Qwen2-7B, YOLOv8, EasyOCR, Nomic-Embed) for **Snapdragon X Elite NPU**, achieving a **~75% reduction** in inference latency.
  - **Smart Automation:** Implemented strictly local AI features for auto-summarization, intent-based reply generation, and task extraction without cloud dependency.
  - **Efficient Architecture:** Designed a Flask REST API with hash-based MySQL caching to eliminate duplicate processing and minimize API overhead.
- **GitHub:** [Link to E.M.Pilot Repo]

<br>

## Project 2. High-Performance Computing & Kernel Optimization
> **System Software & Parallel Computing Research (Individual Project)**

**"Breaking the Memory Wall: Optimization of Matrix Operations and AI Inference."**

- **Tech Stack:** `C++`, `CUDA`, `OpenMP`, `Linux`, `Google Multichase`
- **Key Achievements:**
  - **GPU Memory Optimization (CUDA):** Implemented **Shared Memory Tiling** for GEMM and utilized **Constant Memory** for Convolution kernels. Solved global memory bottlenecks, achieving **~1.4x speedup** compared to naive implementations. 
  - **CPU Parallelization (OpenMP):** Optimized **Softmax Regression** inference by analyzing scheduling policies. Applied **Dynamic Scheduling** to resolve load imbalance issues, resulting in **1.78x speedup**.
  - **System Profiling:** Conducted granular analysis of **L1/L2/LLC cache latencies** and **NUMA** remote memory access penalties using Google Multichase to identify hardware bottlenecks.
- **GitHub:** [Link to HPC Repo]

<br>

## Project 3. LANEIGE Ranking Insight Agent
> **Amorepacific AI Innovation Challenge 2026**

**"Autonomous Agentic Workflow for Real-time Beauty Insights."**

- **Tech Stack:** `Python`, `FastAPI`, `LangChain`, `RAG`, `ChromaDB`, `Amazon PA-API`
- **Key Achievements:**
  - **Agentic Architecture:** Designed a scalable **FastAPI** backend integrated with **LangChain**, featuring an autonomous decision-making loop with 9 custom tools.
  - **Hybrid Data Pipeline:** Implemented **RAG (Retrieval-Augmented Generation)** using ChromaDB (Vector) and SQLite (Relational) to minimize hallucinations and ensure accurate product data retrieval.
  - **Async Processing:** Developed an asynchronous data collector integrating **Amazon PA-API** to gather real-time global ranking insights without blocking user interactions.
- **GitHub:** [Link to Amore Repo]
