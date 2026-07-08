#  Engineering Portfolio: On-Device AI, System Optimization, AI Agents & NLP

##  Project 1. E.M.Pilot - On-Device AI Email Client
> **1st Place Winner (Grand Prize)** @ Qualcomm Edge AI Developer Hackathon (2025)

**"Maximizing NPU efficiency for privacy-focused AI on the edge."**

- **Frontend / Desktop App:** [`jinsunghub/e.m.pilot`](https://github.com/jinsunghub/e.m.pilot)
- **Backend / API Server:** [`rkddlsxo/MailPilot_back`](https://github.com/rkddlsxo/MailPilot_back)
- **Tech Stack:** `Tauri`, `React`, `Vite`, `Python`, `Flask`, `MySQL`, `Qualcomm QNN SDK`, `Qualcomm AI Hub`
- **Key Achievements:**
  - **On-Device Optimization:** Quantized and optimized 4 AI models (Qwen2-7B, YOLOv8, EasyOCR, Nomic-Embed) for **Snapdragon X Elite NPU**, achieving a **~75% reduction** in inference latency.
  - **Smart Automation:** Implemented local AI features for auto-summarization, intent-based reply generation, attachment analysis, and task extraction.
  - **Frontend/Backend Integration:** Built the desktop frontend in my repository and connected it with the team backend API server for Gmail, AI model, and database workflows.

<br>

## Project 2. High-Performance Computing & Kernel Optimization
> **System Software & Parallel Computing Research (Individual Project)**

**"Breaking the Memory Wall: Optimization of Matrix Operations and AI Inference."**

- **Tech Stack:** `C++`, `CUDA`, `OpenMP`, `Linux`, `Google Multichase`
- **Key Achievements:**
  - **GPU Memory Optimization (CUDA):** Implemented **Shared Memory Tiling** for GEMM and utilized **Constant Memory** for Convolution kernels. Solved global memory bottlenecks, achieving **~1.4x speedup** compared to naive implementations. 
  - **CPU Parallelization (OpenMP):** Optimized **Softmax Regression** inference by analyzing scheduling policies. Applied **Dynamic Scheduling** to resolve load imbalance issues, resulting in **1.78x speedup**.
  - **System Profiling:** Conducted granular analysis of **L1/L2/LLC cache latencies** and **NUMA** remote memory access penalties using Google Multichase to identify hardware bottlenecks.
- **GitHub:** [jinsunghub/HPC-System-Optimization](https://github.com/jinsunghub/HPC-System-Optimization)

<br>

## Project 3. LANEIGE Ranking Insight Agent
> **Amorepacific AI Innovation Challenge 2026**

**"Autonomous Agentic Workflow for Real-time Beauty Insights."**

- **Tech Stack:** `Python`, `FastAPI`, `LangChain`, `RAG`, `ChromaDB`, `Amazon PA-API`
- **Key Achievements:**
  - **Agentic Architecture:** Designed a scalable **FastAPI** backend integrated with **LangChain**, featuring an autonomous decision-making loop with 9 custom tools.
  - **Hybrid Data Pipeline:** Implemented **RAG (Retrieval-Augmented Generation)** using ChromaDB (Vector) and SQLite (Relational) to minimize hallucinations and ensure accurate product data retrieval.
  - **Async Processing:** Developed an asynchronous data collector integrating **Amazon PA-API** to gather real-time global ranking insights without blocking user interactions.
- **GitHub:** [terrapin888/amore_ai_agent](https://github.com/terrapin888/amore_ai_agent)

<br>

## Project 4. 멋쟁이사자처럼 AI/NLP Bootcamp Practice
> **LIKELION AI/NLP Bootcamp (2026.06 - 2026.09)**

**"Building NLP fundamentals from tokenization to sequence-to-sequence modeling."**

- **Program:** 멋쟁이사자처럼 AI/NLP Bootcamp
- **Period:** `2026.06 - 2026.09`
- **Tech Stack:** `Python`, `PyTorch`, `SentencePiece`, `Hugging Face Tokenizers`, `Jupyter Notebook`
- **Key Achievements:**
  - **Tokenizer Experiments:** Trained and compared subword tokenizers, analyzed vocabulary size, token frequency, unknown-token behavior, and tokenizer speed.
  - **Sequence Modeling:** Implemented PyTorch Dataset/DataLoader pipelines, RNN-based sentiment classification, WMT14 Seq2Seq translation, and Bahdanau Attention.
  - **Evaluation & Documentation:** Compared model outputs using BLEU, token accuracy, precision/recall, and confusion matrix, then organized the practice notebooks and experiment notes for review.
- **GitHub:** [jinsunghub/nlp-bootcamp](https://github.com/jinsunghub/nlp-bootcamp)
