# Kim Jinsung

### Applied AI & Systems Engineer

I build production-oriented AI systems across **on-device inference, retrieval and verification pipelines, backend services, and performance optimization**. My work focuses on turning model capabilities into measurable, reproducible software—from NPU/GPU acceleration to official-data-grounded AI services.

## Engineering Focus

- **Applied AI systems:** RAG, agentic workflows, semantic retrieval, model serving, and evaluation
- **Systems performance:** CUDA, OpenMP, memory hierarchy analysis, quantization, and on-device inference
- **Reliable AI delivery:** backend APIs, structured metadata, blind/holdout evaluation, and conservative failure handling

## Selected Work

### E.M.Pilot — On-Device AI Email Client
**Grand Prize, Qualcomm Edge AI Developer Hackathon 2025**

Privacy-focused desktop email client that runs summarization, reply generation, OCR, attachment analysis, and task extraction on Snapdragon X Elite devices.

- Quantized and optimized Qwen2-7B, YOLOv8, EasyOCR, and Nomic Embed for the Qualcomm NPU
- Reduced inference latency by approximately **75%**
- Built the Tauri/React desktop client and integrated Gmail, model, and backend workflows
- **Stack:** Python, Tauri, React, Vite, Flask, MySQL, Qualcomm QNN SDK, Qualcomm AI Hub
- **Repository:** [jinsunghub/e.m.pilot](https://github.com/jinsunghub/e.m.pilot)

---

### AI News Fact Verification System
**Jun 2026 – Sep 2026 · LIKELION AI/NLP team project**

AI fact-verification service that extracts numeric claims from news articles, resolves them to official KOSIS table/ITEM/OBJ/period coordinates, and compares them with official values.

- Owned KOSIS Open API integration, metadata retrieval, coordinate resolution, verification pipeline design, and evaluation assets
- Separated semantic table retrieval from exact relational coordinate lookup to reduce unsupported automatic judgments
- Indexed **107,138 KOSIS tables** and built PostgreSQL/SQLite-backed metadata and API-cache workflows
- Evaluated development, table-disjoint blind, and locked URL service scenarios separately to avoid overstating generalization
- Final locked URL50 service QA completed **50/50 article jobs successfully** with a conservative evidence-selection policy
- **Stack:** Python, FastAPI, PostgreSQL, SQLite, BGE-M3, HCX, KOSIS Open API, Pandas
- **Repository:** [jinsunghub/AI-News-Fact-Verification-System](https://github.com/jinsunghub/AI-News-Fact-Verification-System)

---

### High-Performance Computing & Kernel Optimization
**Independent systems research**

Performance-focused implementations and profiling experiments covering GPU kernels, CPU parallelism, cache behavior, and NUMA effects.

- Applied CUDA shared-memory tiling to GEMM and constant memory to convolution kernels, achieving approximately **1.4× speedup**
- Optimized Softmax Regression inference with OpenMP dynamic scheduling, achieving **1.78× speedup**
- Profiled L1/L2/LLC latency and remote NUMA memory access with Google Multichase
- **Stack:** C++, CUDA, OpenMP, Linux, Google Multichase
- **Repository:** [jinsunghub/HPC-System-Optimization](https://github.com/jinsunghub/HPC-System-Optimization)

---

### LANEIGE Ranking Insight Agent
**Amorepacific AI Innovation Challenge 2026**

Agentic backend for collecting and analyzing global beauty-product ranking signals.

- Designed a FastAPI/LangChain workflow with nine domain tools
- Combined ChromaDB vector retrieval with relational lookup for grounded product analysis
- Implemented asynchronous Amazon PA-API collection for non-blocking data ingestion
- **Stack:** Python, FastAPI, LangChain, RAG, ChromaDB, SQLite, Amazon PA-API
- **Repository:** [terrapin888/amore_ai_agent](https://github.com/terrapin888/amore_ai_agent)

## Technical Toolkit

| Area | Technologies |
|---|---|
| AI / ML | Python, PyTorch, Transformers, BGE-M3, RAG, LangChain, SentencePiece |
| Backend / Data | FastAPI, Flask, PostgreSQL, SQLite, MySQL, REST APIs |
| Systems | C++, CUDA, OpenMP, Linux, Qualcomm QNN SDK |
| Client | Tauri, React, Vite |
| Engineering | Profiling, benchmarking, blind/holdout evaluation, reproducible experiment design |

## Foundations & Ongoing Development

Alongside project delivery, I maintain implementation notes and experiments covering tokenizer training, RNN-based classification, Seq2Seq translation, Bahdanau attention, BLEU, and classification evaluation.

- **Repository:** [jinsunghub/nlp-bootcamp](https://github.com/jinsunghub/nlp-bootcamp)
