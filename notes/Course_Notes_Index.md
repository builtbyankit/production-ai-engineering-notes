# 📚 Production AI / LLM Engineering — Course Notes

**Provider:** Krish Naik Academy
**Course:** Advanced Route — Production AI Engineering, Research to Production: LLMs, RAG & Agents, GenAIOps
**Available live recordings:** 19 July–4 October 2026 · 21 classes · 80 hours 6 minutes (dashboard total)
**Notes completed:** 21 of 21 classes · 932 live Q&A entries · 85 diagrams

[Course dashboard](https://learn.krishnaikacademy.com/web/courses/details/6a16f8935e281281cd6b1128) · [Instructor resource hub](https://krishnaikacademy.notion.site/Production-AI-Engineering-3a8eba9593d080d7b876e8ced7d541f1)

Each completed class file follows the class-notes-generator format: explanations in teaching order, supplied examples, relevant diagrams, live Q&A, key pointers, upcoming topics, and a practice checklist. Full transcripts and substantive closing question sessions are the primary sources. Class-matching instructor PDFs, notebooks, and repositories supply diagrams and verified code. Official documentation and research qualify technical claims where needed.

This set covers the live recordings currently posted through 4 October. The induction describes a broader future roadmap; it does not mean every planned RAG, agent, or operations module has already been taught. Session announcements and product examples retain their historical context. Code excerpts are identified by their supplied source; partial methods and illustrative snippets are described as such.

Read the notes in class order, then use each class’s key pointers and action checklist for revision. The live Q&A tables preserve substantive follow-ups and merge repeated questions.

## Onboarding and foundations

| Class | Session | Notes | Recording length |
|---|---|---|---|
| 01 | 19 Jul 2026 | [Induction and the Research-to-Production Roadmap](Class_01_Induction_and_Course_Roadmap_Notes.md) | 5hr 14min |
| 02 | 25 Jul 2026 | [Software Setup and the Engineering Workflow](Class_02_Software_Setup_Notes.md) | 3hr 41min 23s |
| 03 | 26 Jul 2026 | [PyTorch Fundamentals — From Data to a Trained Model](Class_03_PyTorch_Fundamentals_Notes.md) | 4hr 26min 56s |

## Transformers: intuition, architecture, and implementation

| Class | Session | Notes | Recording length |
|---|---|---|---|
| 04 | 1 Aug 2026 | [Transformers 101, Part 1 — Embeddings, Context, and Self-Attention Intuition](Class_04_Transformers_101_Part_1_Notes.md) | 4hr 11min 34s |
| 05 | 2 Aug 2026 | [Transformers 101, Part 2 — Queries, Keys, Values, and Contextualized Vectors](Class_05_Transformers_101_Part_2_Notes.md) | 3hr 45min 39s |
| 06 | 8 Aug 2026 | [Multi-Head Attention and the Transformer Encoder](Class_06_Multi_Head_Attention_Notes.md) | 4hr 1min 50s |
| 07 | 9 Aug 2026 | [Decoder Masking, Positional Encoding, and Cross-Attention](Class_07_Transformer_Architecture_Notes.md) | 4hr 19min 27s |
| 08 | 16 Aug 2026 | [Annotated Transformer Implementation — Designing the Encoder and Decoder](Class_08_Annotated_Transformer_Implementation_Notes.md) | 3hr 44min 8s |
| 09 | 22 Aug 2026 | [Assembling and Training a Transformer from Scratch](Class_09_Transformer_Training_Notes.md) | 3hr 21min 49s |
| 10 | 23 Aug 2026 | [Training and Evaluating the Transformer Translation Model](Class_10_Transformer_Training_and_Inference_Notes.md) | 3hr 34min 57s |

## Tokenization and the transition to prompting

| Class | Session | Notes | Recording length |
|---|---|---|---|
| 11 | 29 Aug 2026 | [Tokenization Foundations — BPE, Bytes, and Model Compatibility](Class_11_Tokenization_Foundations_Notes.md) | 5hr 17min 55s |
| 12 | 30 Aug 2026 | [Byte-Level BPE, WordPiece, and SentencePiece](Class_12_Subword_Tokenization_Notes.md) | 3hr 22min 51s |
| 13 | 5 Sep 2026 | [Custom Domain Tokenization and Vocabulary Trade-offs](Class_13_Custom_Domain_Tokenization_Notes.md) | 3hr 54min 16s |
| 14 | 6 Sep 2026 | [Offset Mapping, Custom BPE Decisions, and Prompting Foundations](Class_14_Offset_Mapping_and_Prompting_Notes.md) | 3hr 22min 54s |

## Prompt engineering and structured generation

| Class | Session | Notes | Recording length |
|---|---|---|---|
| 15 | 12 Sep 2026 | [Prompt Engineering — Reasoning, Structured Outputs, and Evaluation](Class_15_Prompt_Engineering_Notes.md) | 3hr 25min 9s |
| 16 | 13 Sep 2026 | [Structured Generation with Instructor and Outlines](Class_16_Structured_Generation_Notes.md) | 3hr 24min 41s |

## DSPy

| Class | Session | Notes | Recording length |
|---|---|---|---|
| 17 | 19 Sep 2026 | [DSPy Part 1 — Signatures, Modules, and Code Execution](Class_17_DSPy_Part_1_Notes.md) | 3hr 29min 53s |
| 18 | 20 Sep 2026 | [DSPy Part 2 — Bootstrap, MIPROv2, and Reflective Prompt Evolution](Class_18_DSPy_Part_2_Notes.md) | 4hr 11min 16s |

## Fine-tuning transformer architectures

| Class | Session | Notes | Recording length |
|---|---|---|---|
| 19 | 26 Sep 2026 | [Fine-Tuning Encoder, Decoder, and Encoder-Decoder Models](Class_19_Fine_Tuning_Transformer_Architectures_Notes.md) | 2hr 53min 19s |

## KV cache and attention variants

| Class | Session | Notes | Recording length |
|---|---|---|---|
| 20 | 3 Oct 2026 | [KV Cache — Inference, Memory Math, and Multi-Query Attention](Class_20_KV_Cache_Notes.md) | 3hr 25min 10s |
| 21 | 4 Oct 2026 | [Attention Variants — GQA, PagedAttention, and the MLA Introduction](Class_21_Attention_Variants_Notes.md) | 2hr 56min 56s |

## Preparation resources

The three prerequisite entries in the dashboard are external preparation resources. Their links are kept here alongside the 21 live-class notes.

- [Python prerequisites playlist](https://www.youtube.com/watch?v=bPrmA1SEN2k&list=PLZoTAELRMXVNUL99R4bDlVYsncUNvwUBB)
- [Complete Machine Learning NLP One Shot](https://www.youtube.com/watch?v=ENLEjGozrio)
- [Complete Deep Learning in 5 Hours](https://www.youtube.com/watch?v=d2kxUVwWWwU)

## Instructor code and notebook hubs

- [PyTorch primer](https://github.com/sourangshupal/pytorch-primer)
- [Transformers 101 resources](https://krishnaikacademy.notion.site/Transformers-101-3afeba9593d0804fa8e1e301f12ae2ff)
- [Annotated Transformer resources](https://krishnaikacademy.notion.site/Annotated-Transformers-3bfeba9593d080f881b6dec3fe17aec7)
- [Tokenization resources](https://krishnaikacademy.notion.site/Token101-3cceba9593d08062a195eecfcacb3bc9)
- [Custom token and offset-mapping resources](https://krishnaikacademy.notion.site/Custom-Token-3d2eba9593d080599b43c81ee42f91ce)
- [Medical tokenizer comparison pipeline](https://github.com/sourangshupal/tokener/tree/master)
- [Custom-vs-general tokenizer lab and vocabulary sweep](https://github.com/sourangshupal/tokenization-explainer/tree/vocab-size-sweep)
- [Prompt engineering notebooks](https://github.com/sourangshupal/prompt-engineering-notebooks)
- [Structured LLM and DSPy notebooks](https://github.com/sourangshupal/structured-llm-notebooks)
- [Fine-tuning resources](https://krishnaikacademy.notion.site/Finetuning-Transformers-3e7eba9593d08030aedaeedd07002e65)
- [KV cache and attention variants](https://github.com/sourangshupal/kv-cache-attention-variants)

Specific notebooks, original transcript filenames, PDF names, and verification references appear in the corresponding class notes. No class timestamps are included.

## Download and reading options

- Individual Markdown files: open a class link above.
- [Quick revision guide](Course_Quick_Revision.md): the key pointers from each completed class.
- [Single combined Markdown book](Production_AI_Engineering_Complete_Notes.md): all completed classes in sequence.
- [Notes package](Production_AI_Engineering_Notes.zip): the index, individual notes, and combined book.

Mermaid diagrams display in Markdown readers that support Mermaid. A reader without Mermaid support will show the diagram source; the adjacent explanations remain readable.
