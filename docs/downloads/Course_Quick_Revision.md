# 🔑 Production AI / LLM Engineering — Quick Revision

Key pointers extracted from 21 completed class notes. Open the linked class for the explanations, code, diagrams, Q&A, and practice checklist.

[Course index](Course_Notes_Index.md)

## Class 01: Induction and the Research-to-Production Roadmap

[Full class notes](Class_01_Induction_and_Course_Roadmap_Notes.md)

- Induction establishes the roadmap and expectations; it does not count as the implementation lesson for every term mentioned.
- The course assumes Python and basic deep learning and plans to build PyTorch familiarity through practice.
- Read functions/classes and the flow of data; do not rely only on copying a large codebase.
- Model foundations come before the broad retrieval/agent/production application work.
- A pretrained starting point and an adapted model are different from frontier-scale pretraining from scratch.
- SFT and PEFT can be combined; they describe different aspects of adaptation.
- Data preparation and validation are central to both model adaptation and synthetic-data projects.
- Project dataset sizes and 80GB-class GPU examples describe planned workloads, not universal lower bounds.
- Use a suitable existing or rented environment and choose spending according to the experiment.
- MCP and A2A have different integration roles.
- RAG and model adaptation have task-dependent benefits and failure modes; evaluate the actual system.
- Deployment is not the end of the work: latency, cost, permissions, observability, and quality matter afterward.
- Assignments permit AI assistance, while the learner remains responsible for understanding and correctness.
- Scope, exact stacks, pacing, and future research additions were flexible.
- Course completion, public posting, or a capstone does not guarantee a job, salary, clinical safety, or research outcome.
- Define the role or use case you want, then close the relevant gaps across data, models, software, and operations.

---

## Class 02: Software Setup and the Engineering Workflow

[Full class notes](Class_02_Software_Setup_Notes.md)

- The first objective is a working Python development workflow; every tool mentioned is not an immediate installation requirement.
- Follow the dependency file for the exercise instead of installing the future syllabus in advance.
- `uv venv` and `uv init` have different purposes: environment creation versus project initialization.
- `.venv` is a convention that helps keep local environments out of Git; custom names are acceptable with correct ignore rules.
- Paul's `activate` command was a personal alias. Activation syntax depends on the shell and platform.
- Preserve a familiar editor and terminal when they already do the job.
- A coding harness, a model, a provider, and a subscription are distinct layers.
- BYOK support and multimodal support must be checked for the actual product/model combination.
- Buy model access according to usage and task needs; promotional prices quoted in class are not permanent terms.
- System RAM, GPU memory, software support, and runtime are different constraints.
- Use cloud hardware when a task outgrows the laptop; save artifacts and end the resource when finished.
- Docker's client needs access to an engine/daemon to build and run containers.
- WSL 2 can support CUDA; evaluate the documented constraints rather than reject it categorically.
- Read papers for their problem, main contribution, evidence, and relevance to your data.
- Turn difficult equations into small experiments before scaling the implementation.
- Learn the underlying PyTorch computation so high-level framework failures are easier to investigate.
- Before fine-tuning for RAG quality, inspect the ingestion and retrieval pipeline and establish the actual cause.
- AI assistance makes experimentation faster, but you still need to understand and review the code.

---

## Class 03: PyTorch Fundamentals — From Data to a Trained Model

[Full class notes](Class_03_PyTorch_Fundamentals_Notes.md)

- Understand the full path: dataset → transform → batch → model → loss → backward → optimizer → evaluation.
- FashionMNIST uses 28 × 28 grayscale images, ten categories, 60,000 training examples, and 10,000 test examples.
- N, C, H, W describes the image batch; label count and unique class count are different.
- Batch size, learning rate, epoch count, and hidden widths are choices to test.
- The class’s fully connected model flattens each image to 784 features and returns ten logits.
- Default nn.Flatten preserves the batch dimension; unsqueeze adds an axis instead.
- Cross entropy accepts logits; integer class-index targets require long/int64.
- Backward computes gradients, the optimizer updates parameters, and zero_grad clears stored derivatives.
- Clear gradients for fresh optimizer updates; deliberate accumulation is a separate strategy.
- The training forward pass records the history needed by backward.
- eval controls layer behavior; no_grad controls autograd recording.
- Argmax returns an index, which is then mapped to a class.
- More epochs or a single correct prediction does not establish generalization.
- A saved state dictionary needs a compatible receiving architecture.
- Training recovery needs appropriate checkpoint state, including the optimizer when resuming its trajectory.
- Tensor dtype numbers describe bits, not decimal places.
- CPU NumPy/tensor memory sharing does not remove CPU-to-GPU transfer costs.
- Inspect shape, dtype, device, labels, and data quality before guessing at a training error.
- RAG and specialized models can be combined; benchmark the complete workload.
- Production throughput depends on the whole service chain, including downstream limits.
- Fine-tuning quality includes retained capabilities as well as newly learned behavior.

---

## Class 04: Transformers 101, Part 1 — Embeddings, Context, and Self-Attention Intuition

[Full class notes](Class_04_Transformers_101_Part_1_Notes.md)

- A token ID identifies an item; its arbitrary numerical value is not its meaning.
- A vector's width, the vocabulary size, the sequence length, and the training batch size are different quantities.
- A sliding window exposes neighboring positions; a useful dependency may still lie outside it.
- “Noa … great cat” demonstrates why a distant word can be essential context.
- Exponential smoothing introduces weighted influence; its proximity intuition is not a universal NLP rule.
- A 2D or 3D embedding plot is a projection of the original representation.
- Named attributes such as royalty or gender are teaching aids, not guaranteed labels for individual embedding coordinates.
- A pairwise dot product produces a scalar score.
- A normalized coefficient scales a source vector; the weighted sum produces an output vector.
- Each output row has its own coefficients and normalization.
- Attention coefficients calculated for an input are different from parameters learned during training.
- Self-attention is attention within the same sequence; its definition does not depend on the absence of learned weights.
- Bare attention has no position information; complete Transformers still need to represent order.
- Access to every allowed position does not assign every position equal importance.
- Y₁ through Yₙ are intermediate contextual representations, not final predictions.
- Processing an epoch does not create direct attention across separate chunks.
- Backpropagation is part of parameter learning; inference does not require updating weights for every input.
- The next lesson must add the missing architecture before training and model behavior can be understood fully.

---

## Class 05: Transformers 101, Part 2 — Queries, Keys, Values, and Contextualized Vectors

[Full class notes](Class_05_Transformers_101_Part_2_Notes.md)

- Attention creates one context-dependent output per token; it does not automatically pool a sentence.
- Vocabulary size, sequence length, and embedding width are independent.
- A vector inner product yields one scalar; QKᵀ collects all pairwise scalar scores.
- Attention weights A are input-dependent outputs, not the persistent projection parameters.
- Q/K/V originate from the same sequence in self-attention but have different roles.
- The query requests information, keys support matching, and values supply content.
- The projection matrices are shared across token positions within a block.
- Values are projected before aggregation; the final operation is AV.
- Square projections preserve width in the demo; other compatible widths are possible.
- Projection weights are parameters; dimensions and head count are hyperparameters.
- scale=0.5 in the random generator is a standard deviation.
- √d_k scales compatibility scores; softmax makes each attention row sum to one.
- Normalize across the key dimension for each query.
- AX is the simplified raw-vector mixture; AV uses projected values.
- allclose checks tolerance-based agreement.
- A shared static bank lookup naturally has cosine one across the two sentences.
- A changed contextual vector proves context dependence, not automatic semantic quality.
- Newly created nn.Embedding and nn.Linear modules are untrained.
- Compare the numerical outputs with the claim being made; some notebook narrative labels overstate the recorded evidence.
- Batch size and head count are different concepts.
- Built-in MultiheadAttention has its own projections, including an output projection.
- Batched attention needs the correct key-axis transpose.
- Multi-head attention and the full Transformer architecture were deferred for fuller treatment.
- Replacing embeddings in the assignment requires updating incompatible hard-coded widths.

---

## Class 06: Multi-Head Attention and the Transformer Encoder

[Full class notes](Class_06_Multi_Head_Attention_Notes.md)

- Multi-head attention provides several learned views of the same sequence.
- A head includes Q/K/V projections and attention; it is not simply one linear layer.
- S is a raw score table; A is a normalized attention table; each pairwise score is scalar.
- QKᵀ compares query rows with key rows, using matching feature widths.
- Softmax normalizes a query row across the allowed key positions.
- Scaling controls logit magnitude before softmax; saturated probabilities can have small derivatives.
- dₖ refers to the per-head query/key width, not the token count.
- In the base paper: d_model = 512, h = 8, and dₖ = dᵥ = 64.
- Per-head representations come from learned projections of the full input.
- Concatenation joins features; the output projection mixes them and restores the required width.
- More heads at fixed model width mean narrower heads, not automatically h times as many projection parameters.
- The FFN is Linear → ReLU → Linear, with base widths 512 → 2048 → 512.
- Encoder attention mixes positions; the FFN operates separately at each position.
- Each encoder layer has two residual additions and two layer-normalization steps.
- Residual paths add a sublayer's input to that sublayer's output.
- N× describes serial encoder depth; h describes parallel heads within attention.
- Six layers with eight heads do not constitute a single 48-head layer.
- Visualization controls reveal stored attention behavior; they do not modify model training.
- Complete decoder mechanics, positional formulas, and layer-normalization internals were not finished today.

---

## Class 07: Decoder Masking, Positional Encoding, and Cross-Attention

[Full class notes](Class_07_Transformer_Architecture_Notes.md)

- Identify whether a diagram is encoder-only, decoder-only, or encoder–decoder before comparing its numbers.
- Token IDs, hidden width, MLP expansion, vocabulary size, head count, and layer count are different quantities.
- Causal masking hides later **target positions**; padding masks can appear in other attention layers too.
- Zero logits are not zero probabilities. Mask forbidden scores before softmax with a suitable negative sentinel.
- Boolean mask meaning depends on the API and the implementation using it.
- Training positions can be computed in parallel while generation remains dependent on the growing prefix.
- Shifted target input and causal masking both prevent target leakage.
- `pos` is token position; `2i`/`2i+1` are paired feature coordinates.
- Each token gets the whole positional vector, containing both sine and cosine.
- At width four, pair exponents are0 and1/2; `PE(0)=[0,1,0,1]`.
- The position-one last coordinate is approximately0.99995, not0.0995.
- Fixed sinusoidal vectors are generated from a formula; learned positions and newer position methods are alternatives.
- Cross-attention projects decoder Q and encoder-memory K/V; it does not add the two stacks' internal K/V matrices.
- Source and target lengths can differ; the attention matrix is target-length by source-length.
- Encoder–decoder inference still uses encoder memory and cross-attention.
- KV caching, freezing parameters, and removing a module are different operations.
- The supplied notebook shares projection module references and has undefined final helpers; its complete standalone execution is not established.
- Choose configurations through task-specific experiments and evaluate more than training-loss stability alone.

---

## Class 08: Annotated Transformer Implementation — Designing the Encoder and Decoder

[Full class notes](Class_08_Annotated_Transformer_Implementation_Notes.md)

- Encoder self-attention, decoder self-attention, and decoder cross-attention have different input sources.
- Memory is the encoder output; it is not a mask or simply the last encoder K/V pair.
- The source mask is needed by encoder attention and decoder cross-attention.
- Target masking combines padding exclusion and causal constraints.
- Masking blocks contributions; it does not necessarily remove tensor rows.
- Pad IDs are configurable and are not always zero.
- Decoder input is a prefix; the unknown complete target is not required at inference.
- EncoderDecoder.forward returns decoder features; the generator is called separately.
- This generator returns log probabilities.
- Deep copies provide independent parameter storage but initially copy values.
- Layer norm uses feature statistics; it is not softmax.
- The custom LayerNorm differs numerically from built-in nn.LayerNorm.
- Pre-norm and post-norm differ in placement, not in addition order.
- Dropout probability and training/evaluation mode control its effect.
- The notebook’s Boolean mask uses True for allowed positions.
- The causal diagonal is allowed for the shifted-input next-token task.
- Keys transpose along the last two axes, retaining batch/head axes.
- Four MHA linears are Q/K/V/output projections, not four heads.
- Projection → view → transpose produces B × h × L × d_k.
- Masks broadcast over heads using token axes, not embedding-feature partitions.
- Merge heads with transpose → contiguous → view, then output projection.
- The position-wise FFN returns to d_model for residual shape compatibility.
- Token embeddings multiply by √d_model; attention scores divide by √d_k.
- Sine/cosine alternate across feature coordinates, not across token positions.
- Positional buffers are non-parameter model state.
- Full assembly and training were deferred; saved notebook results do not show work completed live today.

---

## Class 09: Assembling and Training a Transformer from Scratch

[Full class notes](Class_09_Transformer_Training_Notes.md)

- A factory connects existing modules; c(attn) means deep copy here.
- N is stack depth; h is head count; vocabulary size is another quantity.
- The companion's residual helper normalizes before its sublayer.
- The generator returns log-probabilities over target vocabulary entries.
- Untrained inference checks wiring, not task quality.
- The start ID is specific to the vocabulary and demonstration.
- Memory is encoder output; the decoder extends its prefix.
- Last decoder position does not mean last vocabulary word.
- Padding comparisons create Booleans; unsqueeze adds an axis.
- Target input and label slices define a next-token offset.
- Those slices do not shift the encoder's source.
- AND combines padding and causal restrictions.
- Non-padding labels supply the meaningful loss denominator.
- Backward calculates gradients; the optimizer changes parameters.
- The shown zero-based accumulation condition triggers on the first batch.
- Accumulation scaling and schedule step units require inspection.
- Clearing gradients to None is distinct from setting zero tensors.
- Warmup rises from a low rate before inverse-square-root decay.
- LambdaLR's factor multiplies the optimizer's base rate.
- Detailed regularization and dataset training were deferred.

---

## Class 10: Training and Evaluating the Transformer Translation Model

[Full class notes](Class_10_Transformer_Training_and_Inference_Notes.md)

- This practical trains an educational encoder–decoder from scratch; later foundation-model adaptation is a different task.
- Perplexity uses reference-token probabilities; BLEU uses reference n-gram matching with length effects.
- Overlapping words do not guarantee that the translated meaning is correct.
- Label smoothing changes the ground-truth target distribution, not the model's argmax output after generation.
- KLDivLoss expects log-space model input in this implementation.
- `reduction="sum"` sums loss values; it does not make the loss equal one.
- Padding labels are excluded from training loss; a wrong padding prediction for a real word is not automatically ignored.
- EOS and padding have different meanings.
- The copy generator has an exclusive randint upper bound and yields one batch at a time.
- German is the source and English is the target for the supplied Multi 30k run.
- Preserve vocabulary mappings together with compatible model weights.
- Default helper arguments can differ from the config actually passed.
- Warm-up begins small, rises, and then decays; a base rate is not the observed rate at every step.
- Check accumulation boundaries/scaling before claiming equivalence to a larger batch.
- `.eval()`, no-grad, log flushing, and releasing unused CUDA cache are separate operations.
- Epoch indices 0–7 mean eight epochs, not seven.
- A saved final checkpoint may bypass retraining when comparing changed epoch counts.
- Greedy decoding chooses an argmax; its displayed `prob` variable contains log-probabilities.
- The source's qualitative translations are imperfect and do not establish a measured benchmark score.

---

## Class 11: Tokenization Foundations — BPE, Bytes, and Model Compatibility

[Full class notes](Class_11_Tokenization_Foundations_Notes.md)

- Tokenization, token IDs, embeddings, and semantic knowledge are different.
- Vocabulary size differs from embedding width and sequence length.
- BPE fitting merges adjacent symbols, including previously merged pieces.
- Count occurrences weighted by corpus word frequency.
- Recompute pair statistics after each merge.
- This notebook’s tied choices are deterministic, not random.
- The merge history is ordered because later rules depend on earlier ones.
- The end-of-word marker belongs to the toy implementation, not every BPE tokenizer.
- A vocabulary/merge budget is a real design choice.
- Statistical tokenizer fitting is training, but not neural-model fine-tuning.
- New words can reuse existing pieces without changing tokenizer rules.
- A missing whole-word entry does not necessarily produce UNK.
- Full byte coverage can encode valid text without unknown-word fallback.
- Unicode code points, UTF-8 bytes, and learned token IDs are not interchangeable.
- Decoding one token can show � when its bytes do not form a complete character.
- Bigger vocabularies can change token counts, but do not guarantee every string becomes shorter.
- WordPiece remains distinct from BPE; SentencePiece is a library supporting multiple models.
- BLT is a language-model architecture using adaptive byte patches.
- Same BPE algorithm or vocabulary size does not establish model compatibility.
- Keep token IDs, special-token meaning, embeddings, and output dimensions aligned.
- Adding entries requires matching model adaptation; editing JSON alone is insufficient.
- Domain knowledge can be supplied through context/retrieval without changing a tokenizer.
- Plain-text token counts may omit request-format/tool/multimodal overhead.
- Retrieval, reranking, generation, and software correctness require different evaluation.
- Vector quantization and LLM quantization are different operations.
- Multiple model judges can help review; their agreement is not automatically truth.

---

## Class 12: Byte-Level BPE, WordPiece, and SentencePiece

[Full class notes](Class_12_Subword_Tokenization_Notes.md)

- Unicode code points, UTF-8 bytes, token IDs, and vocabulary entries are different units.
- Café has four code points and five UTF-8 bytes.
- The byte alphabet has 256 values; the learned BPE vocabulary can be much larger.
- Byte atoms change the starting representation, not the core pair-frequency merge loop.
- Unknown learned characters are different from invalid or unsupported text representation.
- Byte coverage preserves information without guaranteeing language-model understanding.
- Individual token byte sequences may split a UTF-8 character.
- Sequence length and vocabulary size create different model costs.
- Tied training pairs can produce different merge orders.
- WordPiece-style scoring compares pair frequency with component frequencies.
- Training-time pair selection and runtime segmentation are separate procedures.
- Statistical subwords need not match human linguistic boundaries.
- SentencePiece is a library with several algorithms, and supports optional byte fallback.
- A .model file is needed to use the learned tokenizer; .vocab aids inspection.
- Negative piece scores are not negative token IDs.
- Whitespace marker ▁ is not an ordinary underscore.
- Normalization can affect exact original-text recovery.
- Larger vocabularies often shorten familiar inputs, but increase other costs.
- Sampling changes valid segmentation, not a fixed vocabulary's ID meanings.
- A pretrained model and tokenizer must retain compatible token associations.
- Compression efficiency is one evaluation dimension, not the whole objective.
- A textual hex dump is different from internal byte-level tokenization.

---

## Class 13: Custom Domain Tokenization and Vocabulary Trade-offs

[Full class notes](Class_13_Custom_Domain_Tokenization_Notes.md)

- Domain matching changes segmentation efficiency; it does not automatically establish model or clinical quality.
- Tokenizer vocabulary learning, embedding learning, domain pretraining, and task fine-tuning are different operations.
- Use actual tokenizer files and IDs rather than assuming a model’s domain label implies a custom vocabulary.
- Report fertility with its exact word-count convention and token-per-character metrics with their units.
- Curated probes explain mechanisms; held-out corpora test whether the pattern persists.
- Hold algorithm and vocabulary budget constant when testing the training-domain effect.
- A single-token rate is defined over selected strings and boundaries, not over model correctness.
- Requested vocabulary size can exceed the number of merges the data supports.
- A vocabulary increase adds embedding rows; it does not automatically increase hidden width.
- The compression knee, corpus scale, and embedding cost can favor different choices.
- The roughly 6.3 million count is a corpus-scale proxy, not the number of unique vocabulary items.
- Preserve tokenizer-to-embedding ID compatibility with any pretrained model.
- An enormous `model_max_length` sentinel does not establish a real long-context capability.
- The final selection and publishing walkthrough were deferred, and the no-key test-data framework question remained unresolved.

---

## Class 14: Offset Mapping, Custom BPE Decisions, and Prompting Foundations

[Full class notes](Class_14_Offset_Mapping_and_Prompting_Notes.md)

- A tokenizer compression improvement is not a language-model accuracy result.
- The knee rule uses unrounded held-out averages; rounded boundary values can mislead.
- Corpus words, characters, document counts, token pieces, and unique vocabulary entries are different units.
- The 32K decision was a compromise for the lab, not a universal domain-model default.
- A byte alphabet supplies representability; learned merges supply compression.
- `Ġ` represents a space byte. Offset trimming changes spans, not token identity.
- The complete 256-byte alphabet needs a sufficient vocabulary budget.
- Wrapping a tokenizer changes its loading interface, not compatibility with arbitrary model weights.
- `--out` is the verified trainer/wrapper output flag.
- Prompt anatomy makes requirements visible; examples show the desired input/output pattern.
- Persona and system-message tests guide behavior, but one successful test does not establish security or determinism.
- RAG source versions must map to indexed chunk metadata if outdated content is to be removed reliably.

---

## Class 15: Prompt Engineering — Reasoning, Structured Outputs, and Evaluation

[Full class notes](Class_15_Prompt_Engineering_Notes.md)

- A formatted answer, a valid schema, and a correct answer are three different outcomes.
- Worked explanations make assumptions and contradictions visible; they do not guarantee correctness.
- A low-temperature run can still produce a wrong answer or vary across hosted requests.
- State the clock convention in the train example; output-only formatting can hide ambiguity.
- JSON mode does not guarantee the exact keys and types requested in prose.
- Pydantic descriptions communicate meaning; enforceable constraints must be represented explicitly.
- The OpenAI parsing path and a JSON-mode-plus-validation fallback have different generation guarantees.
- A regex subgroup is an exact match operation, not semantic understanding.
- Measure the full output even when the displayed table truncates it.
- A substring grader can pass an incorrect label or a negated statement.
- A sequential chain cannot execute dependent stages simultaneously.
- Map-reduce and reviewer decomposition can be conceptually parallel while their teaching code is synchronous.
- A routed “agent” producing text does not automatically perform external actions.
- Generated templates must match the application's placeholder convention.
- Self-refinement changes prompts and outputs at inference time; it does not train weights.
- Majority agreement is a vote share, not a calibrated truth probability.
- The discount calculation is $126 regardless of how many samples vote otherwise.
- Model-generated code, local Python execution, and model-accessible execution tools are separate layers.

---

## Class 16: Structured Generation with Instructor and Outlines

[Full class notes](Class_16_Structured_Generation_Notes.md)

- Valid JSON, schema compliance, and correct source extraction are separate checks.
- Instructor validates parsed outputs and can re-ask with error feedback; attempts can still be exhausted.
- Outlines can constrain generation through a compatible backend; supported output types differ by backend.
- A schema does not eliminate all task instructions or automatically reduce total cost.
- Custom Python validator logic runs locally; relevant error feedback can guide a subsequent request.
- `ge`/`le` include the boundary, while `gt`/`lt` exclude it.
- Pydantic V2 nullable fields remain required unless a default permits omission.
- Before/after validation modes describe Pydantic processing stages.
- Partial snapshots are provisional; a changing number should not be committed as final data.
- Time to a visible parsed field is different from first-token latency and total completion time.
- Separate streaming and non-streaming requests have no guaranteed latency ordering.
- The local sentiment benchmark was interrupted and the redaction example failed.
- A fixed Choice list prevents out-of-list values, not incorrect in-list decisions.
- Range constraints can produce source-inconsistent “corrections.”
- The batch result was about 0.5 invoices per second, using a sequential loop.
- CFG, vision, DSPy, and KV-cache implementation were future topics.

---

## Class 17: DSPy Part 1 — Signatures, Modules, and Code Execution

[Full class notes](Class_17_DSPy_Part_1_Notes.md)

- A grammar restricts allowed syntax; it does not verify database identifiers, formulas, or facts.
- Decode-time masking prevents disallowed continuations; it is not ordinary post-generation retry.
- A grammar used by the decoder and source data used in a prompt have different size constraints.
- Hosted-API constraint support varies; the local notebook's limitations are not universal.
- A structured invoice can contain wrong readings and inconsistent totals.
- Inspect the actual loaded image before attributing a fallback result to model quality.
- The shown image batch is sequential.
- Signatures define tasks; modules define execution; optimizers seek improvement against metrics.
- DSPy still builds model messages internally.
- Predict is the simplest call; ChainOfThought adds generated reasoning; code-execution modules add a runtime.
- The default interpreter is Deno plus Pyodide/WASM, separate from the notebook kernel.
- Code execution and repair are bounded and do not guarantee task correctness.
- Current ProgramOfThought and CodeAct documentation marks them deprecated.
- RLM is a programmatic context-access strategy, not necessarily a new neural model architecture.
- Typed confidence is not calibrated confidence; production migrations need representative evaluations.

---

## Class 18: DSPy Part 2 — Bootstrap, MIPROv2, and Reflective Prompt Evolution

[Full class notes](Class_18_DSPy_Part_2_Notes.md)

- A signature defines the task; an optimizer improves a specified program against a specified objective.
- Compilation can make many calls even when the invocation is one line.
- These prompt optimizers do not train the underlying LM's weights.
- Mark example inputs with `with_inputs`; keep labels available to the evaluator.
- The class QA generator is seeded template sampling, not LLM generation.
- Forty repeated rows do not equal forty distinct problems.
- Word overlap can admit factually wrong demonstrations.
- The live 0.5 threshold and current notebook's 0.3 threshold should not be silently conflated.
- Test the metric itself before treating its score as meaningful.
- Bootstrap preserves instructions and adds demonstrations.
- Four bootstrapped demos does not necessarily mean four total demos.
- Inspect the inner predictor state and the actual rendered request.
- Bootstrap can stop early when its target is reached.
- The class improvement was 25 percentage points on twelve development rows.
- MIPROv2 changed the instruction but did not beat bootstrap in the live run.
- A Bayesian surrogate models the search objective; it is not a cloned LLM.
- Validation data used to select candidates is not an untouched final test.
- Prompt optimization can overfit even with frozen neural weights.
- A 100% score warrants inspection, not an automatic declaration that the checker is wrong.
- GEPA learns from trace-grounded textual feedback while retaining complementary candidates during search.
- The task and reflection roles can use the same or different models.
- A Pareto pool does not automatically become a runtime request router.
- Save compiled state and re-optimize deliberately.

---

## Class 19: Fine-Tuning Encoder, Decoder, and Encoder-Decoder Models

[Full class notes](Class_19_Fine_Tuning_Transformer_Architectures_Notes.md)

- These are full fine-tuning exercises, with all listed parameters trainable.
- The two classifiers use different tasks, so their scores do not compare architectures fairly.
- Model head, tokenizer, label mappings, and input contract must match the task.
- Class label IDs are different from vocabulary IDs and character offsets.
- DistilBERT does not require original BERT’s token-type inputs.
- Encoders can mask padding without using a causal mask.
- The dictionary variables `k` and `v` are unrelated to attention KV.
- Best-checkpoint selection follows weighted F1 in the classifier notebooks.
- Macro F1, weighted F1, precision, and recall answer different questions.
- Row-normalized confusion-matrix diagonals are class recall.
- Batch size, precision, and accumulation need measurement; they are not automatic quality upgrades.
- GPT-2’s configured pad ID lets its classifier find the last non-padding token.
- The AG News validation/test subsets both originate from the official test pool and are not individually exactly balanced.
- The custom inference outputs include incorrect predictions even with confident scores.
- The FLAN-T5 lesson was assigned companion study after a brief live introduction.
- Summarization needs source/target tokenization, ignored target padding, and generated-text evaluation.
- ROUGE is overlap, not a factuality guarantee.
- Saving a model is different from publishing it, and publishing is different from serving it.
- `Trainer.push_to_hub` takes a commit message before repository-selection configuration.
- Retention, annotation quality, hardware fit, and scaling remained design questions requiring application-specific evidence.

---

## Class 20: KV Cache — Inference, Memory Math, and Multi-Query Attention

[Full class notes](Class_20_KV_Cache_Notes.md)

- Cache old K/V because a fixed causal prefix's representations do not change when later tokens arrive.
- Cached attention still attends over history; all decoding is not constant-cost.
- Prefill handles known prompt positions; ordinary output generation has successive-token dependencies.
- TTFT, TPOT, throughput, and complete generation time are different measurements.
- KV memory is a product of architecture, precision, cached lengths, and concurrency.
- Cache payload grows linearly with sequence length, holding the other factors fixed.
- Use exact bytes and distinguish MB/GB from MiB/GiB.
- MQA preserves multiple query heads and shares one KV head per layer/position.
- Total Q width and per-query-head width are different.
- K and V are separate projections, even when shared across query heads.
- No weight updates occur during ordinary inference; shared training gradients are a separate mechanism.
- MQA requires compatible learned architecture; `use_cache` does not convert MHA.
- Smaller cache reduces traffic/capacity pressure, not the hardware's bandwidth ceiling.
- Prompt-prefix reuse, active KV state, and conversation/response storage have different lifecycles.
- A measured cache speedup for one model/run is not a universal benchmark or an MQA quality result.

---

## Class 21: Attention Variants — GQA, PagedAttention, and the MLA Introduction

[Full class notes](Class_21_Attention_Variants_Notes.md)

- GQA changes KV sharing while preserving the query-head role in the illustrated architecture.
- Distinct query heads can produce distinct attention patterns over shared keys.
- Sharing ratio is query-head count divided by KV-head count.
- The ratio of four was a design starting point, not a universal law.
- Memory retained and memory reduced are complementary percentages.
- Eight-head MQA retains 12.5% of matched MHA cache; thirty-two-head MQA retains 3.125%.
- Cache reduction is not a measured percentage of information or quality loss.
- Persistent KV cache and temporary query memory are different allocations.
- A model fitting by itself does not establish serving capacity under concurrent long requests.
- Total context length includes the prompt and continuation.
- Paging preserves logical order while allowing noncontiguous physical blocks.
- Internal tail waste is bounded by fewer than one block per sequence.
- Insufficient total capacity is different from external fragmentation.
- Prefix reuse requires compatible exact prefixes and relevant model identity.
- Hash lookup finds reusable blocks; block tables map logical locations.
- Reference counters protect active users of a shared block.
- Counter zero can make storage eligible for reuse without requiring immediate erasure.
- Prefix caching reuses prompt states, not a cached final response.
- MLA uses a learned low-rank projection directly from the attention input.
- Latent expansion is not universal lossless reconstruction of arbitrary original vectors.
- Count MLA's actual latent and positional tensors rather than blindly multiplying a joint width by two.
- Paging and latent/head-sharing mechanisms operate on different axes.
- RoPE's full derivation and the complete MLA integration remained next-class work.

---
