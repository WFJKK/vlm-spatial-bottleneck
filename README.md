# Synthetic Dataset Generation for VLM Engineering Documentation QA

**Author:** Joshua Kames-King

**Built with:** Claude Code (Anthropic) was used extensively as a development tool throughout this project. The dataset is designed for fine-tuning Mistral's vision-language models.

---

A synthetic data generation pipeline that produces multimodal training data for vision-language models on engineering design compliance checking. The pipeline generates technical drawings of mechanical plates with holes, paired with specification documents and exhaustive question-answer sets with step-by-step reasoning chains.

This dataset aims to specifically train VLMs to 1) strengthen quantitative visual "understanding" and 2) perform multi-modal multi-hop reasoning as described in more detail in the next subsection.

## Motivation and General Idea

Current VLMs struggle with cross-modal reasoning on engineering documents — extracting rules from text specifications and applying them to visual diagrams. DesignQA showed that even GPT-4o and LLaVA perform poorly on compliance checking tasks.

It is a priori not obvious what the exact failure mode is: a lack of ability of VLMs to infer quantitative data from the images, or multi-modal reasoning itself, or even a combination of both. There are hints in the literature that VLMs struggle significantly with quantitative visual questions ("how far apart are these objects?"), see for example Liao et al. (Q-Spatial, arXiv:2409.09788). Hence, we consider a self-created synthetic dataset that targets both failure modes through two controllable meta-parameters:

1. **Annotation density**: Each drawing is generated at three levels: fully annotated (all dimensions labeled), partially annotated (some dimensions removed), and unannotated (no dimension labels). At full annotation the task reduces to OCR + logic, isolating the reasoning component. At no annotation the model must infer measurements from visual proportions alone, directly training the quantitative spatial skill that the literature identifies as deficient. SpatialVLM (Chen et al., CVPR 2024) showed that this gap is data-driven rather than architectural — training on synthetic spatial data significantly improved quantitative estimation. Hence we are hopeful that synthetic data will help here too.

2. **Rule complexity**: Specification documents range from simple single-threshold rules ("all holes shall have diameter 8.0 ± 0.3mm") to conditional rules that require multi-hop cross-modal reasoning ("for Class A joints where hole spacing < 20mm, minimum edge distance shall be ≥ 2.0× hole diameter"). This forces the model to perform image → text → image → compute chains of increasing depth, targeting the cross-modal reasoning gap identified by DesignQA.

By varying these two axes independently, the dataset serves both as training data (graduated difficulty acts as a curriculum) and as a diagnostic tool (performance across the grid reveals whether failures stem from spatial inference, reasoning, or their combination).

In addition, we include step-by-step reasoning chains as worked examples in the training data (but not as part of the evaluation).

Each example requires the model to:
1. **Parse rules** from a specification document (text)
2. **Extract measurements** from a technical drawing (image)
3. **Apply rules to measurements** and determine compliance (reasoning)

## Architecture

The pipeline has five stages:

```
sampler.py → spec_generator.py → question_generator.py → renderer.py → orchestrator.py
```

### 1. Parameter Sampler (`sampler.py`)

Generates plate configurations with controlled compliance states using **constructive sampling**:
- Decides the desired outcome first (which rules should be violated, by which holes)
- Places holes to achieve that exact compliance state
- Verifies geometric validity and single-rule violations

This avoids the reject-and-retry problem of random generation. Each violating hole fails exactly one intended rule, giving clean ground truth.

**Parameters:**
- Plate dimensions: 80–160mm × 50–100mm
- Holes: 3–8 per plate, diameters 6–16mm
- Four rule types: tolerance, edge distance, spacing, bolt population

### 2. Specification Generator (`spec_generator.py`)

Converts plate configurations into readable specification documents. The same underlying rules are presented with varying complexity:

- **Simple**: 2–3 rules, direct statements ("All holes shall have diameter 10.0 ± 0.5mm")
- **Multi-rule**: 4 rules including bolt population, single zone
- **Conditional**: 4 rules, two zones, table lookups, material-class mapping requiring multi-hop reasoning

Rule order is shuffled in the document while preserving IDs, forcing models to locate rules by ID rather than position.

For conditional complexity, the model must chain multiple lookups:
1. Read material from header → "Aluminum 6061-T6"
2. Look up material-to-class mapping → Class II
3. Find tolerance table, Class II row
4. Determine hole size category (small/large)
5. Extract tolerance and compute acceptable range

### 3. Question Generator (`question_generator.py`)

Produces exhaustive question-answer pairs with **annotation-aware reasoning chains**:

- **Per-component compliance**: Every hole × every rule checked individually
- **Full audit**: "List all violations" — tests systematic completeness
- **Measurement extraction**: Distance between holes or edge distances
- **Rule selection** (conditional only): Tests spec parsing ("What tolerance applies to Zone B?")
- **Counterfactual**: "What minimum edge distance would H3 need to comply?" — tests backward reasoning

The reasoning chains adapt to the annotation level:
- **Full**: Exact values from labels — "H1 diameter is 7.9mm."
- **Partial**: Mix of exact (for visible labels) and approximate — "From the scale bar, H2 diameter appears approximately 8mm."
- **Minimal**: All approximate — "From the scale bar, H3 appears approximately 10mm from the top edge."

This ensures the training data teaches reasoning that matches what the model can actually see in each image. The annotation visibility decisions are shared between the renderer and question generator so they always agree on what's shown.

### 4. Image Renderer (`renderer.py`)

Generates technical drawings with three annotation levels:

- **Full**: All diameters labeled, edge distances shown, spacing annotations
- **Partial**: Some annotations randomly hidden (ensuring hidden ones are relevant to violations)
- **Minimal**: Hole IDs only, scale bar, no dimension labels

The annotation level controls how hard it is to extract measurements from the image. The same questions apply regardless of annotation level — only the visual information changes.

### 5. Pipeline Orchestrator (`orchestrator.py`)

Generates balanced datasets with even distribution across:
- 3 rule complexities × 3 annotation levels × 4 violation counts = 36 combinations

Outputs:
- `dataset.jsonl` — one record per example with image path, spec text, all QA pairs, and full metadata
- `images/` — PNG technical drawings
- `stats.json` — dataset statistics

## Design Decisions

### Why constructive sampling?

Random hole placement with post-hoc compliance checking leads to either: (a) most examples being fully compliant (which is not useful), or (b) messy multi-rule violations that make ground truth ambiguous. Constructive sampling guarantees exact control over the compliance state.

### Why three annotation levels?

This creates a natural curriculum:
- **Full annotation** teaches the reasoning pattern: parse rule → extract value → compute → conclude
- **Minimal annotation** forces visual inference: the model already knows the reasoning, but must get numbers from geometry instead of labels, which might be one of the failure modes as described in the motivation section
- The same questions and answers apply regardless — only the information source changes

### Why exhaustive hole × rule questions?

Each example checks every hole against every rule. This teaches systematic compliance checking rather than cherry-picking. A model trained on incomplete checks would learn to be incomplete.

### Avoiding data contamination

The data teaches reasoning skills, not memorizable facts, by construction:
- Every example has unique plate dimensions, hole positions, and rule parameters
- Rule order is shuffled in specs
- Material-class mapping is consistent but tolerance values vary per example
- The model cannot memorize "Aluminum = ±0.5mm" because the tolerance for each class changes between examples

## Evaluation

The evaluation script (`evaluate.py`) scores model predictions against ground truth on five metrics. There are two directions of failure: reasoning and quantitative spatial inference ability, and models may have different behaviours with regard to these two dimensions.

### 1. Compliance Classification Accuracy
Binary Yes/No for each hole × rule pair. Broken down by:
- **Annotation level**: Does accuracy drop from full → partial → minimal? This gap measures quantitative spatial inference ability.
- **Rule complexity**: simple → multi_rule → conditional. Here we are probing multi-modal reasoning.
- **Rule type**: tolerance vs edge distance vs spacing vs bolt
- **Answer balance**: accuracy on Yes vs No answers (detects bias)

### 2. Full Audit F1
The model produces a violation list, ground truth is a list. Precision catches hallucinated violations, recall catches missed ones. F1 combines both.

### 3. Measurement Extraction Error
Absolute error in mm between predicted and true distances. Directly measures spatial information extraction from images.

### 4. Rule Understanding Accuracy
For rule_selection questions — can the model correctly parse the spec to find applicable parameters?

### 5. Counterfactual Accuracy
Can the model compute correct thresholds for compliance? Tests backward reasoning.

### Running Evaluation

```bash
# Generate predictions (one JSONL line per question)
# Format: {"example_id": "EX-0000", "question_index": 0, "prediction": "Yes"}

# Score predictions
python evaluate.py --predictions predictions.jsonl --ground_truth dataset/dataset.jsonl

# Test with mock models
python test_evaluate.py
```

## Usage

### Setup

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
```

### Sample Dataset

A pre-generated sample of 200 examples (~3,500 questions) is included in `dataset/`. This contains:

- `dataset.jsonl` — one JSON record per example, each with the image path, full specification text, all question-answer pairs with reasoning chains, and metadata (rule complexity, annotation level, violation count, hole positions, rule parameters)
- `images/` — 200 PNG technical drawings at varying annotation levels (full, partial, minimal)
- `stats.json` — summary statistics showing the distribution across complexities, annotation levels, violation counts, and question types

This sample is ready to use for fine-tuning or evaluation without running the pipeline. To regenerate or produce a larger dataset:

### Generate Dataset

```bash
# Default: 200 examples in ./dataset/
uv run python orchestrator.py

# Custom
uv run python orchestrator.py --num 500 --output ./my_dataset --seed 0
```

### Output Structure

```
dataset/
  dataset.jsonl      # All examples with QA pairs and metadata
  stats.json         # Dataset statistics
  images/
    EX-0000.png
    EX-0001.png
    ...
```

### JSONL Record Format

```json
{
  "example_id": "EX-0000",
  "image": "images/EX-0000.png",
  "spec_text": "SPEC-GP-672: Guide Plate GP-672 Design Requirements\n\nRule R1: ...",
  "questions": [
    {
      "type": "per_component_compliance",
      "question": "Does hole H1 comply with Rule R1?",
      "answer": "Yes",
      "reasoning": "H1 diameter is 8.1mm. Rule R1 specifies nominal 8.0 ± 0.3mm..."
    }
  ],
  "metadata": {
    "seed": 0,
    "rule_complexity": "simple",
    "annotation_level": "full",
    "num_violations": 1,
    "plate_width": 131.0,
    "plate_height": 63.0,
    "holes": [...],
    "rules": [...]
  }
}
```

## Known Limitations & Future Work

- **Spacing violations are underrepresented** (~5% of violations) due to geometric constraints in constructive placement. Improved from <1% by biasing placement toward plate interior, but still lower than other rule types.
- **Visual fidelity is synthetic** — matplotlib drawings, not real CAD output. Transfer to real engineering documents is an open question.
- **Linguistic diversity is limited** — template-based specs use the same sentence patterns. An LLM paraphrase step could add variety.

## Development

### Package Management

Uses `uv` with `pyproject.toml` for dependency management.

### Code Quality

- **Language:** Python 3.12
- **Type hints:** All functions have comprehensive type annotations
- **Type checking:** Passes `pyright` in basic mode with 0 errors, 0 warnings
- **Dependencies:** numpy, matplotlib

```bash
# Verify type checking
uv run pyright sampler.py spec_generator.py question_generator.py renderer.py orchestrator.py evaluate.py
```
