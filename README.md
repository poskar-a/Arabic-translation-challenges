# WMT26 Arabic-Asian Machine Translation Challenge

A comprehensive machine translation fine-tuning and evaluation framework for Arabic-to-Asian language pairs (English, Hindi, Urdu). This project fine-tunes three state-of-the-art multilingual translation models and evaluates them across multiple metrics.

## Project Overview

This challenge focuses on:
- **Language Pairs**: Arabic ↔ English, Arabic ↔ Hindi, Arabic ↔ Urdu (6 directions total)
- **Models**: NLLB-200 (3.3B), MADLAD-400 (10B), GemmaX2 (28.9B)
- **Approach**: 
  - Baseline zero-shot inference on all models
  - Fine-tune each model with LoRA/QLoRA on training sets
  - Comprehensive evaluation using BLEU, ChrF2++, TER, COMET-22, and COMET-Kiwi

## Quick Start

### Installation

1. **Clone the repository** (if applicable):
   ```bash
   cd wmt
   ```

2. **Set up Python environment** (recommended Python 3.10+):
   ```bash
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set environment variables** (optional):
   ```bash
   export WMT_ROOT=/path/to/wmt  # If using non-standard layout
   ```

### Configuration

All paths, models, and hyperparameters are centralized in [utlis/config.py](utlis/config.py):
- Dataset directory: `dataset/`
- Output directory: `outputs/`
- Log directory: `logs/`
- Checkpoint directory: `checkpoints/`

## Running the Pipeline

The main orchestrator is [run_pipeline.py](run_pipeline.py) with three pipeline stages:

### 1. Zero-Shot Baseline
Translate devtest with pre-trained models (no fine-tuning):
```bash
python run_pipeline.py --stage zero_shot
python run_pipeline.py --stage zero_shot --model nllb  # Single model
```

### 2. Fine-Tune Models
Fine-tune all models on train/dev splits using optimal training strategies:
```bash
python run_pipeline.py --stage finetune
python run_pipeline.py --stage finetune --model madlad --direction ar-hi  # Single direction
```

**Training Strategy by Model**:
- **NLLB**: Seq2SeqTrainer with standard fine-tuning
- **MADLAD**: Seq2SeqTrainer with standard fine-tuning
- **GemmaX2**: SFTTrainer with LoRA (4-bit QLoRA quantization for memory efficiency)

### 3. Evaluate Fine-Tuned Models
Load saved checkpoints and compute metrics on devtest:
```bash
python run_pipeline.py --stage eval
python run_pipeline.py --stage eval --model gemmax2 --split dev  # Evaluate on dev split
```

### 4. Run Full Pipeline
Execute all stages sequentially:
```bash
python run_pipeline.py --stage all
```

## Project Structure

```
wmt/
├── requirements.txt              # Python dependencies
├── run_pipeline.py              # Main orchestrator script
├── resume_finetune.py           # Resume interrupted fine-tuning
├── dataset/                     # Training/dev/test data
│   ├── Ar-En/                  # Arabic-English pairs
│   ├── Ar-Hi/                  # Arabic-Hindi pairs
│   └── Ar-Ur/                  # Arabic-Urdu pairs
├── checkpoints/                 # Pre-trained model checkpoints
│   ├── gemmax2/                # GemmaX2 base models
│   ├── madlad/                 # MADLAD base models
│   └── nllb/                   # NLLB base models
├── finetuned_checkpoints/       # Saved fine-tuned models
│   ├── gemmax2/
│   ├── madlad/
│   └── nllb/
├── outputs/                     # Translation outputs & metrics
│   ├── zeroshot_all_results.json
│   └── eval_finetuned_all_results_dev.json
├── logs/                        # Training logs
├── EDA/                         # Exploratory Data Analysis
│   ├── CORPUS_ANALYSIS.md      # Detailed corpus statistics
│   ├── eda_train.py            # Training corpus analysis
│   ├── cross_split_eda.py      # Train/dev/test split analysis
│   ├── evaluate_mt_noreference.py
│   ├── eda_train/              # Analysis outputs
│   └── eda_split/              # Split analysis outputs
├── pipelines/                   # Model-specific training/inference scripts
│   ├── train_nllb.py
│   ├── train_madlad.py
│   ├── train_gemmax2.py
│   ├── infer_nllb.py
│   ├── infer_madlad.py
│   ├── infer_gemmax2.py
│   ├── evaluate.py
│   └── run_eval_finetuned.py
├── utlis/                       # Utilities
│   ├── config.py               # Central configuration
│   ├── data_loader.py          # Dataset loading
│   ├── gpu_utils.py            # GPU device mapping
│   └── diagnose_translations.py
└── submission/                  # Challenge submission artifacts
    ├── GEMMAX2/
    ├── MADLAD/
    └── NLLB/
```

## Data Overview

### Dataset Statistics

| Pair | Sentence Pairs | Arabic Tokens | Target Tokens | Mean Arabic Length | Mean Target Length |
|------|---------------:|--------------:|---------------:|--------------------:|--------------------:|
| Ar-En | 21,000 | 558,133 | 670,845 | 26.58 | 31.95 |
| Ar-Hi | 20,100 | 539,522 | 770,493 | 26.84 | 38.33 |
| Ar-Ur | 20,300 | 543,873 | 840,793 | 26.79 | 41.42 |

### Data Splits

Each language pair includes:
- **train**: ~20K sentence pairs for fine-tuning
- **dev**: Development set for validation during training
- **devtest**: Test set for final evaluation

**Key Finding**: 65.66% of sentences are duplicates when aggregated across pairs, indicating the Arabic source is heavily reused while target languages differ.

### Data Quality Notes
- **Alignment**: Perfect line-count agreement between source and target
- **Empty Sentences**: None detected
- **Vocabulary**: Diverse target language vocabularies (especially for Hindi and Urdu)
-  **Cross-Split Leakage**: Some overlap exists between train/dev/test (documented in [EDA/eda_split/LEAKAGE_REPORT.md](EDA/eda_split/LEAKAGE_REPORT.md))

## Exploratory Data Analysis (EDA)

Comprehensive corpus analysis is available in the `EDA/` directory:

### Key EDA Artifacts

1. **[CORPUS_ANALYSIS.md](EDA/CORPUS_ANALYSIS.md)**: Detailed statistical breakdown
   - Corpus composition and alignment verification
   - Sentence-length distributions
   - Vocabulary size and diversity
   - Out-of-vocabulary (OOV) rates
   - Semantic similarity and distribution analysis

2. **eda_train/**: Training corpus analysis
   - `alignment_check.json`: Line-count verification
   - `summary.csv`: Per-file and per-language statistics
   - `per_file/`: File-level detailed metrics
   - `per_language/`: Language-level aggregated statistics

3. **eda_split/**: Train/dev/test split analysis
   - `TRAIN_DEV_ANALYSIS_REPORT.md`: Split representativeness
   - `LEAKAGE_REPORT.md`: Cross-split overlap analysis
   - `csv/`: Raw metric CSVs
   - `plots/`: Visualization plots

### Running EDA Scripts

Generate or update analysis:
```bash
# Training corpus analysis
python EDA/eda_train.py

# Train/dev/test split analysis
python EDA/cross_split_eda.py

# No-reference evaluation on outputs
python EDA/evaluate_mt_noreference.py
```

## Evaluation Metrics

Models are evaluated using 5 complementary metrics:

1. **BLEU** (SacreBLEU): Word n-gram precision
2. **ChrF2++**: Character-level F-score (better for morphologically rich languages)
3. **TER** (Translation Edit Rate): Human-like edit operations needed
4. **COMET-22**: Neural metric trained on human judgments
5. **COMET-Kiwi**: Reference-free metric (uses source + MT only)

Evaluation is performed in [pipelines/evaluate.py](pipelines/evaluate.py).

### Expected Outputs

Results are stored as JSON in `outputs/`:
- `zeroshot_all_results.json`: Baseline metrics for all models/directions
- `eval_finetuned_all_results_dev.json`: Fine-tuned model metrics

Example structure:
```json
{
  "ar-en": {
    "nllb": {
      "bleu": 28.5,
      "chrf2pp": 52.3,
      "ter": 58.2,
      "comet22": 0.65,
      "kiwi": 0.72
    }
  }
}
```

## Model-Specific Details

### NLLB-200-3.3B
- Multilingual encoder-decoder
- Covers 202 languages
- Fine-tuned using Seq2SeqTrainer
- LoRA not required (smaller model)

### MADLAD-400-10B
- Google's 10B multilingual MT model
- Covers 250 languages
- Fine-tuned using Seq2SeqTrainer
- High-quality baseline for many pairs

### GemmaX2-28.9B (Gemma2)
- Large causal LM adapted for translation
- Fine-tuned using SFTTrainer with LoRA
- Quantized to 4-bit (NF4) for memory efficiency
- Best performance but highest computational cost

See individual training scripts in [pipelines/](pipelines/) for model-specific configurations.

## Resuming Training

If fine-tuning is interrupted:

```bash
python resume_finetune.py              # Resume all models
python resume_finetune.py --model nllb # Resume only NLLB
python resume_finetune.py --dry_run    # Preview what would resume
```

This script checks `finetuned_checkpoints/` and continues training only for incomplete directions.

## Advanced Usage

### Single Direction Fine-Tune
For quick iteration on a specific pair:
```bash
python pipelines/train_nllb.py \
  --model_name_or_path facebook/nllb-200-3.3B \
  --direction ar-en \
  --output_dir finetuned_checkpoints/nllb/ar-en
```

### Inference Only
Use pre-trained models without fine-tuning:
```bash
python pipelines/infer_nllb.py \
  --model_name_or_path facebook/nllb-200-3.3B \
  --input_file dataset/Ar-En/devtest_ar_ar-en.txt \
  --output_file outputs/nllb_ar-en_baseline.txt
```


## Key Files Reference

| File | Purpose |
|------|---------|
| [run_pipeline.py](run_pipeline.py) | Master orchestrator for all stages |
| [utlis/config.py](utlis/config.py) | Central configuration (paths, models, params) |
| [utlis/data_loader.py](utlis/data_loader.py) | Dataset loading and preprocessing |
| [utlis/gpu_utils.py](utlis/gpu_utils.py) | GPU device management |
| [pipelines/train_*.py](pipelines/) | Model-specific training scripts |
| [pipelines/infer_*.py](pipelines/) | Model-specific inference scripts |
| [pipelines/evaluate.py](pipelines/evaluate.py) | Multi-metric evaluation framework |
| [EDA/CORPUS_ANALYSIS.md](EDA/CORPUS_ANALYSIS.md) | Detailed corpus statistics |

## License & Attribution

This project is part of the WMT26 Arabic-Asian Machine Translation Challenge.

## Support

For issues or questions:
1. Check [EDA/CORPUS_ANALYSIS.md](EDA/CORPUS_ANALYSIS.md) for data-related questions
2. Review model-specific training scripts in [pipelines/](pipelines/)
3. Check GPU and CUDA configuration in [utlis/gpu_utils.py](utlis/gpu_utils.py)

---
