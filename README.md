# Standalone Scientific Peptide Extractor

A ready-to-run, standalone pipeline that extracts validated biological peptide and protein sequences from plant science PDFs. It combines heuristic text mining with a 3-pass LLM verification engine to deliver high-precision results with full evidence provenance.

## Pipeline Architecture

```
Input PDFs + Supplementary Files
        │
        ▼
  Step 1: PDF Parsing & Candidate Extraction (prepare_input.py)
        │
        ▼
  Step 2: PlantPepDB Sequence Scoring
        │
        ▼
  Step 3: SciBERT Biomedical Context Scoring
        │
        ▼
  Step 4: 3-Pass LLM Verification (Ollama)
        ├── Pass 1: Sequence Verification
        │     → Is this a real peptide? With evidence line & confidence score.
        ├── Pass 2: Technique Verification
        │     → Is each method genuinely used? With evidence line & per-technique score.
        └── Pass 3: Species Verification
              → What organism was studied? With evidence line & confidence score.
        │
        ▼
  Output: verified_peptides.csv (13 columns) + verified_peptides.json
```

## Prerequisites

1. **Tesseract OCR** — Required for parsing image-embedded PDFs.
   - Ubuntu/Linux: `sudo apt-get install tesseract-ocr`
   - macOS: `brew install tesseract`

2. **Python Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Local LLM (Ollama)** — Required for the verification step.
   - Install from [ollama.com](https://ollama.com)
   - Start the server: `ollama serve`
   - Pull the default model: `ollama pull llama3.1:8b`

## Usage

### Basic — PDFs only
```bash
python run_extractor.py --pdf_dir /path/to/pdfs --output_dir ./results
```

### With Supplementary Files (recommended)
If you have supplementary FASTA (`.fa`), Excel (`.xlsx`), or CSV (`.csv`) files alongside
your PDFs, pass them too so the pipeline can extract sequences from them:
```bash
python run_extractor.py \
    --pdf_dir /path/to/pdfs \
    --supp_dir /path/to/pdfs \
    --output_dir ./results
```
> **Tip**: If supplementary files are in the same folder as the PDFs, use the same path for both `--pdf_dir` and `--supp_dir`.

### All Arguments
| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--pdf_dir` | ✅ | — | Directory containing input PDFs |
| `--output_dir` | ✅ | — | Directory to save results |
| `--supp_dir` | ❌ | None | Directory with supplementary files (`.fa`, `.xlsx`, `.csv`, `.tsv`, `.txt`) |
| `--concurrency` | ❌ | 4 | Number of parallel LLM requests. Lower if RAM is limited. |

## Output

Results are saved to your `--output_dir`:

### `verified_peptides.csv` — Main Output (13 columns)

| Column | Type | Description |
|--------|------|-------------|
| `PMID` | string | PubMed ID of the source paper |
| `Sequence` | string | Verified amino acid sequence |
| `Sequence_Evidence_Line` | string | Exact sentence in the paper where the sequence appears |
| `Sequence_LLM_Confidence` | float 0–1 | LLM confidence score for the sequence |
| `Techniques` | JSON array | LLM-verified experimental technique names |
| `Technique_Scores` | JSON array | Per-technique LLM confidence scores |
| `Technique_Evidence_Lines` | JSON array | Exact sentences proving each technique was used |
| `Species` | string | LLM-verified plant species (e.g. `Arabidopsis thaliana`) |
| `Species_LLM_Confidence` | float 0–1 | LLM confidence score for the species |
| `Species_Evidence_Line` | string | Exact sentence in the paper mentioning the species |
| `Is_Synthesized` | bool | Whether the peptide is a synthetic/synthesized peptide |
| `Gold_Standard_Match` | bool / blank | Match against gold standard (blank if no gold standard provided) |
| `Comments` | string | LLM reasoning for accepting or rejecting the sequence |

### Other Output Files
- `verified_peptides.json` — Machine-readable JSON with full scoring details
- `llm_verification_log.jsonl` — Detailed trace of all LLM calls (all 3 passes) for auditing

## Notes

- The pipeline skips re-parsing if `parsed_candidates/` already exists in the output directory. Delete it to force a fresh parse.
- Each candidate requires up to 3 LLM calls (sequence + techniques + species). Use `--concurrency 2` on machines with limited RAM/VRAM.
- Temperature is fixed at `0` for deterministic, reproducible results.

## Evaluated Papers
The extractor was benchmarked against a gold standard dataset using the following plant science papers (identified by PubMed ID, provided as `PMID.pdf`):
- 11842184
- 12015123
- 15125775
- 25233276
- 25807486
- 28041928
