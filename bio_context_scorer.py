import json
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
import numpy as np
from typing import Dict, List

MODELS = {
    "PubMedBERT": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
    "BioBERT": "dmis-lab/biobert-base-cased-v1.2",
    "SciBERT": "allenai/scibert_scivocab_uncased",
    "BioLinkBERT": "michiyasunaga/BioLinkBERT-base"
}

class BioContextScorer:
    def __init__(self, model_name: str, gold_standard_path: str = None):
        if model_name not in MODELS:
            raise ValueError(f"Unknown model {model_name}")
            
        self.model_name = model_name
        self.hf_model_id = MODELS[model_name]
        self.gold_standard_path = Path(gold_standard_path) if gold_standard_path else None
        
        self.tokenizer = None
        self.model = None
        self.reference_embeddings = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.is_loaded = False
        
    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def load(self):
        if self.is_loaded:
            return
            
        print(f"Loading {self.model_name} for context embedding...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.hf_model_id)
        self.model = AutoModel.from_pretrained(self.hf_model_id).to(self.device)
        self.model.eval()
        
        ref_contexts = []
        if self.gold_standard_path and self.gold_standard_path.exists():
            print("Loading gold-standard validation contexts...")
            with self.gold_standard_path.open("r", encoding="utf-8") as f:
                gold_data = json.load(f)
                
            # Extract validation statements
            ref_contexts = [
                item.get("validation_statement", "") 
                for item in gold_data 
                if item.get("validation_statement")
            ]
            
            # If no validation statements, fallback to sequence source or some default
            if not ref_contexts:
                ref_contexts = [
                    item.get("sequence_source", "peptide synthesized and analyzed") 
                    for item in gold_data
                ]
        else:
            print("No gold standard provided. Using default biomedical reference contexts.")
            ref_contexts = [
                "The synthetic peptide was analyzed via mass spectrometry.",
                "Peptides were synthesized by solid-phase peptide synthesis and purified by HPLC.",
                "The translated product of the open reading frame encodes a short peptide.",
                "Amino acid sequence of the mature peptide was determined by Edman degradation.",
                "Two radiolabeled in vitro translation products of 12-aa and 24-aa residues were obtained.",
                "The identified sequence corresponds to an antimicrobial peptide.",
                "Signal peptide cleavage site was predicted based on the sequence.",
                "The purified fraction exhibited strong peptide activity."
            ]
            
        # Deduplicate to save compute
        ref_contexts = list(set(ref_contexts))
        print(f"Found {len(ref_contexts)} unique reference contexts.")
        
        self.reference_embeddings = self._encode(ref_contexts)
        self.is_loaded = True
        print(f"{self.model_name} initialized.")
        
    def _encode(self, texts: List[str]):
        # Batch size 16 for safety on memory
        all_embeddings = []
        batch_size = 16
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                encoded_input = self.tokenizer(batch, padding=True, truncation=True, max_length=256, return_tensors='pt').to(self.device)
                model_output = self.model(**encoded_input)
                embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])
                
                # Normalize embeddings
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                all_embeddings.append(embeddings.cpu())
                
        return torch.cat(all_embeddings, dim=0)

    def score_context(self, context: str) -> float:
        if not self.is_loaded:
            self.load()
            
        if not context:
            return 0.0
            
        cand_embed = self._encode([context])
        
        # Cosine similarity is just dot product for normalized embeddings
        similarities = torch.mm(cand_embed, self.reference_embeddings.transpose(0, 1))
        
        # We take the maximum similarity to any valid reference context
        max_sim = similarities.max().item()
        
        # Optional: scale negative cosine similarities to 0, though context should be >0
        return max(0.0, float(max_sim))

if __name__ == "__main__":
    scorer = BioContextScorer("BioBERT", "data/gold_standard.json")
    print(scorer.score_context("Two radiolabeled in vitro translation products of 12-aa (peptide A) and 24-aa (peptide B) residues were obtained that coeluted from two different HPLC columns with synthetic peptides deduced from ORF A and ORF B, respectively."))
