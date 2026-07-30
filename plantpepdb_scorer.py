import re
import numpy as np
from pathlib import Path
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from typing import Dict, Any

def normalize_sequence(seq: str) -> str:
    return re.sub(r'[^A-Z]', '', seq.upper())

class PlantPepDBScorer:
    def __init__(self, fasta_path: str):
        self.fasta_path = Path(fasta_path)
        self.sequences = set()
        self.global_aa_counts = Counter()
        self.lengths = []
        self.median_length = 0
        self.vectorizer = None
        self.nn_model = None
        
        self.is_loaded = False
        
    def load(self):
        if self.is_loaded:
            return
            
        print("Loading PlantPepDB database...")
        with self.fasta_path.open("r") as f:
            current_seq = []
            for line in f:
                if line.startswith(">"):
                    if current_seq:
                        s = "".join(current_seq)
                        self.sequences.add(s)
                        self.global_aa_counts.update(s)
                        self.lengths.append(len(s))
                    current_seq = []
                else:
                    current_seq.append(line.strip().upper())
            if current_seq:
                s = "".join(current_seq)
                self.sequences.add(s)
                self.global_aa_counts.update(s)
                self.lengths.append(len(s))
                
        self.median_length = np.median(self.lengths) if self.lengths else 0
        total_aa = sum(self.global_aa_counts.values())
        self.global_aa_freq = {k: v / total_aa for k, v in self.global_aa_counts.items()} if total_aa else {}
        
        print("Building k-mer similarity index (this might take a minute)...")
        # To avoid memory overflow, we can use a sample if sequences > 500k, but let's try full
        seq_list = list(self.sequences)
        if len(seq_list) > 500000:
            print(f"Subsampling {len(seq_list)} sequences to 500,000 for kNN index to save memory/time.")
            import random
            random.seed(42)
            seq_list = random.sample(seq_list, 500000)
            
        self.vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(3, 3))
        X = self.vectorizer.fit_transform(seq_list)
        
        self.nn_model = NearestNeighbors(n_neighbors=1, metric='cosine', n_jobs=-1)
        self.nn_model.fit(X)
        self.is_loaded = True
        print("PlantPepDBScorer fully initialized.")

    def score_candidate(self, sequence: str) -> Dict[str, Any]:
        if not self.is_loaded:
            self.load()
            
        seq = normalize_sequence(sequence)
        if not seq:
            return {"sequence_score": 0.0, "exact_match": False, "nearest_similarity": 0.0}
            
        exact_match = seq in self.sequences
        
        if exact_match:
            # User rule: if exact match length wise and sequence wise -> 1 out of 1
            return {
                "sequence_score": 1.0,
                "exact_match": True,
                "nearest_similarity": 1.0,
                "sequence_length_score": 1.0,
                "amino_acid_validity_score": 1.0,
                "amino_acid_composition_score": 1.0,
                "kmer_similarity_score": 1.0
            }
            
        # 1. Length Score
        # E.g. normalized by median
        diff = abs(len(seq) - self.median_length)
        length_score = max(0.0, 1.0 - (diff / max(self.median_length, 1)))
        
        # 2. Validity Score
        valid_chars = set("ACDEFGHIKLMNPQRSTVWY")
        valid_count = sum(1 for c in seq if c in valid_chars)
        validity_score = valid_count / len(seq)
        
        # 3. Composition Score
        cand_counts = Counter(seq)
        cand_freq = {k: v / len(seq) for k, v in cand_counts.items()}
        # Cosine similarity between composition vectors
        keys = set(cand_freq.keys()).union(set(self.global_aa_freq.keys()))
        vec1 = np.array([cand_freq.get(k, 0) for k in keys])
        vec2 = np.array([self.global_aa_freq.get(k, 0) for k in keys])
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        comp_score = 0.0
        if norm1 > 0 and norm2 > 0:
            comp_score = float(np.dot(vec1, vec2) / (norm1 * norm2))
            
        # 4. K-mer / Nearest neighbor similarity
        cand_vec = self.vectorizer.transform([seq])
        distances, _ = self.nn_model.kneighbors(cand_vec)
        # cosine distance to similarity
        nearest_sim = max(0.0, 1.0 - distances[0][0])
        
        # Combine into sequence score
        # Let's weight nearest_similarity highly, and penalize heavily if validity is low
        sequence_score = (
            (nearest_sim * 0.5) + 
            (comp_score * 0.3) + 
            (length_score * 0.2)
        ) * (validity_score ** 2) # Heavily penalize non-standard amino acids
        
        return {
            "sequence_score": float(sequence_score),
            "exact_match": False,
            "nearest_similarity": float(nearest_sim),
            "sequence_length_score": float(length_score),
            "amino_acid_validity_score": float(validity_score),
            "amino_acid_composition_score": float(comp_score),
            "kmer_similarity_score": float(nearest_sim)
        }

if __name__ == "__main__":
    scorer = PlantPepDBScorer("cleaned_sequences.fasta")
    print(scorer.score_candidate("MELCWLTTIHGS"))
