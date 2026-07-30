import json
import csv
import re
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
import numpy as np

def normalize_sequence(seq):
    return re.sub(r"[^A-Z]", "", str(seq).upper())

def main():
    gold_file = Path("../data/gold_standard_final.json")
    output_dir = Path("output/results")
    extracted_file = Path("output/verified_peptides.csv")
    
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        
    # Load Gold Standard
    with open(gold_file, "r") as f:
        gold_data = json.load(f)
        
    gold_standard = defaultdict(set)
    for item in gold_data:
        pmid = str(item.get("pmid", ""))
        seq = normalize_sequence(item.get("sequence", ""))
        if pmid and seq:
            gold_standard[pmid].add(seq)
            
    # Load Extracted
    extracted_data = defaultdict(set)
    if extracted_file.exists():
        with open(extracted_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                pmid = str(row.get("PMID", ""))
                seq = normalize_sequence(row.get("Sequence", ""))
                if pmid and seq:
                    extracted_data[pmid].add(seq)
    else:
        print(f"Error: {extracted_file} not found. Ensure extraction has finished.")
        return
        
    all_pmids = set(gold_standard.keys()).union(set(extracted_data.keys()))
    
    results = []
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    for pmid in sorted(list(all_pmids)):
        gold_set = gold_standard[pmid]
        ext_set = extracted_data[pmid]
        
        tp = len(gold_set.intersection(ext_set))
        fp = len(ext_set - gold_set)
        fn = len(gold_set - ext_set)
        
        total_tp += tp
        total_fp += fp
        total_fn += fn
        
        results.append({
            "PMID": pmid,
            "TP": tp,
            "FP": fp,
            "FN": fn
        })
        
    # Global metrics
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    report_content = (
        f"Evaluation Report\n"
        f"=================\n\n"
        f"Total True Positives (TP): {total_tp}\n"
        f"Total False Positives (FP): {total_fp}\n"
        f"Total False Negatives (FN): {total_fn}\n\n"
        f"Precision: {precision:.4f}\n"
        f"Recall: {recall:.4f}\n"
        f"F1 Score: {f1:.4f}\n\n"
        f"Per-PMID Results saved to metrics_by_pmid.csv\n"
    )
    
    with open(output_dir / "evaluation_report.txt", "w") as f:
        f.write(report_content)
        
    with open(output_dir / "metrics_by_pmid.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["PMID", "TP", "FP", "FN"])
        writer.writeheader()
        for r in results:
            writer.writerow(r)
            
    # Plotting Global Metrics
    plt.figure(figsize=(8, 6))
    bars = plt.bar(["TP", "FP", "FN"], [total_tp, total_fp, total_fn], color=["green", "red", "orange"])
    plt.title("Overall Extraction Metrics")
    plt.ylabel("Count")
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, int(yval), va='bottom', ha='center')
    plt.savefig(output_dir / "global_metrics.png")
    plt.close()
    
    # Plotting Per-PMID Metrics if not too many
    if len(results) <= 20:
        labels = [r["PMID"] for r in results]
        tps = [r["TP"] for r in results]
        fps = [r["FP"] for r in results]
        fns = [r["FN"] for r in results]
        
        x = np.arange(len(labels))
        width = 0.25
        
        plt.figure(figsize=(12, 6))
        plt.bar(x - width, tps, width, label='TP', color='green')
        plt.bar(x, fps, width, label='FP', color='red')
        plt.bar(x + width, fns, width, label='FN', color='orange')
        
        plt.ylabel('Count')
        plt.title('Extraction Metrics by PMID')
        plt.xticks(x, labels, rotation=45, ha="right")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "metrics_by_pmid.png")
        plt.close()

    print("Evaluation completed. Results saved to:", output_dir)

if __name__ == "__main__":
    main()
