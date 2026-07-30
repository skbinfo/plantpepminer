#!/usr/bin/env python3

import json
import re
import argparse
from pathlib import Path

import fitz
import pandas as pd
import pytesseract

from PIL import Image, ImageOps
from Bio import SeqIO

MIN_PEPTIDE_LENGTH = 6
MAX_PEPTIDE_LENGTH = 120
COMMON_WORD_ONLY = set("AEILNSTR")

# =====================================================
# CONFIG
# =====================================================

AA = "ACDEFGHIKLMNPQRSTVWY"

AA_PATTERN = re.compile(rf"^[{AA}]+$")
SEQ_REGEX = re.compile(r"\b[A-Z01]{3,120}(?:\s+[A-Z01]{3,120}){0,3}\b")
CANDIDATE_REGEX = re.compile(r"\b[A-Z01]{3,120}(?:\s+[A-Z01]{3,120}){0,3}\b")


BLACKLIST = {
    "REFERENCES",
    "REFERENCE",
    "ARTICLE",
    "LETTER",
    "FASTA",
    "QIAGEN",
    "METHODS",
    "RESULTS",
    "DISCUSSION",
    "ABSTRACT",
    "SEQUENCE",
    "SEQUENCES",
    "PEPTIDE",
    "PEPTIDES",
    "PROTEIN",
    "PROTEINS",
    "AUTHOR",
    "AUTHORS",
    "JOURNAL",
    "EDITOR",
    "EDITORS",
    "REVIEW",
    "REVIEWS",
    "MERISTEMS",
    "ICREA",
    "RIKEN",
    "ADVANCE",
    "TECHNICAL",
    "MARKER",
    "FERRITIN",
    "CATALASE",
    "LACTATE",
    "DERIVATIVES",
    "SIGNIFICANCE",
    "TRANSCRIPTS",
    "ASSISTANCE",
    "RECEIVE",
    "ANGELA",
    "KATHARINA",
    "STACEY",
    "INTRODUCTION",
    "BACKGROUND",
    "CONCLUSION",
    "CONCLUSIONS",
    "MATERIALS",
    "EXPERIMENTAL",
    "SUPPLEMENTARY",
    "SUPPLEMENTAL",
    "FIGURE",
    "FIGURES",
    "TABLE",
    "TABLES",
    "LEGEND",
    "LEGENDS",
    "APPENDIX",
    "DATASET",
    "DATABASE",
    "GENOME",
    "GENOMIC",
    "GENE",
    "GENES",
    "MRNA",
    "CDNA",
    "DNA",
    "RNA",
    "PCR",
    "QPCR",
    "ELISA",
    "MALDI",
    "TOF",
    "SDS",
    "PAGE",
    "HPLC",
    "LCMS",
    "TRICINE",
    "EDMAN",
    "WESTERN",
    "BLOT",
    "SPECTROMETRY",
    "CHROMATOGRAPHY",
    "ARABIDOPSIS",
    "ESCHERICHIA",
    "BACILLUS",
    "STAPHYLOCOCCUS",
    "PSEUDOMONAS",
    "CANDIDA",
    "SUPPLEMENT",
    "SUPPLEMENTS"
}

PEPTIDE_CONTEXT_KEYWORDS = [
    "peptide",
    "peptides",
    "amino acid",
    "sequence",
    "synthetic peptide",
    "antimicrobial peptide",
    "residues",
    "purified peptide",
    "isolated peptide",
    "deduced peptide",
    "mature peptide",
    "precursor peptide",
    "fragment",
    "mass spectrometry",
    "ms/ms",
    "maldi",
    "edman",
    "n-terminal",
    "hplc"
]

TABLE_CONTEXT_KEYWORDS = [
    "table",
    "supplementary file",
    "supplemental file",
    "fasta",
    "sequence column",
    "peptide sequence",
    "sequence",
    "amino acid sequence"
]

SECTION_TITLE_WORDS = {
    "ABSTRACT",
    "INTRODUCTION",
    "BACKGROUND",
    "RESULTS",
    "METHODS",
    "MATERIALS",
    "DISCUSSION",
    "CONCLUSION",
    "CONCLUSIONS",
    "REFERENCES",
    "ACKNOWLEDGMENTS",
    "ACKNOWLEDGEMENTS",
    "FIGURE",
    "TABLE",
    "SUPPLEMENTARY"
}

PMID_PATTERNS = [
    r"PMID[:\s]+(\d+)",
    r"PubMed\s+ID[:\s]+(\d+)",
    r"PubMed\s+Identifier[:\s]+(\d+)"
]

# =====================================================
# Concatinate sequence properly
# =====================================================

def fix_hyphenated_sequences(text):

    prev = None

    while prev != text:

        prev = text

        text = re.sub(
            rf'([{AA}]{{2,}})-\s*\n\s*([{AA}]{{2,}})',
            r'\1\2',
            text
        )

        text = re.sub(
            rf'([{AA}]{{1,}})-\s*\n\s*([{AA}]{{1,}})',
            r'\1\2',
            text
        )

    return text
    

# =====================================================
# Image Processing
# =====================================================

def preprocess_for_ocr(img_path):

    img = Image.open(img_path)

    img = ImageOps.grayscale(img)

    img = ImageOps.autocontrast(img)

    img = img.resize(
        (
            img.width * 3,
            img.height * 3
        )
    )

    return img

def remove_back_matter(text):

    stop_sections = [
        "Acknowledgments",
        "Acknowledgements",
        "References",
        "Literature Cited",
        "Bibliography"
    ]

    lines = text.splitlines()

    cutoff = len(lines)

    for i, line in enumerate(lines):

        for section in stop_sections:

            if re.match(
                rf"^\s*(?:\[?\d+\]?\.?\s*)?{re.escape(section)}[\.\:\s]*$",
                line,
                flags=re.IGNORECASE
            ):
                cutoff = min(cutoff, i)

    return "\n".join(lines[:cutoff])

def extract_abbreviations(text):
    abbrevs = set()
    match = re.search(r"Abbreviations\s*:\s*(.*)", text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        abbrev_text = match.group(1).split("\n\n")[0]
        parts = re.split(r'[;]', abbrev_text)
        for part in parts:
            part = part.strip()
            submatch = re.match(r"^([A-Za-z0-9]+)\s*,", part)
            if submatch:
                abbrevs.add(submatch.group(1))
            else:
                submatch2 = re.match(r"(?:^|\n)([A-Za-z0-9]+)\s*,", part)
                if submatch2:
                    abbrevs.add(submatch2.group(1))
    return abbrevs

# =====================================================
# PMID
# =====================================================

def extract_pmid(text):

    for pattern in PMID_PATTERNS:

        m = re.search(
            pattern,
            text,
            flags=re.IGNORECASE
        )

        if m:
            return m.group(1)

    return None


# =====================================================


def merge_numbered_sequence_lines(text):
    lines = text.splitlines()
    merged_lines = []
    
    # Pattern: optional spaces, start number, sequence(letters and spaces), optional end number
    pattern = re.compile(r"^\s*(\d+)\s+([A-Z\s]+?)(?:\s+(\d+))?\s*$")
    
    i = 0
    while i < len(lines):
        line = lines[i]
        m1 = pattern.match(line)
        if not m1:
            merged_lines.append(line)
            i += 1
            continue
            
        start1 = int(m1.group(1))
        seq1 = re.sub(r"\s+", "", m1.group(2))
        
        # Check if it's a pure AA sequence
        if not re.match(r"^[ACDEFGHIKLMNPQRSTVWY]+$", seq1, re.IGNORECASE):
            merged_lines.append(line)
            i += 1
            continue
            
        # It's a valid numbered sequence line. Let's see if the next line is a continuation.
        expected_next_start = start1 + len(seq1)
        
        j = i + 1
        while j < len(lines):
            nxt_line = lines[j]
            m2 = pattern.match(nxt_line)
            if not m2:
                break
                
            start2 = int(m2.group(1))
            seq2 = re.sub(r"\s+", "", m2.group(2))
            
            if start2 == expected_next_start and re.match(r"^[ACDEFGHIKLMNPQRSTVWY]+$", seq2, re.IGNORECASE):
                expected_next_start += len(seq2)
                j += 1
            else:
                break
                
        if j > i + 1:
            # We merged something! Replace the whole block with the merged pure sequence
            merged_seq = "".join(re.sub(r"[\s\d]+", "", l) for l in lines[i:j])
            merged_lines.append(merged_seq)
            i = j
        else:
            merged_lines.append(line)
            i += 1

    return "\n".join(merged_lines)

def join_numbered_sequences(text):

    def replacer(match):
        return re.sub(r"[\s\d]+", "", match.group(0))

    lines = text.splitlines()
    merged = []
    
    aa_word = rf"[{AA}]{{3,}}"
    pattern = rf"\b{aa_word}(?:[\s\d]+{aa_word})+\b"

    for line in lines:
        new_line = re.sub(pattern, replacer, line)
        merged.append(new_line)

    return "\n".join(merged)

# =====================================================
# SEQUENCE EXTRACTION
# =====================================================

def normalize_ocr_sequence(seq):

    return (
        seq.upper()
        .replace("0", "Q")
        .replace("O", "Q")
        .replace("1", "@")
        .replace("I", "L")
        .replace("@", "I")
    )


def normalize_candidate_sequence(seq, source):

    seq = re.sub(r"[^A-Z01]", "", seq.upper())

    if source == "ocr":
        seq = normalize_ocr_sequence(seq)

    return seq


def reconstruct_sequence_text(text):

    text = fix_hyphenated_sequences(text)

    def join_hyphenated_candidate(match):

        left = match.group(1)
        right = match.group(2)

        if left in BLACKLIST or right in BLACKLIST:
            return match.group(0)

        return left + right

    prev = None

    while prev != text:
        prev = text

        text = re.sub(
            r"\b([A-Z01]{2,})-([A-Z01]{2,})\b",
            join_hyphenated_candidate,
            text
        )

        text = re.sub(
            r"\b((?:[A-Z01][ \t\-]+){3,}[A-Z01])\b",
            lambda m: re.sub(r"[ \t\-]+", "", m.group(1)),
            text
        )

        text = re.sub(
            r"\b([A-Z01]{2,})-\s*\n\s*([A-Z01]{2,})\b",
            r"\1\2",
            text
        )

    return text


def is_section_title(seq, context):

    if seq in SECTION_TITLE_WORDS:
        return True

    compact_context = context.strip()

    return (
        seq in SECTION_TITLE_WORDS and
        len(compact_context.split()) <= 5
    )


def is_likely_gene_or_protein_name(seq, context):

    if re.fullmatch(r"[A-Z]{2,6}\d{1,3}[A-Z]?", seq):
        return True

    context_lower = context.lower()

    gene_terms = [
        "gene",
        "genes",
        "mrna",
        "transcript",
        "transcripts",
        "cdna",
        "orf",
        "locus",
        "protein family",
        "domain"
    ]

    peptide_terms = [
        "peptide",
        "amino acid",
        "residue",
        "sequence",
        "mass spectrometry",
        "maldi",
        "edman"
    ]

    return (
        any(term in context_lower for term in gene_terms) and
        not any(term in context_lower for term in peptide_terms)
    )


def is_likely_peptide(seq, context):

    if len(seq) < MIN_PEPTIDE_LENGTH or len(seq) > MAX_PEPTIDE_LENGTH:
        return False

    if seq in BLACKLIST or is_section_title(seq, context):
        return False

    if not AA_PATTERN.fullmatch(seq):
        return False

    if len(set(seq)) <= 2 and len(seq) > 8:
        return False

    if set(seq).issubset(COMMON_WORD_ONLY) and len(seq) < 8:
        return False

    if is_likely_gene_or_protein_name(seq, context):
        return False

    return True


def score_candidate(seq, context, source, occurrences=1, table_like=False):

    score = 0
    context_lower = context.lower()

    for keyword in PEPTIDE_CONTEXT_KEYWORDS:
        if keyword in context_lower:
            score += 3

    for keyword in TABLE_CONTEXT_KEYWORDS:
        if keyword in context_lower:
            score += 5

    if table_like:
        score += 8

    if re.search(rf"\b(peptide|sequence|residues?)\W{{0,20}}{re.escape(seq)}\b", context, flags=re.IGNORECASE):
        score += 6

    if re.search(rf"\b{re.escape(seq)}\W{{0,30}}(peptide|sequence|residues?|analyzed|purified|isolated)\b", context, flags=re.IGNORECASE):
        score += 6

    if source in {"fasta", "csv", "tsv", "xlsx", "txt"}:
        score += 4

    if source == "ocr":
        score -= 1

    if occurrences > 1:
        score += min(occurrences - 1, 5) * 2

    # Bonus for long, high-complexity sequences (very likely real biology, not OCR artifacts)
    if len(seq) >= 20 and len(set(seq)) >= 10:
        score += 4
    elif len(seq) >= 15 and len(set(seq)) >= 8:
        score += 2

    return max(score, 0)


def find_techniques_in_context(context):
    """Return list of canonical technique names found anywhere in context."""
    found = []
    for technique, patterns in TECHNIQUE_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, context, flags=re.IGNORECASE):
                found.append(technique)
                break
    return found


def get_exact_line(text, target_str):
    """Return the first line in text that contains target_str."""
    for line in text.splitlines():
        if target_str in line:
            return line.strip()
    return ""


def find_techniques_and_lines(context):
    """Search context sentence-by-sentence for technique patterns.

    Returns:
        names  (list[str]): canonical technique names found
        sents  (list[str]): parallel list of exact sentences where each technique appears
    """
    sentences = split_sentences(context)
    names = []
    sents = []
    seen = set()

    for technique, patterns in TECHNIQUE_PATTERNS.items():
        for pattern in patterns:
            # Search sentence by sentence so we capture the exact evidence sentence
            for sent in sentences:
                if re.search(pattern, sent, flags=re.IGNORECASE):
                    if technique not in seen:
                        seen.add(technique)
                        names.append(technique)
                        sents.append(sent.strip())
                    break  # Only record first sentence per technique
            if technique in seen:
                break  # Already found this technique; move to next

    # Remove techniques that are superseded by a more-specific variant already found
    suppressed = set()
    for specific, general in TECHNIQUE_SUPERSEDES.items():
        if specific in seen:
            for g in general:
                suppressed.add(g)

    if suppressed:
        filtered_names = []
        filtered_sents = []
        for n, s in zip(names, sents):
            if n not in suppressed:
                filtered_names.append(n)
                filtered_sents.append(s)
        names, sents = filtered_names, filtered_sents

    return names, sents


# When the key technique is found, suppress the listed general techniques
# (because the more-specific name already captures the evidence)
TECHNIQUE_SUPERSEDES = {
    "MALDI-TOF MS":              ["MALDI-TOF", "Tandem MS"],
    "LC-MS/MS":                  ["Tandem MS", "HPLC"],
    "RP-HPLC":                   ["HPLC"],
    "Tricine SDS-PAGE":          ["SDS-PAGE"],
    "Wheat Germ Translation":    ["In vitro Translation"],
    "Solid Phase Peptide Synthesis": ["Synthetic Peptide"],
    "Peptide Analog":            [],  # no suppression but listed for completeness
    "ESI-QTOF":                  ["ESI-MS"],
    "Orbitrap MS":               ["Tandem MS"],
    "Strong Cation Exchange HPLC": ["HPLC"],
}


# Evidence scoring weights per canonical technique name
EVIDENCE_WEIGHTS = {
    # Anchoring base scores (applied before additive)
    "Synthetic Peptide":            2,
    "Peptide Analog":               2,
    "In vitro Translation":         3,
    "Wheat Germ Translation":       3,
    "Site-Directed Mutagenesis":    3,
    "RT-PCR":                       2,
    # Mass spectrometry additive
    "MALDI-TOF":                    4,
    "MALDI-TOF MS":                 4,
    "ESI-MS":                       4,
    "ESI-QTOF":                     4,
    "Tandem MS":                    4,
    "Orbitrap MS":                  4,
    "LC-MS/MS":                     5,
    # Sequencing additive
    "Edman Sequencing":             5,
    "Peptide Sequencing":           5,
    "Protein Sequencing":           5,
    # Gel/immunological additive
    "SDS-PAGE":                     3,
    "Tricine SDS-PAGE":             3,
    "Native PAGE":                  2,
    "Western Blot":                 5,
    "Immunoprecipitation":          4,
    "Pull-down Assay":              4,
    "ELISA":                        3,
    # Chromatography additive
    "RP-HPLC":                      3,
    "HPLC":                         2,
    "Strong Cation Exchange HPLC":  3,
    "Affinity Chromatography":      2,
    "Size Exclusion Chromatography":2,
    "Solid Phase Peptide Synthesis":2,
    # Structural additive
    "NMR":                          4,
    "Circular Dichroism":           3,
    "X-ray Crystallography":        4,
    "Cryo-EM":                      4,
    # Ribosome / translation
    "Ribosome Profiling":           2,
}


def compute_evidence_score(techniques):
    """Compute cumulative evidence score from a list of canonical technique names.

    Scoring logic:
    - If no techniques are found, the score is 1 (regex-only detection).
    - Otherwise, sum the weights for each unique technique found.
    """
    if not techniques:
        return 1.0
    total = 0.0
    seen = set()
    for t in techniques:
        if t not in seen and t in EVIDENCE_WEIGHTS:
            total += EVIDENCE_WEIGHTS[t]
            seen.add(t)
    # If techniques were found but none matched the weight table (e.g. Agarose Gel),
    # treat as regex-only.
    return total if total > 0 else 1.0



def is_table_like_context(context):

    context_lower = context.lower()

    if any(k in context_lower for k in TABLE_CONTEXT_KEYWORDS):
        return True

    lines = [line for line in context.splitlines() if line.strip()]

    return any(
        len(re.split(r"\s{2,}|\t", line.strip())) >= 2
        for line in lines
    )


def extract_candidate_sequences(text, source="text", bad_words=None):

    seen = set()
    results = []
    reconstructed_text = reconstruct_sequence_text(text)
    occurrences = {}

    for raw_seq in CANDIDATE_REGEX.findall(reconstructed_text):
        seq = normalize_candidate_sequence(raw_seq, source)
        occurrences[seq] = occurrences.get(seq, 0) + 1

    for match in CANDIDATE_REGEX.finditer(reconstructed_text):

        original_seq = match.group()
        seq = normalize_candidate_sequence(original_seq, source)

        if seq in seen:
            continue
            
        if bad_words and seq in bad_words:
            continue
        
        start = max(0, match.start() - 1000)
        end = min(len(reconstructed_text), match.end() + 1000)

        context = reconstructed_text[start:end]
        table_like = is_table_like_context(context)

        if not is_likely_peptide(seq, context):
            continue

        score = score_candidate(
            seq,
            context,
            source,
            occurrences.get(seq, 1),
            table_like
        )

        if score < 3:
            continue
        
        seen.add(seq)

        techs, tech_lines = find_techniques_and_lines(context)
        evidence_score = compute_evidence_score(techs)
        results.append({
            "sequence": seq,
            "original_sequence": original_seq,
            "normalized_sequence": seq,
            "length": len(seq),
            "score": score,
            "source": source,
            "context": context,
            "techniques": techs,
            "technique_lines": tech_lines,
            "sequence_line": get_exact_line(reconstructed_text, original_seq),
            "evidence_score": evidence_score
        })

    return results


def extract_table_sequence_candidates(text, source="pdf_text", bad_words=None):

    results = []
    seen = set()
    lines = text.splitlines()

    for i, line in enumerate(lines):
        if not re.search(r"\b(peptide|sequence|residue|amino acid)\b", line, flags=re.IGNORECASE):
            continue

        window_lines = lines[i:i + 20]

        for j, row in enumerate(window_lines[1:], start=i + 1):
            cells = [c.strip() for c in re.split(r"\t|\s{2,}", row) if c.strip()]

            if len(cells) < 2:
                cells = row.split()

            for cell in cells:
                for match in CANDIDATE_REGEX.finditer(reconstruct_sequence_text(cell)):
                    original_seq = match.group()
                    seq = normalize_candidate_sequence(original_seq, source)
                    
                    if bad_words and seq in bad_words:
                        continue
                        
                    context_start = max(0, j - 3)
                    context_end = min(len(lines), j + 4)
                    context = "\n".join(lines[context_start:context_end])

                    if seq in seen or not is_likely_peptide(seq, context):
                        continue

                    seen.add(seq)
                    score = score_candidate(seq, context, source, 1, True)

                    techs, tech_lines = find_techniques_and_lines(context)
                    evidence_score = compute_evidence_score(techs)
                    results.append({
                        "sequence": seq,
                        "original_sequence": original_seq,
                        "normalized_sequence": seq,
                        "length": len(seq),
                        "score": score,
                        "source": source,
                        "context": context,
                        "techniques": techs,
                        "technique_lines": tech_lines,
                        "sequence_line": get_exact_line(text, original_seq),
                        "evidence_score": evidence_score
                    })

    return results


# =====================================================
# SUPPLEMENTARY PARSING
# =====================================================

def add_sequence(records, seq, source, identifier=None, context=""):

    original_seq = seq.upper()
    seq = normalize_candidate_sequence(original_seq, source)

    if not is_likely_peptide(seq, context):
        return

    score = score_candidate(
        seq,
        context,
        source,
        1,
        source in {"csv", "tsv", "xlsx", "fasta"}
    )

    techs, tech_lines = find_techniques_and_lines(context)
    evidence_score = compute_evidence_score(techs)
    results_dict = {
        "id": identifier,
        "sequence": seq,
        "original_sequence": original_seq,
        "normalized_sequence": seq,
        "length": len(seq),
        "score": score,
        "source": source,
        "context": context,
        "techniques": techs,
        "technique_lines": tech_lines,
        "sequence_line": "from supplementary",
        "evidence_score": evidence_score
    }
    
    records.append(results_dict)


def parse_supp_file(filepath):

    filepath = Path(filepath)

    ext = filepath.suffix.lower()

    records = []

    if ext in [".fa", ".fasta", ".fna"]:

        for record in SeqIO.parse(filepath, "fasta"):

            add_sequence(
                records,
                str(record.seq),
                "fasta",
                record.id,
                record.description
            )

    elif ext == ".csv":

        df = pd.read_csv(
            filepath,
            dtype=str,
            keep_default_na=False
        )

        for _, row in df.iterrows():

            row_context = " | ".join(str(value) for value in row)

            for value in row:

                for seq in SEQ_REGEX.findall(str(value)):

                    add_sequence(
                        records,
                        seq,
                        "csv",
                        context=row_context
                    )

    elif ext == ".tsv":

        df = pd.read_csv(
            filepath,
            sep="\t",
            dtype=str,
            keep_default_na=False
        )

        for _, row in df.iterrows():

            row_context = " | ".join(str(value) for value in row)

            for value in row:

                for seq in SEQ_REGEX.findall(str(value)):

                    add_sequence(
                        records,
                        seq,
                        "tsv",
                        context=row_context
                    )

    elif ext in [".xlsx", ".xls"]:

        sheets = pd.read_excel(
            filepath,
            sheet_name=None,
            dtype=str
        )

        for _, df in sheets.items():

            df = df.fillna("")

            for _, row in df.iterrows():

                row_context = " | ".join(str(value) for value in row)

                for value in row:

                    for seq in SEQ_REGEX.findall(str(value)):

                        add_sequence(
                            records,
                            seq,
                            "xlsx",
                            context=row_context
                        )

    elif ext == ".txt":

        text = filepath.read_text(
            encoding="utf-8",
            errors="ignore"
        )

        for seq in SEQ_REGEX.findall(text):

            add_sequence(
                records,
                seq,
                "txt",
                context=text[:500]
            )

    return records


# =====================================================
# TECHNIQUE EXTRACTION
# =====================================================

# Each key is the EXACT canonical technique name that will appear in the output CSV.
# These are specific experimental techniques, not broad categories.
TECHNIQUE_PATTERNS = {
    # ── Mass Spectrometry ─────────────────────────────────────────────────────
    "MALDI-TOF": [
        r"\bMALDI[\-\s]TOF(?!\s*MS)(?!\s*/)",
        r"\bMALDI\s+TOF(?!\s*MS)",
        r"matrix[\-\s]assisted\s+laser\s+desorption[/\\\s]ionization",
    ],
    "MALDI-TOF MS": [
        r"\bMALDI[\-\s]TOF[\-\s]MS\b",
        r"\bMALDI[\-\s]TOF\s+mass\s+spectrometry\b",
        r"\bMALDI[\-\s]TOF/TOF\b",
    ],
    "LC-MS/MS": [
        r"\bLC[\-\s]MS/MS\b",
        r"\bnano[\-\s]LC[\-\s]MS/MS\b",
        r"\bnanoLC[\-\s]MS/MS\b",
        r"\bliquid\s+chromatography[\-\s]tandem\s+mass\s+spectrometry\b",
        r"\bLC/MS/MS\b",
    ],
    "ESI-MS": [
        r"\bESI[\-\s]MS(?!/MS)\b",
        r"\belectrospray\s+ionization\s+mass\s+spectrometry\b",
        r"\belectrospray\s+mass\s+spectrometry\b",
    ],
    "ESI-QTOF": [
        r"\bESI[\-\s]QTOF\b",
        r"\bQ[\-]TOF\b",
        r"\bquadrupole\s+time[\-]of[\-]flight\b",
    ],
    "Orbitrap MS": [
        r"\bOrbitrap\b",
        r"\bFT[\-]ICR\s*MS\b",
        r"\bFourier\s+transform\s+ion\s+cyclotron\b",
    ],
    "Tandem MS": [
        r"\bMS/MS\b",
        r"\btandem\s+mass\s+spectrometry\b",
        r"\btriple\s+quadrupole\b",
        r"\bMS2\b",
    ],
    # ── Sequencing ────────────────────────────────────────────────────────────
    "Edman Sequencing": [
        r"\bEdman\s+degradation\b",
        r"\bEdman\s+sequencing\b",
        r"\bEdman\s+microsequencing\b",
        r"\bN[\-]terminal\s+sequencing\b",
    ],
    "Peptide Sequencing": [
        r"\bpeptide\s+sequencing\b",
        r"\bde\s+novo\s+sequencing\b",
        r"\bde\s+novo\s+peptide\s+sequencing\b",
        r"\bMS/MS\s+sequencing\b",
    ],
    "Protein Sequencing": [
        r"\bprotein\s+sequencing\b",
        r"\bN[\-]terminal\s+amino\s+acid\s+sequencing\b",
        r"\baminoacid\s+sequencing\b",
    ],
    # ── Gel Electrophoresis ────────────────────────────────────────────────────
    "SDS-PAGE": [
        r"\bSDS[\-\s]PAGE\b",
        r"\bsodium\s+dodecyl\s+sulfate[\-\s]polyacrylamide\b",
    ],
    "Tricine SDS-PAGE": [
        r"\bTricine[\-\s]SDS[\-\s]PAGE\b",
        r"\bTricine\s+gel\b",
    ],
    "Native PAGE": [
        r"\bNative[\-\s]PAGE\b",
        r"\bBlue\s+Native\s+PAGE\b",
        r"\bnon[\-]denaturing\s+PAGE\b",
    ],
    "Agarose Gel": [
        r"\bagarose\s+gel\b",
        r"\bagarose\s+gel\s+electrophoresis\b",
    ],
    # ── Immunological ─────────────────────────────────────────────────────────
    "Western Blot": [
        r"\bWestern\s+blot\b",
        r"\bWestern\s+blotting\b",
        r"\bimmunoblot\b",
        r"\bimmuno[\-]blot\b",
    ],
    "Immunoprecipitation": [
        r"\bimmunoprecipitation\b",
        r"\bco[\-]immunoprecipitation\b",
        r"\bCoIP\b",
    ],
    "ELISA": [
        r"\bELISA\b",
        r"\benzyme[\-]linked\s+immunosorbent\b",
    ],
    # ── Synthetic Peptide ──────────────────────────────────────────────────────
    "Synthetic Peptide": [
        r"\bsynthetic\s+peptide\b",
        r"\bchemically\s+synthesized\s+peptide\b",
        r"\bpeptide\s+synthesis\b",
        r"\bpeptides?\s+were\s+synthesized\b",
        r"\bpeptide\s+purchased\s+from\b",
        r"\bGenScript\b",
        r"\bSmartox\b",
        r"\bGL\s+Biochem\b",
        r"\bPeptide\s+2\.0\b",
        r"\bThermo\s+Fisher\s+custom\s+peptide\b",
        r"\bAutomated\s+peptide\s+synthesizer\b",
        r"\bcustom\s+peptide\s+synthesis\b",
    ],
    "Solid Phase Peptide Synthesis": [
        r"\bSolid[\-\s]Phase\s+Peptide\s+Synthesis\b",
        r"\bSPPS\b",
        r"\bFmoc\s+chemistry\b",
        r"\bBoc\s+chemistry\b",
        r"\bFmoc[\-]based\s+synthesis\b",
    ],
    "Peptide Analog": [
        r"\bpeptide\s+analog\b",
        r"\bsynthetic\s+analog\b",
        r"\bbiotinylated\s+peptide\b",
        r"\bFITC[\-]labell?ed\s+peptide\b",
        r"\bfluorescent\s+peptide\b",
        r"\bfluorescein[\-]labell?ed\s+peptide\b",
    ],
    # ── Chromatography ─────────────────────────────────────────────────────────
    "RP-HPLC": [
        r"\bRP[\-]HPLC\b",
        r"\breverse[d]?[\-\s]phase\s+HPLC\b",
        r"\breverse[d]?[\-\s]phase\s+high[\-\s]performance\s+liquid\s+chromatography\b",
        r"\breverse[d]?[\-\s]phase\s+liquid\s+chromatography\b",
    ],
    "HPLC": [
        # RP-HPLC is matched first (dict insertion order), so this catches standalone HPLC
        r"\bHPLC\b",
        r"\bhigh[\-]performance\s+liquid\s+chromatography\b",
        r"\bhigh[\-]pressure\s+liquid\s+chromatography\b",
    ],
    "Strong Cation Exchange HPLC": [
        r"\bstrong\s+cation\s+exchange\b",
        r"\bSCX[\-]HPLC\b",
        r"\bcation\s+exchange\s+HPLC\b",
    ],
    "Affinity Chromatography": [
        r"\baffinity\s+chromatography\b",
        r"\baffinity\s+purification\b",
        r"\bGST\s+pull[\-]down\b",
        r"\bHis[\-]tag\s+purification\b",
    ],
    "Size Exclusion Chromatography": [
        r"\bsize\s+exclusion\s+chromatography\b",
        r"\bSEC\b",
        r"\bgel\s+filtration\b",
    ],
    "Reverse Phase Chromatography": [
        r"\breverse[d]?[\-\s]phase\s+chromatography\b(?!.*HPLC)",
        r"\bRP\s+chromatography\b",
    ],
    # ── Translation / Expression ────────────────────────────────────────────────
    "In vitro Translation": [
        r"\bin\s+vitro\s+translation\b",
        r"\bcell[\-]free\s+translation\b",
        r"\bRabbit\s+Reticulocyte\s+Lysate\b",
        r"\btranslation\s+assay\b",
    ],
    "Wheat Germ Translation": [
        r"\bwheat\s+germ\s+extract\b",
        r"\bwheat\s+germ\s+translation\b",
        r"\bwheat[\-]germ\s+cell[\-]free\b",
    ],
    "Site-Directed Mutagenesis": [
        r"\bsite[\-]directed\s+mutagenesis\b",
        r"\bsite[\-]specific\s+mutagenesis\b",
        r"\bpoint\s+mutation\b",
    ],
    "RT-PCR": [
        r"\bRT[\-]PCR\b",
        r"\breverse\s+transcription\s+PCR\b",
        r"\bqRT[\-]PCR\b",
    ],
    "Ribosome Profiling": [
        r"\bribosome\s+profiling\b",
        r"\bRibo[\-]seq\b",
        r"\bpolysome\s+profiling\b",
    ],
    # ── Pull-down / Interaction ────────────────────────────────────────────────
    "Pull-down Assay": [
        r"\bpull[\-]down\s+assay\b",
        r"\bpull\s+down\s+assay\b",
        r"\bGST[\-]pull[\-]down\b",
        r"\bHis[\-]pull[\-]down\b",
    ],
    # ── Structural ─────────────────────────────────────────────────────────────
    "NMR": [
        r"\bNMR\b",
        r"\bnuclear\s+magnetic\s+resonance\b",
    ],
    "Circular Dichroism": [
        r"\bcircular\s+dichroism\b",
        r"\bCD\s+spectroscopy\b",
        r"\bCD\s+spectra\b",
    ],
    "X-ray Crystallography": [
        r"\bX[\-]ray\s+crystal\w*\b",
        r"\bprotein\s+crystallography\b",
    ],
    "Cryo-EM": [
        r"\bcryo[\-]EM\b",
        r"\bcryo[\-]electron\s+microscopy\b",
    ],
}


def split_sentences(text):

    text = text.replace("\n", " ")

    return [
        s.strip()
        for s in re.split(
            r"(?<=[.!?])\s+",
            text
        )
        if s.strip()
    ]


def find_techniques(text, source):

    results = []
    seen = set()

    for sentence in split_sentences(text):

        for technique, patterns in TECHNIQUE_PATTERNS.items():

            for pattern in patterns:

                if re.search(
                    pattern,
                    sentence,
                    flags=re.IGNORECASE
                ):

                    key = (
                        technique,
                        sentence
                    )

                    if key in seen:
                        break

                    seen.add(key)

                    results.append({
                        "technique": technique,
                        "statement": sentence,
                        "source": source
                    })

                    break

    return results


def find_candidate_page(candidate, page_texts):

    seq = candidate["sequence"]
    original_seq = candidate.get("original_sequence", seq)

    for page in page_texts:
        page_text = reconstruct_sequence_text(page["text"])

        if seq in page_text or original_seq in page_text:
            return page["page"]

    return None


def build_sequence_evidence(candidate):

    context = candidate.get("context", "")
    techniques = candidate.get("techniques") or find_techniques_in_context(context)
    technique_lines = candidate.get("technique_lines", [])
    # Back-fill technique_lines if missing (older candidates)
    if not technique_lines and techniques:
        _, technique_lines = find_techniques_and_lines(context)

    return {
        "sequence": candidate["sequence"],
        "original_sequence": candidate.get("original_sequence", candidate["sequence"]),
        "normalized_sequence": candidate.get("normalized_sequence", candidate["sequence"]),
        "sequence_line": candidate.get("sequence_line", ""),
        "techniques": techniques,
        "technique_lines": technique_lines,
        "evidence_score": candidate.get("evidence_score", compute_evidence_score(techniques)),
        "evidence_text": context,
        "score": candidate.get("score", 0),
        "source": candidate.get("source")
    }


def build_llm_candidate(candidate, pmid, page_texts):

    return {
        "pmid": pmid,
        "sequence": candidate["sequence"],
        "original_sequence": candidate.get("original_sequence", candidate["sequence"]),
        "normalized_sequence": candidate.get("normalized_sequence", candidate["sequence"]),
        "score": candidate.get("score", 0),
        "context": candidate.get("context", ""),
        "sequence_line": candidate.get("sequence_line", ""),
        "techniques": candidate.get("techniques", []),
        "technique_lines": candidate.get("technique_lines", []),
        "evidence_score": candidate.get("evidence_score", 1.0),
        "source": candidate.get("source"),
        "page": find_candidate_page(candidate, page_texts)
    }


def merge_candidate_lists(candidate_lists):

    merged = {}

    for candidate_list in candidate_lists:
        for candidate in candidate_list:
            seq = candidate["sequence"]
            existing = merged.get(seq)

            if existing is None or candidate.get("score", 0) > existing.get("score", 0):
                merged[seq] = candidate
                continue

            existing_techniques = set(existing.get("techniques", []))
            existing_techniques.update(candidate.get("techniques", []))
            existing["techniques"] = sorted(existing_techniques)

            # Merge technique_lines without duplicates, preserving order
            existing_tlines = existing.get("technique_lines", [])
            new_tlines = candidate.get("technique_lines", [])
            merged_tlines_set = set(existing_tlines)
            for tl in new_tlines:
                if tl not in merged_tlines_set:
                    existing_tlines.append(tl)
                    merged_tlines_set.add(tl)
            existing["technique_lines"] = existing_tlines

            # Re-compute evidence_score from merged techniques
            existing["evidence_score"] = compute_evidence_score(existing["techniques"])

            if candidate.get("score", 0) == existing.get("score", 0):
                existing["score"] = candidate.get("score", 0)

    return sorted(
        merged.values(),
        key=lambda c: c.get("score", 0),
        reverse=True
    )

# =====================================================
# MAIN
# =====================================================

def main():

    parser = argparse.ArgumentParser()

    group = parser.add_mutually_exclusive_group(
         required=True
    )

    group.add_argument(
        "--pdf",
        help="Single PDF"
    )

    group.add_argument(
        "--pdf_dir",
        help="Directory containing PDFs"
    )

    parser.add_argument(
        "--supp_dir",
        help="Directory containing supplementary files",
        default=None
    )

    parser.add_argument(
        "--out",
        default="output"
    )

    args = parser.parse_args()

    pdf_files = []

    if args.pdf:

        pdf_files.append(
            Path(args.pdf)
        )

    elif args.pdf_dir:

        pdf_files.extend(
            sorted(
                Path(args.pdf_dir).glob("*.pdf")
            )
        )

    for pdf_file in pdf_files:

        doc = fitz.open(pdf_file)

        page_texts = []
        text_parts = []

        for page_num, page in enumerate(doc):

            txt = page.get_text()

            text_parts.append(txt)

            page_texts.append({
                "page": page_num + 1,
                "text": txt
            })

        raw_text = "\n".join(text_parts)

        pmid = extract_pmid(raw_text)

        if pmid is None:
            pmid = pdf_file.stem

        paper_dir = Path(args.out) / pmid

        paper_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        with open(
            paper_dir / "page_texts.json",
            "w"
        ) as f:

            json.dump(
                page_texts,
                f,
                indent=2
            )

        images_dir = paper_dir / "images"

        images_dir.mkdir(
            exist_ok=True
        )

        ocr_dir = paper_dir / "ocr"

        ocr_dir.mkdir(
            exist_ok=True
        )

    # --------------------------
    # TEXT
    # --------------------------

        (paper_dir / "raw_text.txt").write_text(
            raw_text,
            encoding="utf-8"
        )

        cleaned_text = raw_text
        cleaned_text = fix_hyphenated_sequences(cleaned_text)
        cleaned_text = remove_back_matter(cleaned_text)

        (paper_dir / "text.txt").write_text(
            cleaned_text,
            encoding="utf-8"
        )

    # --------------------------
    # IMAGES + OCR
    # --------------------------

        img_count = len(list(images_dir.glob("*")))
        ocr_file_path = paper_dir / "ocr.txt"
        if ocr_file_path.exists() and ocr_file_path.stat().st_size > 0:
            ocr_text = ocr_file_path.read_text(encoding="utf-8")
            print(f"Using existing OCR text for PMID {pmid}")
        else:
            ocr_parts = []

            for page_num in range(len(doc)):

                page = doc[page_num]

                for img in page.get_images(full=True):

                    try:

                        xref = img[0]

                        base_image = doc.extract_image(xref)

                        ext = base_image["ext"]

                        img_file = (
                            images_dir /
                            f"page_{page_num+1}_img_{img_count}.{ext}"
                        )

                        img_file.write_bytes(
                            base_image["image"]
                        )

                        txt = pytesseract.image_to_string(
                            preprocess_for_ocr(img_file),
                            config="--oem 3 --psm 6"
                        )
                    
                        ocr_file = (
                            ocr_dir /
                            f"page_{page_num+1}_img_{img_count}.txt"
                        )

                        ocr_file.write_text(
                            txt,
                            encoding="utf-8"
                        )

                        ocr_parts.append(txt)

                        img_count += 1

                    except Exception as e:
                        print(f"OCR error in {img_file}: {e}")

            ocr_text = "\n".join(ocr_parts)

            ocr_file_path.write_text(
                ocr_text,
                encoding="utf-8"
            )

        # --------------------------
        # TECHNIQUES
        # --------------------------
        
        techniques = []
        
        techniques.extend(
            find_techniques(
                cleaned_text,
                "pdf_text"
            )
        )
        
        techniques.extend(
            find_techniques(
                ocr_text,
                "ocr"
            )
        )

        unique = []
        seen = set()
        
        for t in techniques:
        
            key = (
                t["technique"],
                t["statement"]
            )
        
            if key in seen:
                continue
        
            seen.add(key)
        
            unique.append(t)
        
        techniques = unique
        
        with open(
            paper_dir / "techniques.json",
            "w"
        ) as f:
        
            json.dump(
                techniques,
                f,
                indent=2,
                ensure_ascii=False
            )

    # --------------------------
    # SEQUENCE CANDIDATES
    # --------------------------
        cleaned_text = merge_numbered_sequence_lines(cleaned_text)
        cleaned_text = join_numbered_sequences(cleaned_text)
        ocr_text = merge_numbered_sequence_lines(ocr_text)
        ocr_text = join_numbered_sequences(ocr_text)
        
        abbreviations = extract_abbreviations(cleaned_text)
        
        text_candidates = extract_candidate_sequences(
            cleaned_text,
            source="pdf_text",
            bad_words=abbreviations
        )
        
        ocr_candidates = extract_candidate_sequences(
            ocr_text,
            source="ocr",
            bad_words=abbreviations
        )

        table_candidates = extract_table_sequence_candidates(
            cleaned_text,
            source="pdf_text",
            bad_words=abbreviations
        )
        
        candidates = merge_candidate_lists([
            text_candidates,
            ocr_candidates,
            table_candidates
        ])
        
        with open(
            paper_dir / "sequence_candidates.json",
            "w"
        ) as f:
        
            json.dump(
                candidates,
                f,
                indent=2,
                ensure_ascii=False
            )


        with open(
            paper_dir / "evidence_chunks.json",
            "w"
        ) as f:
        
            json.dump(
                [
                    {
                        "sequence": c["sequence"],
                        "source": c["source"],
                        "context": c["context"]
                    }
                    for c in candidates
                ],
                f,
                indent=2,
                ensure_ascii=False
            )

        sequence_evidence_candidates = [
            build_sequence_evidence(c)
            for c in candidates
        ]

        with open(
            paper_dir / "sequence_evidence_candidates.json",
            "w"
        ) as f:

            json.dump(
                sequence_evidence_candidates,
                f,
                indent=2,
                ensure_ascii=False
            )

        llm_verification_candidates = [
            build_llm_candidate(c, pmid, page_texts)
            for c in candidates
        ]

        with open(
            paper_dir / "llm_verification_candidates.json",
            "w"
        ) as f:

            json.dump(
                llm_verification_candidates,
                f,
                indent=2,
                ensure_ascii=False
            )

    # --------------------------
    # SUPPLEMENTARY
    # --------------------------

        supp_sequences = []

        if args.supp_dir:

            supp_dir = Path(args.supp_dir)

            if supp_dir.exists():


                for file in sorted(supp_dir.iterdir()):
    
                    if not file.is_file():
                        continue
    
                    if not file.name.startswith(
                        str(pmid)
                    ):
                        continue
    
                    supp_sequences.extend(
                        parse_supp_file(file)
                    )
    
                seen = set()
                unique = []
                
                for r in supp_sequences:
                
                    if r["sequence"] in seen:
                        continue
                
                    seen.add(r["sequence"])
                    unique.append(r)
                
                supp_sequences = unique
    
        with open(
            paper_dir /
            "supplementary_sequences.json",
            "w"
        ) as f:

            json.dump(
                supp_sequences,
                f,
                indent=2
            )

        verification_candidates = merge_candidate_lists([
            candidates,
            supp_sequences
        ])

        sequence_evidence_candidates = [
            build_sequence_evidence(c)
            for c in verification_candidates
        ]

        with open(
            paper_dir / "sequence_evidence_candidates.json",
            "w"
        ) as f:

            json.dump(
                sequence_evidence_candidates,
                f,
                indent=2,
                ensure_ascii=False
            )

        llm_verification_candidates = [
            build_llm_candidate(c, pmid, page_texts)
            for c in verification_candidates
        ]

        with open(
            paper_dir / "llm_verification_candidates.json",
            "w"
        ) as f:

            json.dump(
                llm_verification_candidates,
                f,
                indent=2,
                ensure_ascii=False
            )

    # --------------------------
    # METADATA
    # --------------------------

        metadata = {
            "pmid": pmid,
            "pdf_name": pdf_file.name,
            "image_count": img_count,
            "supplementary_count": len(
                supp_sequences
            )
        }
        with open(
            paper_dir / "metadata.json",
            "w"
        ) as f:
            json.dump(
                metadata,
                f,
                indent=2
            )
    
        doc.close()
    
        print(
            f"Prepared input for PMID {pmid}"
        )


if __name__ == "__main__":
    main()
