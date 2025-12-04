# src/data_loader.py
import numpy as np
import pandas as pd
from pathlib import Path
import os

def _load_input_file(input_path: Path) -> pd.DataFrame:
    """
    Internal helper: load an STS.input.* file.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(
        input_path,
        sep="\t",
        header=None,
        encoding="utf-8",
        quoting=3
    )

    if df.shape[1] == 3:
        df.columns = ["id", "s1", "s2"]
        df = df[["s1", "s2"]]
    elif df.shape[1] == 2:
        df.columns = ["s1", "s2"]
    else:
        # Fallback for weird files: take first 2 text cols
        df = df.iloc[:, :2]
        df.columns = ["s1", "s2"]

    return df

def _find_file(root: Path, pattern_list: list) -> Path:
    """
    Helper to try multiple filename patterns (e.g. standard vs surprise).
    """
    for pat in pattern_list:
        candidate = root / pat
        if candidate.exists():
            return candidate
    # If none found, return the first pattern so the error message makes sense
    return root / pattern_list[0]

def load_sts_split(input_path: Path, gs_path: Path, source_name: str) -> pd.DataFrame:
    """
    Load one STS dataset and return: ['source', 's1', 's2', 'score']
    """
    df_in = _load_input_file(input_path)

    if not gs_path.exists():
        raise FileNotFoundError(f"Gold Score file not found: {gs_path}")

    scores = pd.read_csv(
        gs_path,
        sep="\t",
        header=None,
        names=["score"],
        encoding="utf-8"
    )

    # Size mismatch fix (common in SemEval data)
    if len(df_in) != len(scores):
        print(f"Warning: Size mismatch for {source_name} (In: {len(df_in)}, GS: {len(scores)}) -- Truncating.")
        min_len = min(len(df_in), len(scores))
        df_in = df_in.iloc[:min_len]
        scores = scores.iloc[:min_len]

    df = pd.concat([df_in, scores], axis=1)
    df["source"] = source_name
    return df[["source", "s1", "s2", "score"]]

def load_sts_train(root: str) -> pd.DataFrame:
    """
    Load training datasets (Standard 3).
    """
    root = Path(root)
    dfs = []
    datasets = ["MSRpar", "MSRvid", "SMTeuroparl"]
    
    for ds in datasets:
        # Try both standard and surprise patterns just in case
        in_path = _find_file(root, [f"STS.input.{ds}.txt", f"STS.input.surprise.{ds}.txt"])
        gs_path = _find_file(root, [f"STS.gs.{ds}.txt", f"STS.gs.surprise.{ds}.txt"])
        
        if in_path.exists() and gs_path.exists():
            dfs.append(load_sts_split(in_path, gs_path, ds))
        else:
            print(f"Train set {ds} not found.")

    if not dfs:
        raise ValueError("No training data found in " + str(root))
        
    return pd.concat(dfs, ignore_index=True)

def load_sts_test(root: str) -> pd.DataFrame:
    """
    Load ALL test sets (Inputs only).
    Handles 'surprise' naming automatically.
    """
    root = Path(root)
    dfs = []
    datasets = ["MSRpar", "MSRvid", "SMTeuroparl", "OnWN", "SMTnews"]

    for ds in datasets:
        # Look for standard OR surprise filenames
        pat_list = [f"STS.input.{ds}.txt", f"STS.input.surprise.{ds}.txt"]
        target_path = _find_file(root, pat_list)

        if target_path.exists():
            df = _load_input_file(target_path)
            df["source"] = ds
            # Add synthetic ID if missing
            if "id" not in df.columns:
                df["id"] = np.arange(len(df))
            dfs.append(df[["source", "id", "s1", "s2"]])
        else:
            print(f"Test file for {ds} not found (checked {pat_list}).")

    return pd.concat(dfs, ignore_index=True)

def load_sts_test_with_gs(root: str) -> pd.DataFrame:
    """
    Load ALL test inputs + gold scores.
    """
    root = Path(root)
    dfs = []
    datasets = ["MSRpar", "MSRvid", "SMTeuroparl", "OnWN", "SMTnews"]

    for ds in datasets:
        # Check inputs
        in_pats = [f"STS.input.{ds}.txt", f"STS.input.surprise.{ds}.txt"]
        in_path = _find_file(root, in_pats)

        # Check gold scores
        gs_pats = [f"STS.gs.{ds}.txt", f"STS.gs.surprise.{ds}.txt"]
        gs_path = _find_file(root, gs_pats)

        if in_path.exists() and gs_path.exists():
            dfs.append(load_sts_split(in_path, gs_path, ds))
        else:
            print(f"Skipping {ds}: Files not found.")

    return pd.concat(dfs, ignore_index=True)