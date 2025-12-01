# src/data_loader.py
import numpy as np
import pandas as pd
from pathlib import Path


def _load_input_file(input_path: Path) -> pd.DataFrame:
    """
    Internal helper: load an STS.input.* file.
    Supports:
      - 3 columns: id, s1, s2  (id is ignored)
      - 2 columns: s1, s2
    Returns a DataFrame with columns ['s1', 's2'].
    """
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
        raise ValueError(f"Unexpected number of columns in {input_path}: {df.shape[1]}")

    return df


def load_sts_split(input_path: str, gs_path: str, source_name: str) -> pd.DataFrame:
    """
    Load one STS dataset (e.g. MSRpar) and return a DataFrame with:
      ['source', 's1', 's2', 'score']

    Works whether the input file has 2 or 3 columns.
    """
    input_path = Path(input_path)
    gs_path = Path(gs_path)

    df_in = _load_input_file(input_path)

    # gold scores: one float per line
    scores = pd.read_csv(
        gs_path,
        sep="\t",
        header=None,
        names=["score"],
        encoding="utf-8"
    )

    assert len(df_in) == len(scores), f"Input and gold size mismatch for {source_name}!"

    df = pd.concat([df_in, scores], axis=1)
    df["source"] = source_name
    return df[["source", "s1", "s2", "score"]]


def load_sts_train(root: str) -> pd.DataFrame:
    """
    Load all three training datasets (MSRpar, MSRvid, SMTeuroparl) and concatenate.
    'root' is the folder that contains STS.input.* and STS.gs.* files.
    """
    root = Path(root)

    dfs = []
    dfs.append(load_sts_split(root / "STS.input.MSRpar.txt",
                              root / "STS.gs.MSRpar.txt",
                              "MSRpar"))
    dfs.append(load_sts_split(root / "STS.input.MSRvid.txt",
                              root / "STS.gs.MSRvid.txt",
                              "MSRvid"))
    dfs.append(load_sts_split(root / "STS.input.SMTeuroparl.txt",
                              root / "STS.gs.SMTeuroparl.txt",
                              "SMTeuroparl"))

    all_df = pd.concat(dfs, ignore_index=True)
    return all_df


def load_sts_test_file(input_path: str, source_name: str) -> pd.DataFrame:
    """
    Load a single STS.test input file (no gold scores).
    It may have either 2 or 3 columns:
      - 3: id, s1, s2
      - 2: s1, s2   (we'll create synthetic ids)
    Returns columns: ['source', 'id', 's1', 's2'].
    """
    input_path = Path(input_path)
    df = pd.read_csv(input_path, sep="\t", header=None,
                     encoding="utf-8", quoting=3)

    if df.shape[1] == 3:
        df.columns = ["id", "s1", "s2"]
    elif df.shape[1] == 2:
        df.columns = ["s1", "s2"]
        df["id"] = np.arange(len(df))
    else:
        raise ValueError(f"Unexpected number of columns in {input_path}: {df.shape[1]}")

    df["source"] = source_name
    return df[["source", "id", "s1", "s2"]]


def load_sts_test(root: str) -> pd.DataFrame:
    """
    Load and concatenate all test sets we care about
    (e.g. MSRpar, MSRvid, SMTeuroparl; plus surprise sets if needed).
    """
    root = Path(root)
    dfs = []
    dfs.append(load_sts_test_file(root / "STS.input.MSRpar.txt", "MSRpar"))
    dfs.append(load_sts_test_file(root / "STS.input.MSRvid.txt", "MSRvid"))
    dfs.append(load_sts_test_file(root / "STS.input.SMTeuroparl.txt", "SMTeuroparl"))
    # add surprise sets if your project wants them

    return pd.concat(dfs, ignore_index=True)


def load_sts_test_with_gs(root: str) -> pd.DataFrame:
    """
    If your 'test' folder ALSO has STS.gs.*.txt files and you are allowed
    to see them, this loads test inputs + gold scores, same format as train.
    """
    root = Path(root)
    dfs = []
    dfs.append(load_sts_split(root / "STS.input.MSRpar.txt",
                              root / "STS.gs.MSRpar.txt",
                              "MSRpar"))
    dfs.append(load_sts_split(root / "STS.input.MSRvid.txt",
                              root / "STS.gs.MSRvid.txt",
                              "MSRvid"))
    dfs.append(load_sts_split(root / "STS.input.SMTeuroparl.txt",
                              root / "STS.gs.SMTeuroparl.txt",
                              "SMTeuroparl"))

    return pd.concat(dfs, ignore_index=True)
