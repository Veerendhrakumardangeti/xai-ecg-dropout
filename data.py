"""
data.py
=======
Multi-dataset data loaders for ECG classification.

Supported datasets:
  - PTB-XL         (21,837 ECGs, European, 5 diagnostic superclasses)
  - CPSC 2018      (6,877  ECGs, Chinese,  9 rhythm classes)
  - Chapman-Shaoxing (10,646 ECGs, Chinese clinical, 11 classes)

All datasets are unified to an 8-class label space:
  0 = NORM   (normal sinus rhythm)
  1 = AF     (atrial fibrillation / flutter)
  2 = LBBB   (left bundle branch block)
  3 = RBBB   (right bundle branch block)
  4 = ST     (ST / T-wave changes, MI, ischemia)
  5 = AVB    (AV block, 1st / 2nd / 3rd degree)
  6 = PVC    (premature ventricular / supraventricular contractions)
  7 = OTHER  (all remaining conditions)

Public API:
  - ECGDataset         : torch Dataset (returns meta, ecg, label, idx)
  - load_ptbxl(...)    : returns (train_loader, val_loader, test_loader, cls_num_list)
  - load_cpsc2018(...) : same signature
  - load_chapman(...)  : same signature
  - load_dataset(name, data_path, sampling_rate, batch_size) : unified dispatcher
  - UNIFIED_CLASS_NAMES, NUM_CLASSES
"""

import os
import ast
import glob
import numpy as np
import pandas as pd
import wfdb
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────────────────────────────────────────
# Label spaces
# ─────────────────────────────────────────────────────────────────────────────

# PTB-XL native 5 superclasses
PTBXL_CLASS_NAMES  = ['NORM', 'MI', 'STTC', 'CD', 'HYP']

# Unified 8-class space for CPSC 2018 and Chapman
UNIFIED_CLASS_NAMES = ['NORM', 'AF', 'LBBB', 'RBBB', 'ST', 'AVB', 'PVC', 'OTHER']

# Default (kept for backward compat with CPSC/Chapman paths)
NUM_CLASSES = len(UNIFIED_CLASS_NAMES)   # 8


def get_class_info(dataset):
    """Return (num_classes, class_names) for the given dataset."""
    if dataset == 'ptbxl':
        return len(PTBXL_CLASS_NAMES), PTBXL_CLASS_NAMES
    return len(UNIFIED_CLASS_NAMES), UNIFIED_CLASS_NAMES


# ─────────────────────────────────────────────────────────────────────────────
# Shared ECGDataset — same structure for all three datasets
# ─────────────────────────────────────────────────────────────────────────────

class ECGDataset(Dataset):
    """
    Universal ECG dataset.

    Expects a DataFrame with columns:
      - 'filepath'  : str, WFDB record path (no extension)
      - 'label'     : int, unified 8-class label (0–7)
      - meta columns: 'age', 'sex' (minimum required)
        optionally: 'height', 'weight', 'infarction_stadium1',
                    'infarction_stadium2', 'pacemaker'

    Returns (meta_tensor, ecg_tensor, label_tensor, index).
    ecg_tensor shape: (12, L, 1)  where L = number of timesteps (usually 1000).
    """

    def __init__(self, df, x_scaler=None, target_length=1000):
        self.df = df.reset_index(drop=True)
        self.target_length = target_length
        self.labels = torch.tensor(df['label'].values, dtype=torch.long)

        # ── meta features ───────────────────────────────────────────────────
        meta_cols = ['age', 'sex']
        for col in ['height', 'weight', 'infarction_stadium1',
                    'infarction_stadium2', 'pacemaker']:
            if col in df.columns:
                meta_cols.append(col)

        self.X = df[meta_cols].fillna(0).astype(float)
        if x_scaler is not None:
            self.X = pd.DataFrame(
                x_scaler.transform(self.X), columns=self.X.columns
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # read ECG from WFDB
        signal, _ = wfdb.rdsamp(row['filepath'])   # (L, 12)
        signal = signal.T                            # (12, L)

        # pad / truncate to target_length
        L = signal.shape[1]
        if L < self.target_length:
            signal = np.pad(signal, ((0, 0), (0, self.target_length - L)))
        else:
            signal = signal[:, :self.target_length]

        # per-sample per-lead standardisation (handles high-gain datasets like Georgia)
        signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
        std = signal.std(axis=1, keepdims=True)
        std[std < 1e-8] = 1.0
        signal = (signal - signal.mean(axis=1, keepdims=True)) / std

        signal = np.expand_dims(signal, axis=-1)    # (12, L, 1)

        meta  = torch.tensor(self.X.iloc[idx].values, dtype=torch.float32)
        ecg   = torch.tensor(signal, dtype=torch.float32)
        label = self.labels[idx]
        return meta, ecg, label, idx


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_loaders(train_df, val_df, test_df, batch_size, target_length=1000, num_classes=NUM_CLASSES):
    """Fits a StandardScaler on train meta, returns DataLoaders + cls_num_list."""
    meta_cols = ['age', 'sex']
    for col in ['height', 'weight', 'infarction_stadium1',
                'infarction_stadium2', 'pacemaker']:
        if col in train_df.columns:
            meta_cols.append(col)

    x_scaler = StandardScaler().fit(train_df[meta_cols].fillna(0).astype(float))

    train_ds = ECGDataset(train_df, x_scaler, target_length)
    val_ds   = ECGDataset(val_df,   x_scaler, target_length)
    test_ds  = ECGDataset(test_df,  x_scaler, target_length)

    import torch
    num_workers = min(4, os.cpu_count() or 1)
    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin)

    cls_num_list = np.bincount(train_df['label'].values,
                               minlength=num_classes).astype(np.float64)
    return train_loader, val_loader, test_loader, cls_num_list


# ─────────────────────────────────────────────────────────────────────────────
# PTB-XL
# ─────────────────────────────────────────────────────────────────────────────
# Official stratified folds 1-8 train / 9 val / 10 test.
# Uses PTB-XL native 5 diagnostic superclasses from scp_statements.csv:
#   NORM(0), MI(1), STTC(2), CD(3), HYP(4)
# ─────────────────────────────────────────────────────────────────────────────

_PTBXL_SUPERCLASS_TO_IDX = {c: i for i, c in enumerate(PTBXL_CLASS_NAMES)}


def _ptbxl_native_label(scp_dict, scp_df):
    """Map PTB-XL scp_codes dict → native 5-class label using diagnostic_superclass."""
    best_idx = None
    best_conf = -1
    for code, conf in scp_dict.items():
        if code in scp_df.index:
            sc = scp_df.loc[code, 'diagnostic_superclass'] \
                 if 'diagnostic_superclass' in scp_df.columns \
                 else scp_df.loc[code, 'diagnostic_class'] \
                 if 'diagnostic_class' in scp_df.columns else None
            if sc in _PTBXL_SUPERCLASS_TO_IDX and conf > best_conf:
                best_conf = conf
                best_idx = _PTBXL_SUPERCLASS_TO_IDX[sc]
    return best_idx if best_idx is not None else 0  # default NORM


def load_ptbxl(data_path, sampling_rate=100, batch_size=32):
    """
    Load PTB-XL dataset.

    Directory layout expected:
        <data_path>/ptbxl_database.csv
        <data_path>/scp_statements.csv
        <data_path>/records100/...  (if sampling_rate=100)
        <data_path>/records500/...  (if sampling_rate=500)
    """
    ecg_df = pd.read_csv(os.path.join(data_path, 'ptbxl_database.csv'),
                         index_col='ecg_id')
    ecg_df.scp_codes = ecg_df.scp_codes.apply(ast.literal_eval)

    scp_df = pd.read_csv(os.path.join(data_path, 'scp_statements.csv'), index_col=0)
    scp_df = scp_df[scp_df.diagnostic == 1]

    # unified labels
    ecg_df['label'] = ecg_df.scp_codes.apply(
        lambda d: _ptbxl_native_label(d, scp_df)
    )

    # file paths
    rate_dir = 'records100' if sampling_rate == 100 else 'records500'
    col = 'filename_lr' if sampling_rate == 100 else 'filename_hr'
    ecg_df['filepath'] = ecg_df[col].apply(
        lambda f: os.path.join(data_path, f)
    )

    # meta columns
    ecg_df['age']    = ecg_df.age.fillna(0)
    ecg_df['sex']    = ecg_df.sex.astype(float).fillna(0)
    ecg_df['height'] = ecg_df.height.where(ecg_df.height >= 50, np.nan).fillna(0)
    ecg_df['weight'] = ecg_df.weight.fillna(0)
    ecg_df['infarction_stadium1'] = ecg_df.infarction_stadium1.map(
        {'unknown': 0, 'Stadium I': 1, 'Stadium I-II': 2,
         'Stadium II': 3, 'Stadium II-III': 4, 'Stadium III': 5}
    ).fillna(0)
    ecg_df['infarction_stadium2'] = ecg_df.infarction_stadium2.map(
        {'unknown': 0, 'Stadium I': 1, 'Stadium II': 2, 'Stadium III': 3}
    ).fillna(0)
    ecg_df['pacemaker'] = (ecg_df.pacemaker == 'ja, pacemaker').astype(float)

    train_df = ecg_df[ecg_df.strat_fold <= 8].reset_index(drop=True)
    val_df   = ecg_df[ecg_df.strat_fold == 9].reset_index(drop=True)
    test_df  = ecg_df[ecg_df.strat_fold == 10].reset_index(drop=True)

    target_length = 1000 if sampling_rate == 100 else 5000
    return _make_loaders(train_df, val_df, test_df, batch_size, target_length,
                         num_classes=len(PTBXL_CLASS_NAMES))


# ─────────────────────────────────────────────────────────────────────────────
# CPSC 2018
# ─────────────────────────────────────────────────────────────────────────────
# Supports two layouts:
#
# Layout A — original CPSC 2018 release:
#   <data_path>/REFERENCE.csv      (record, label 1-9)
#   <data_path>/TrainingSet/*.mat
#
# Layout B — PhysioNet Challenge 2020 subset (auto-detected):
#   <data_path>/g1/*.hea  <data_path>/g2/*.hea  ...
#   Labels embedded in .hea as  #Dx: <SNOMED-CT codes>
#
# Original 9 labels (Layout A numeric / Layout B SNOMED):
#   1/426783006 = Normal, 2/164889003 = AF,   3/270492004 = 1AVB,
#   4/164909002 = LBBB,   5/59118001  = RBBB,
#   6/284470004|63593006 = PAC,  7/427172004|17338001 = PVC,
#   8/429622005 = STD,    9/164931005 = STE
# ─────────────────────────────────────────────────────────────────────────────

_CPSC_TO_UNIFIED = {
    1: 0,   # Normal  → NORM
    2: 1,   # AF      → AF
    3: 5,   # 1AVB    → AVB
    4: 2,   # LBBB    → LBBB
    5: 3,   # RBBB    → RBBB
    6: 6,   # PAC     → PVC/PAC
    7: 6,   # PVC     → PVC/PAC
    8: 4,   # STD     → ST
    9: 4,   # STE     → ST
}

# SNOMED-CT code → unified label (Challenge 2020 layout)
_SNOMED_TO_UNIFIED = {
    '426783006': 0,  # Normal sinus rhythm → NORM
    '164889003': 1,  # AF → AF
    '164890007': 1,  # AF flutter → AF
    '270492004': 5,  # 1st degree AVB → AVB
    '195042002': 5,  # 2nd degree AVB → AVB
    '27885002':  5,  # 3rd degree AVB → AVB
    '164909002': 2,  # LBBB → LBBB
    '59118001':  3,  # RBBB → RBBB
    '713427006': 3,  # Incomplete RBBB → RBBB
    '284470004': 6,  # PAC → PVC/PAC
    '63593006':  6,  # PAC (alt code) → PVC/PAC
    '427172004': 6,  # PVC → PVC/PAC
    '17338001':  6,  # PVC (alt code) → PVC/PAC
    '429622005': 4,  # ST depression → ST
    '164931005': 4,  # ST elevation → ST
}


def _parse_hea_dx(hea_path):
    """Return (unified_label, age, sex) parsed from a WFDB .hea file."""
    label = 7   # OTHER default
    age   = 0
    sex   = 0
    with open(hea_path, 'r') as f:
        for line in f:
            # normalise: "# Dx:" → "#Dx:"
            line = line.strip().replace('# ', '#', 1)
            if line.startswith('#Dx:'):
                codes = [c.strip() for c in line.split(':', 1)[1].split(',')]
                for code in codes:
                    if code in _SNOMED_TO_UNIFIED:
                        label = _SNOMED_TO_UNIFIED[code]
                        break   # take first matching code
            elif line.startswith('#Age:'):
                try:
                    age = float(line.split(':', 1)[1].strip())
                except ValueError:
                    age = 0
            elif line.startswith('#Sex:'):
                val = line.split(':', 1)[1].strip().upper()
                sex = 1 if val in ('F', 'FEMALE') else 0
    return label, age, sex


def load_cpsc2018(data_path, sampling_rate=500, batch_size=32):
    """
    Load CPSC 2018 dataset.

    sampling_rate is ignored (CPSC is always 500 Hz); ECG is padded/truncated
    to 1000 timesteps (= 2 s, matching PTB-XL 100 Hz × 10 s).

    Auto-detects layout:
      Layout A: <data_path>/REFERENCE.csv + TrainingSet/*.mat
      Layout B: <data_path>/g*/*.hea  (PhysioNet Challenge 2020 subset)
    """
    from sklearn.model_selection import train_test_split

    ref_path = os.path.join(data_path, 'REFERENCE.csv')

    if os.path.exists(ref_path):
        # ── Layout A ──────────────────────────────────────────────────────────
        ref = pd.read_csv(ref_path, header=None, names=['record', 'label'])
        records_dir = os.path.join(data_path, 'TrainingSet')
        rows = []
        for _, row in ref.iterrows():
            record_name = str(row['record']).strip()
            fp = os.path.join(records_dir, record_name)
            unified = _CPSC_TO_UNIFIED.get(int(row['label']), 7)
            rows.append({'filepath': fp, 'label': unified, 'age': 0.0, 'sex': 0.0})

    else:
        # ── Layout B (Challenge 2020) ─────────────────────────────────────────
        hea_files = sorted(glob.glob(os.path.join(data_path, 'g*', '*.hea')))
        if not hea_files:
            raise FileNotFoundError(
                f"No REFERENCE.csv and no g*/*.hea files found in {data_path}"
            )
        rows = []
        for hea in hea_files:
            record_path = hea[:-4]   # strip .hea → WFDB record stem
            unified, age, sex = _parse_hea_dx(hea)
            rows.append({'filepath': record_path, 'label': unified,
                         'age': age, 'sex': sex})

    df = pd.DataFrame(rows)
    print(f"[CPSC2018] {len(df)} records loaded. "
          f"Label dist: {dict(zip(*np.unique(df['label'], return_counts=True)))}")

    # 80% train / 10% val / 10% test  (no official fold — random split)
    train_df, tmp = train_test_split(df, test_size=0.2, random_state=42,
                                     stratify=df['label'])
    val_df, test_df = train_test_split(tmp, test_size=0.5, random_state=42,
                                       stratify=tmp['label'])

    train_df = train_df.reset_index(drop=True)
    val_df   = val_df.reset_index(drop=True)
    test_df  = test_df.reset_index(drop=True)

    return _make_loaders(train_df, val_df, test_df, batch_size, target_length=1000)


# ─────────────────────────────────────────────────────────────────────────────
# Chapman-Shaoxing
# ─────────────────────────────────────────────────────────────────────────────
# Download: https://physionet.org/content/ecg-arrhythmia/1.0.0/
#
# Directory layout:
#   <data_path>/Diagnostics.csv         (FileName, Rhythm, Beat)
#   <data_path>/ECGDataDenoised/*.csv   (one CSV per record, 5000 rows × 12 cols)
#   ── OR ──
#   <data_path>/WFDBRecords/**/*.hea    (WFDB format)
#
# Chapman has 11 rhythm categories; we map to the unified 8-class space.
# ─────────────────────────────────────────────────────────────────────────────

# Chapman rhythm label string → unified label
_CHAPMAN_TO_UNIFIED = {
    # Normal
    'SR':    0,   # Sinus Rhythm
    'AFIB':  1,   # Atrial Fibrillation
    'AFL':   1,   # Atrial Flutter
    'LBBB':  2,
    'RBBB':  3,
    'IRBBB': 3,   # Incomplete RBBB
    'ST':    4,   # ST-elevation/depression
    'AVB':   5,
    'AVNRT': 6,   # AV nodal re-entrant tachycardia (PVC group)
    'SVT':   6,   # Supraventricular tachycardia
    'AT':    6,   # Atrial tachycardia
    'SAAWR': 7,   # Sinus arrhythmia / other
    'SI':    0,   # Sinus irregularity (treat as NORM)
    'SB':    0,   # Sinus bradycardia (treat as NORM)
    'ST_':   4,   # ST changes
}


def load_chapman(data_path, sampling_rate=500, batch_size=32):
    """
    Load Chapman-Shaoxing dataset.

    Two supported formats are auto-detected:

    Format A — WFDB records (preferred):
        <data_path>/WFDBRecords/**/*.hea
        <data_path>/Diagnostics.csv   (FileName, Rhythm, PatientAge, Gender)

    Format B — CSV records:
        <data_path>/ECGDataDenoised/<ecg_id>.csv  (5000 rows × 12 columns)
        <data_path>/Diagnostics.csv

    sampling_rate is informational only; ECG is always padded/truncated to 1000 samples.
    """
    diag_path = os.path.join(data_path, 'Diagnostics.csv')
    diag = pd.read_csv(diag_path)

    # Normalise column names (dataset has inconsistent capitalisation)
    diag.columns = [c.strip() for c in diag.columns]
    diag = diag.rename(columns={
        'FileName': 'record', 'Rhythm': 'rhythm',
        'PatientAge': 'age', 'Gender': 'sex', 'Beat': 'beat'
    })

    # Sex → numeric
    diag['sex'] = diag['sex'].map({'MALE': 0, 'FEMALE': 1,
                                   'M': 0, 'F': 1}).fillna(0)
    diag['age'] = pd.to_numeric(diag['age'], errors='coerce').fillna(0)

    # Detect record format
    wfdb_dir = os.path.join(data_path, 'WFDBRecords')
    csv_dir  = os.path.join(data_path, 'ECGDataDenoised')
    use_wfdb = os.path.isdir(wfdb_dir)

    rows = []
    for _, row in diag.iterrows():
        record_name = str(row['record']).strip()
        rhythm = str(row.get('rhythm', '')).strip().upper()
        unified = _CHAPMAN_TO_UNIFIED.get(rhythm, 7)

        if use_wfdb:
            # Find the .hea file recursively
            matches = glob.glob(
                os.path.join(wfdb_dir, '**', record_name), recursive=True
            )
            if not matches:
                continue
            fp = matches[0].replace('.hea', '')
        else:
            fp = os.path.join(csv_dir, record_name)
            if not os.path.exists(fp + '.csv') and not os.path.exists(fp):
                continue

        rows.append({
            'filepath': fp,
            'label':    unified,
            'age':      row['age'],
            'sex':      row['sex'],
            '_format':  'wfdb' if use_wfdb else 'csv',
        })

    df = pd.DataFrame(rows)

    # 80/10/10 stratified split
    from sklearn.model_selection import train_test_split
    train_df, tmp = train_test_split(df, test_size=0.2, random_state=42,
                                     stratify=df['label'])
    val_df, test_df = train_test_split(tmp, test_size=0.5, random_state=42,
                                       stratify=tmp['label'])

    train_df = train_df.reset_index(drop=True)
    val_df   = val_df.reset_index(drop=True)
    test_df  = test_df.reset_index(drop=True)

    if not use_wfdb:
        # Override ECGDataset to load CSV files
        return _make_loaders_csv_chapman(train_df, val_df, test_df, batch_size)

    return _make_loaders(train_df, val_df, test_df, batch_size, target_length=1000)


class _ChapmanCSVDataset(Dataset):
    """Dataset for Chapman-Shaoxing CSV format (ECGDataDenoised/*.csv)."""

    def __init__(self, df, x_scaler=None, target_length=1000):
        self.df = df.reset_index(drop=True)
        self.target_length = target_length
        self.labels = torch.tensor(df['label'].values, dtype=torch.long)

        meta_cols = ['age', 'sex']
        self.X = df[meta_cols].fillna(0).astype(float)
        if x_scaler is not None:
            self.X = pd.DataFrame(
                x_scaler.transform(self.X), columns=self.X.columns
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fp = row['filepath']
        csv_fp = fp if fp.endswith('.csv') else fp + '.csv'

        ecg_df = pd.read_csv(csv_fp, header=None)   # (5000, 12)
        signal = ecg_df.values.T.astype(np.float32)  # (12, 5000)

        # Truncate / pad to target_length
        L = signal.shape[1]
        if L < self.target_length:
            signal = np.pad(signal, ((0, 0), (0, self.target_length - L)))
        else:
            signal = signal[:, :self.target_length]

        signal = np.expand_dims(signal, axis=-1)      # (12, L, 1)

        meta  = torch.tensor(self.X.iloc[idx].values, dtype=torch.float32)
        ecg   = torch.tensor(signal, dtype=torch.float32)
        return meta, ecg, self.labels[idx], idx


def _make_loaders_csv_chapman(train_df, val_df, test_df, batch_size):
    """Equivalent to _make_loaders but uses _ChapmanCSVDataset."""
    from sklearn.preprocessing import StandardScaler
    x_scaler = StandardScaler().fit(
        train_df[['age', 'sex']].fillna(0).astype(float)
    )
    num_workers = min(4, os.cpu_count() or 1)

    def _dl(df, shuffle):
        ds = _ChapmanCSVDataset(df, x_scaler)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=num_workers, pin_memory=True)

    cls_num_list = np.bincount(train_df['label'].values,
                               minlength=NUM_CLASSES).astype(np.float64)
    return _dl(train_df, True), _dl(val_df, False), _dl(test_df, False), cls_num_list


# ─────────────────────────────────────────────────────────────────────────────
# Chapman-Shaoxing v2  (45,152 ECGs, WFDB + SNOMED-CT)
# ─────────────────────────────────────────────────────────────────────────────
# Download: https://physionet.org/content/ecg-arrhythmia/1.0.0/
#
# Directory layout after unzip:
#   <data_path>/WFDBRecords/<xx>/<xxx>/<JS#####>.hea
#   <data_path>/WFDBRecords/<xx>/<xxx>/<JS#####>.mat
#
# Labels are SNOMED-CT codes in .hea  — same format as CPSC 2018 Layout B,
# so we reuse _parse_hea_dx() and _SNOMED_TO_UNIFIED directly.
# ─────────────────────────────────────────────────────────────────────────────

def load_chapman_v2(data_path, sampling_rate=500, batch_size=32):
    """
    Load Chapman-Shaoxing v2 (ecg-arrhythmia 1.0.0, 45,152 records).

    Expects WFDB layout:
        <data_path>/WFDBRecords/**/*.hea
    """
    from sklearn.model_selection import train_test_split

    hea_files = sorted(glob.glob(
        os.path.join(data_path, 'WFDBRecords', '**', '*.hea'), recursive=True
    ))
    if not hea_files:
        raise FileNotFoundError(
            f"No WFDBRecords/**/*.hea files found in {data_path}"
        )

    rows = []
    for hea in hea_files:
        record_path = hea[:-4]   # strip .hea → WFDB record stem
        unified, age, sex = _parse_hea_dx(hea)
        rows.append({'filepath': record_path, 'label': unified,
                     'age': age, 'sex': sex})

    df = pd.DataFrame(rows)
    print(f"[ChapmanV2] {len(df)} records loaded. "
          f"Label dist: {dict(zip(*np.unique(df['label'], return_counts=True)))}")

    train_df, tmp = train_test_split(df, test_size=0.2, random_state=42,
                                     stratify=df['label'])
    val_df, test_df = train_test_split(tmp, test_size=0.5, random_state=42,
                                       stratify=tmp['label'])

    train_df = train_df.reset_index(drop=True)
    val_df   = val_df.reset_index(drop=True)
    test_df  = test_df.reset_index(drop=True)

    return _make_loaders(train_df, val_df, test_df, batch_size, target_length=5000)


# ─────────────────────────────────────────────────────────────────────────────
# Georgia 12-Lead ECG Challenge Database  (~10,344 ECGs, WFDB + SNOMED-CT)
# ─────────────────────────────────────────────────────────────────────────────
# Directory layout:
#   <data_path>/Georgia/*.hea
#   <data_path>/Georgia/*.mat
# Labels are SNOMED-CT codes in .hea — reuses _parse_hea_dx() directly.
# ─────────────────────────────────────────────────────────────────────────────

def load_georgia(data_path, sampling_rate=500, batch_size=32):
    """
    Load Georgia 12-Lead ECG Challenge dataset (~10,344 records).

    Expects WFDB layout:
        <data_path>/Georgia/*.hea  (flat directory)
    """
    from sklearn.model_selection import train_test_split

    hea_files = sorted(glob.glob(os.path.join(data_path, 'Georgia', '*.hea')))
    if not hea_files:
        # try flat layout
        hea_files = sorted(glob.glob(os.path.join(data_path, '*.hea')))
    if not hea_files:
        raise FileNotFoundError(f"No *.hea files found in {data_path}")

    rows = []
    for hea in hea_files:
        record_path = hea[:-4]
        unified, age, sex = _parse_hea_dx(hea)
        rows.append({'filepath': record_path, 'label': unified,
                     'age': age, 'sex': sex})

    df = pd.DataFrame(rows)
    print(f"[Georgia] {len(df)} records loaded. "
          f"Label dist: {dict(zip(*np.unique(df['label'], return_counts=True)))}")

    train_df, tmp = train_test_split(df, test_size=0.2, random_state=42,
                                     stratify=df['label'])
    val_df, test_df = train_test_split(tmp, test_size=0.5, random_state=42,
                                       stratify=tmp['label'])

    train_df = train_df.reset_index(drop=True)
    val_df   = val_df.reset_index(drop=True)
    test_df  = test_df.reset_index(drop=True)

    return _make_loaders(train_df, val_df, test_df, batch_size, target_length=5000)


# ─────────────────────────────────────────────────────────────────────────────
# Unified dispatcher
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset(name, data_path, sampling_rate=100, batch_size=32):
    """
    Unified dataset loader.

    Args:
        name          : 'ptbxl' | 'cpsc2018' | 'chapman'
        data_path     : path to the dataset root directory
        sampling_rate : for PTB-XL: 100 or 500  (ignored for others)
        batch_size    : DataLoader batch size

    Returns:
        (train_loader, val_loader, test_loader, cls_num_list)
    """
    loaders = {
        'ptbxl':       load_ptbxl,
        'cpsc2018':    load_cpsc2018,
        'chapman':     load_chapman,
        'chapman_v2':  load_chapman_v2,
        'georgia':     load_georgia,
    }
    if name not in loaders:
        raise ValueError(
            f"Unknown dataset '{name}'. Choose from: {list(loaders.keys())}"
        )
    return loaders[name](data_path, sampling_rate, batch_size)



