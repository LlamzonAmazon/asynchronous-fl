"""
PTB-XL Dataset Loader for Federated Learning Thesis
Loads and preprocesses ECG data from PhysioNet PTB-XL dataset
"""

import os
import pandas as pd
import numpy as np
import wfdb
from pathlib import Path
from typing import Tuple, List
import ast

class PTBXLDataset:
    """Load and preprocess PTB-XL ECG dataset"""
    
    def __init__(self, data_path: str = './PTB-XL'):
        """
        Args:
            data_path: Path to PTB-XL dataset directory
        """
        self.data_path = Path(data_path)
        self.sampling_rate = 100  # Use 100Hz version for efficiency
        
        if not self.data_path.exists():
            print(f"Dataset not found at {self.data_path}")
            print("Download from: https://physionet.org/content/ptb-xl/1.0.3/")
            print("Or run: wget -r -N -c -np https://physionet.org/files/ptb-xl/1.0.3/")
    
    def load_raw_data(self) -> pd.DataFrame:
        """Load the metadata CSV file"""
        metadata_path = self.data_path / 'ptbxl_database.csv'
        print(f"Loading metadata from {metadata_path}")
        
        df = pd.read_csv(metadata_path, index_col='ecg_id')
        df.scp_codes = df.scp_codes.apply(lambda x: ast.literal_eval(x))
        
        print(f"Loaded {len(df)} ECG records")
        return df
    
    def load_signal(self, filename: str) -> np.ndarray:
        """Load a single ECG signal"""
        record_path = self.data_path / filename
        record = wfdb.rdsamp(str(record_path.with_suffix('')))
        signal = record[0]  # Extract signal data
        return signal
    
    def aggregate_diagnostic_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert detailed SCP codes to superclass labels"""
        # Load SCP statement definitions
        agg_df = pd.read_csv(self.data_path / 'scp_statements.csv', index_col=0)
        agg_df = agg_df[agg_df.diagnostic == 1]
        
        def aggregate_diagnostic(y_dic):
            tmp = []
            for key in y_dic.keys():
                if key in agg_df.index:
                    tmp.append(agg_df.loc[key].diagnostic_class)
            return list(set(tmp))
        
        df['diagnostic_superclass'] = df.scp_codes.apply(aggregate_diagnostic)
        return df
    
    def prepare_data(self, 
                     superclass: str = 'NORM',
                     test_fold: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare train/test data for a specific diagnostic class
        
        Args:
            superclass: Diagnostic superclass to classify (NORM, MI, STTC, CD, HYP)
            test_fold: Which fold to use for testing (10 is standard test set)
        
        Returns:
            X_train, y_train, X_test, y_test
        """
        print(f"\nPreparing data for superclass: {superclass}")
        
        # Load metadata
        df = self.load_raw_data()
        df = self.aggregate_diagnostic_labels(df)
        
        # Create binary labels
        df['target'] = df.diagnostic_superclass.apply(lambda x: 1 if superclass in x else 0)
        
        # Split by standard folds
        train_df = df[df.strat_fold != test_fold]
        test_df = df[df.strat_fold == test_fold]
        
        print(f"Train samples: {len(train_df)} (positive: {train_df.target.sum()})")
        print(f"Test samples: {len(test_df)} (positive: {test_df.target.sum()})")
        
        # Load signals
        print("Loading training signals...")
        X_train = np.array([self.load_signal(f) for f in train_df.filename_hr])
        y_train = train_df.target.values
        
        print("Loading test signals...")
        X_test = np.array([self.load_signal(f) for f in test_df.filename_hr])
        y_test = test_df.target.values
        
        print(f"Signal shape: {X_train.shape}")
        return X_train, y_train, X_test, y_test
    
    def get_available_superclasses(self) -> List[str]:
        """Get list of available diagnostic superclasses"""
        df = self.load_raw_data()
        df = self.aggregate_diagnostic_labels(df)
        
        all_classes = set()
        for classes in df.diagnostic_superclass:
            all_classes.update(classes)
        
        return sorted(list(all_classes))


def test_data_loading():
    """Test the data loader"""
    print("=" * 60)
    print("PTB-XL Data Loader Test")
    print("=" * 60)
    
    # Initialize dataset
    dataset = PTBXLDataset(data_path='./PTB-XL')
    
    # Show available classes
    print("\nAvailable diagnostic superclasses:")
    classes = dataset.get_available_superclasses()
    for c in classes:
        print(f"  - {c}")
    
    # Load a small subset for testing
    print("\n" + "=" * 60)
    print("Loading NORM (normal) classification data...")
    print("=" * 60)
    
    X_train, y_train, X_test, y_test = dataset.prepare_data(
        superclass='NORM',
        test_fold=10
    )
    
    print("\n" + "=" * 60)
    print("Data Statistics:")
    print("=" * 60)
    print(f"Training set: {X_train.shape}")
    print(f"  - Signal length: {X_train.shape[1]} samples")
    print(f"  - Number of leads: {X_train.shape[2]} leads")
    print(f"  - Class balance: {y_train.sum()} positive / {len(y_train) - y_train.sum()} negative")
    print(f"\nTest set: {X_test.shape}")
    print(f"  - Class balance: {y_test.sum()} positive / {len(y_test) - y_test.sum()} negative")
    
    print("\n" + "=" * 60)
    print("DATA LOADED SUCCESSFULLY.")
    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    # Run test
    X_train, y_train, X_test, y_test = test_data_loading()