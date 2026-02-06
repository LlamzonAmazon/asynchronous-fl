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

class PTBXLDataLoader:
    """Load and preprocess PTB-XL ECG dataset"""
    
    def __init__(self, data_path: str = './PTB-XL'):
        """
        Args:
            data_path: Path to PTB-XL dataset directory
        """
        self.data_path = Path(data_path)
        self.sampling_rate = 100  # Using 100Hz version for efficiency
        
        if not self.data_path.exists():
            print(f"Dataset not found at {self.data_path}")
    
    def load_raw_data(self) -> pd.DataFrame:
        """Load metadata CSV"""

        metadata_path = self.data_path / 'ptbxl_database.csv'
        print(f"Loading metadata from {metadata_path}")
        
        df = pd.read_csv(metadata_path, index_col='ecg_id')
        df.scp_codes = df.scp_codes.apply(lambda x: ast.literal_eval(x))
        
        print(f"Loaded {len(df)} ECG records")
        return df
    
    def load_signal(self, filename: str) -> np.ndarray:
        """Load one ECG signal"""

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
        # 1 (positive): ECG has NORM (normal heart)
        # 0 (negative): ECG does not have NORM (abnormal heart)
        df['target'] = df.diagnostic_superclass.apply(lambda x: 1 if superclass in x else 0)
        
        # Split by standard folds
        train_df = df[df.strat_fold != test_fold] # Loads in fold 1-9
        test_df = df[df.strat_fold == test_fold] # Loads in fold 10
        
        print(f"Training samples: {len(train_df)} (positive: {train_df.target.sum()})")
        print(f"Test samples: {len(test_df)} (positive: {test_df.target.sum()})")
        
        # Load signals
        print("Loading training signals...")
        X_train = np.array([self.load_signal(f) for f in train_df.filename_hr])
        y_train = train_df.target.values
        
        print("Loading test signals...")
        X_test = np.array([self.load_signal(f) for f in test_df.filename_hr])
        y_test = test_df.target.values
        
        return X_train, y_train, X_test, y_test
    
    def get_available_superclasses(self) -> List[str]:
        """Get list of available diagnostic superclasses"""

        df = self.load_raw_data()
        df = self.aggregate_diagnostic_labels(df)
        
        all_classes = set()
        for classes in df.diagnostic_superclass:
            all_classes.update(classes)

        # NORM:     Normal ECG
        # MI:       Myocardial infarction (Abnormal heath repolarization patterns)
        # STTC:     ST-T wave abnormality (Abnormal ST segment and T wave on an ECG)
        # CD:       Cardiac dysrhythmia (Abnormal heart rhythm)
        # HYP:      Hypertrophy (Abnormal heart chamber thickness)
        
        return sorted(list(all_classes))


def main():
    print("=" * 60)
    print("PTB-XL Data Loader Test")
    print("=" * 60)
    
    # Initialize dataset
    dataset = PTBXLDataLoader(data_path='./PTB-XL')
    
    # Show available classes
    print("\nAvailable diagnostic superclasses:")
    classes = dataset.get_available_superclasses()
    for c in classes:
        print(f"  - {c}")
    
    # Load a small subset for testing
    print("\n" + "=" * 60)
    print("Loading NORM (normal) classification data")
    print("=" * 60)
    
    X_train, y_train, X_test, y_test = dataset.prepare_data(
        # Using NORM as the superclass
        superclass='NORM',

        # Using fold=10 as it is the standard train/test split by the dataset authors
        # Folds 1-9 are used for model training, fold 10 is used for model evaluation (testing)
        # Dataset is folded like this to support different experimental and training setups
        # Each fold has 44% NORM and 56% abnormal (non-NORM) labels 
        # GOOD class balance for classification training; each fold is representative of the dataset
        test_fold=10
    )
    
    print("\n" + "=" * 60)
    print("Data Statistics:")
    print("=" * 60)

    # Training set
    # X_train is a 3D array of shape (num_ecgs, signal_length, num_leads)
    # X_train is the ECG signals of the training set
    # y_train is the labels of the training set corresponding to each ECG signal in X_train
    print(f"Training set shape: {X_train.shape}")
    
    # Number of ECGs in the training set
    print(f"  - Number of ECGs: {X_train.shape[0]}")

    # Number of samples per recording (5000 samples)
    # This is computed to be 5000 samples per recording 
    # = 10 second recording * 500 samples per second
    # This is how the heart's electrical activity changes over 10 seconds, allowing the model to see complete heartbeat cycles and detect abnormal patterns
    print(f"  - Number of samples: {X_train.shape[1]}")

    # Number of leads for each ECG reading at a given moment in the recording (12 leads)
    # Leads: Number of measurements at different parts of the body
    # The values of each lead per recording are used to train the model to classify the ECG as normal or abnormal
    # These values are shown as a time sequence over the heart's electrical activity
    print(f"  - Number of leads: {X_train.shape[2]}")

    # Testing set
    print(f"\nTest set shape: {X_test.shape}")
    print(f"  - Number of ECGs: {X_test.shape[0]}")
    print(f"  - Signal recording length: {X_test.shape[1]}")
    print(f"  - Number of leads: {X_test.shape[2]}")

    # Class balances
    Training_Normal_count = y_train.sum()
    Training_Abnormal_count = len(y_train) - y_train.sum()
    print(f"\nTraining class balance: {Training_Normal_count} normal / {Training_Abnormal_count} abnormal ({Training_Normal_count / len(y_train) * 100:.1f}% / {Training_Abnormal_count / len(y_train) * 100:.1f}%)")
    
    Testing_Normal_count = y_test.sum()
    Testing_Abnormal_count = len(y_test) - y_test.sum()
    print(f"Test class balance: {Testing_Normal_count} normal / {Testing_Abnormal_count} abnormal ({Testing_Normal_count / len(y_test) * 100:.1f}% / {Testing_Abnormal_count / len(y_test) * 100:.1f}%)")
    
    print("\n" + "=" * 60)
    print("Data load complete.")
    print("=" * 60)

    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    # Run test
    X_train, y_train, X_test, y_test = main()


'''

The model looks at 12 leads of the heart's electrical activity over 10 seconds.
Each moment is 1/500th of a second (0.002s), so the value of these 12 leads the model is looking at changes every 0.002 seconds.
The model processes the ENTIRE 10-second sequence at once.
Shallow layers of the model will look at readings over short windows of time (e.g. 100ms), while deep layers will look at readings over longer windows of time (e.g. 1 second).

Shallow layers learn FAST patterns that are calculated over shorter periods of time:
- Individual spikes
- Immediate changes
- "What's happening RIGHT NOW in this small window?"
- Shallow layers detect the RAW SHAPES

Deep layers learn SLOW patterns that are calculated over longer periods of time; they can look at larger portions of the 10s recording (5000 samples):
- Overall rhythm
- Trends over multiple heartbeats  
- "What's the big picture across 10 seconds?"
- Deep layers INTERPRET what those shapes mean

My thesis is betting on the hypothesis that deep layers don't need every update because they're learning slower, more stable concepts/patterns.


EXAMPLE ECG MODEL:

Input: (5000, 12)
    ↓
Conv Layer 1 (Shallow): kernel size = 50 samples
- Looks at 0.1 second windows
- Detects: "spike", "dip", "plateau"
    ↓
Conv Layer 2 (Shallow): kernel size = 100 samples  
- Looks at 0.2 second windows
- Detects: "QRS complex", "T wave", "P wave"
    ↓
Conv Layer 3 (Middle): kernel size = 500 samples
- Looks at 1 second windows
- Detects: "complete heartbeat", "beat interval"
    ↓
Conv Layer 4 (Deep): kernel size = 1000 samples
- Looks at 2 second windows  
- Detects: "rhythm pattern", "beat-to-beat variation"
    ↓
Global Pooling (Deep): looks at ALL 5000 samples
- Looks at entire 10 seconds
- Detects: "overall heart rate", "rhythm regularity", "consistent abnormality"
    ↓
Dense Layers (Deep):
- Combines all patterns
- Decides: "This is MI" or "This is normal"
    ↓
Output Layer: 0.85 (85% abnormal)

'''