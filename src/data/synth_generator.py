"""
Synthetic Urine Metabolomics Data Generator

Generates realistic synthetic metabolomic profiles for demonstration purposes.
This is NOT real patient data. For real data, see README_DATA.md.

Usage:
    python src/data/synth_generator.py --out data/synthetic --n 200 --n_metab 500
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple
import warnings
warnings.filterwarnings('ignore')


class SyntheticMetabolomicsGenerator:
    """Generate synthetic urine metabolomic profiles with realistic characteristics."""
    
    def __init__(
        self,
        n_samples: int = 200,
        n_metabolites: int = 500,
        n_batches: int = 3,
        case_ratio: float = 0.3,
        cancer_subtypes: list = None,
        random_seed: int = 42
    ):
        """
        Initialize generator.
        
        Args:
            n_samples: Total number of samples
            n_metabolites: Number of metabolite features
            n_batches: Number of batches (for batch effect simulation)
            case_ratio: Proportion of cancer cases (0.0-1.0)
            cancer_subtypes: List of cancer subtypes (e.g., ['prostate', 'bladder', 'kidney'])
            random_seed: Random seed for reproducibility
        """
        self.n_samples = n_samples
        self.n_metabolites = n_metabolites
        self.n_batches = n_batches
        self.case_ratio = case_ratio
        self.cancer_subtypes = cancer_subtypes or ['prostate', 'bladder', 'kidney', 'other']
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
    def generate_metadata(self) -> pd.DataFrame:
        """Generate sample metadata."""
        n_cases = int(self.n_samples * self.case_ratio)
        n_controls = self.n_samples - n_cases
        
        # Sample IDs
        sample_ids = [f"SAMPLE_{i:04d}" for i in range(1, self.n_samples + 1)]
        
        # Pseudonymized patient IDs
        patient_ids = [f"P{np.random.randint(1000, 9999):04d}" for _ in range(self.n_samples)]
        
        # Relative collection dates (days since study start)
        collection_dates = np.random.randint(0, 365, self.n_samples)
        
        # Age ranges
        age_ranges = np.random.choice(
            ['18-30', '31-45', '46-60', '61-75', '76+'],
            size=self.n_samples,
            p=[0.1, 0.2, 0.3, 0.3, 0.1]
        )
        
        # Sex
        sex = np.random.choice(['M', 'F'], size=self.n_samples, p=[0.6, 0.4])
        
        # Diagnosis labels
        labels = ['control'] * n_controls
        case_labels = []
        for _ in range(n_cases):
            subtype = np.random.choice(self.cancer_subtypes, p=[0.4, 0.3, 0.2, 0.1])
            case_labels.append(f"cancer_{subtype}")
        labels.extend(case_labels)
        np.random.shuffle(labels)
        
        # Tumor stage (only for cases)
        tumor_stages = []
        for label in labels:
            if label == 'control':
                tumor_stages.append('N/A')
            else:
                tumor_stages.append(np.random.choice(['T1', 'T2', 'T3', 'T4'], p=[0.3, 0.4, 0.2, 0.1]))
        
        # Batch assignment (with some stratification)
        batch_ids = []
        for i, label in enumerate(labels):
            if label == 'control':
                batch_ids.append(f"BATCH_{np.random.randint(1, self.n_batches + 1)}")
            else:
                # Cases slightly more concentrated in certain batches (realistic)
                batch_ids.append(f"BATCH_{np.random.choice([1, 2, 3], p=[0.4, 0.35, 0.25])}")
        
        # Instrument IDs
        instrument_ids = [f"INST_{np.random.randint(1, 3)}" for _ in range(self.n_samples)]
        
        # Internal standard values (simulated)
        internal_standard_values = np.random.lognormal(mean=10, sigma=0.2, size=self.n_samples)
        
        # Creatinine normalization factor
        creatinine_factors = np.random.lognormal(mean=1.0, sigma=0.3, size=self.n_samples)
        
        # Optional clinical metadata
        urinalysis_flags = np.random.choice(['normal', 'abnormal', 'N/A'], 
                                           size=self.n_samples, p=[0.7, 0.2, 0.1])
        smoking_status = np.random.choice(['never', 'former', 'current', 'N/A'],
                                         size=self.n_samples, p=[0.5, 0.2, 0.2, 0.1])
        bmi = np.random.normal(25, 4, self.n_samples).clip(18, 40)
        bmi = [f"{b:.1f}" if np.random.random() > 0.1 else "N/A" for b in bmi]
        
        metadata = pd.DataFrame({
            'sample_id': sample_ids,
            'patient_id_pseudonym': patient_ids,
            'collection_date_relative': collection_dates,
            'age_range': age_ranges,
            'sex': sex,
            'diagnosis_label': labels,
            'tumor_stage': tumor_stages,
            'batch_id': batch_ids,
            'instrument_id': instrument_ids,
            'internal_standard_values': internal_standard_values,
            'creatinine_normalization_factor': creatinine_factors,
            'urinalysis_flag': urinalysis_flags,
            'smoking_status': smoking_status,
            'BMI': bmi
        })
        
        return metadata
    
    def generate_metabolite_data(self, metadata: pd.DataFrame) -> pd.DataFrame:
        """Generate metabolite intensity data with realistic patterns."""
        n_samples = len(metadata)
        
        # Base metabolite intensities (log-normal distribution, typical for LC-MS)
        base_intensities = np.random.lognormal(mean=8, sigma=2, size=(n_samples, self.n_metabolites))
        
        # Add batch effects
        batch_effects = {}
        for batch_id in metadata['batch_id'].unique():
            batch_effect = np.random.normal(0, 0.5, self.n_metabolites)
            batch_effects[batch_id] = batch_effect
        
        for i, batch_id in enumerate(metadata['batch_id']):
            base_intensities[i, :] += batch_effects[batch_id]
        
        # Add instrument effects
        for inst_id in metadata['instrument_id'].unique():
            inst_mask = metadata['instrument_id'] == inst_id
            inst_effect = np.random.normal(0, 0.2, self.n_metabolites)
            base_intensities[inst_mask, :] += inst_effect
        
        # Add disease-specific signatures
        # Select ~10% of metabolites as "biomarkers"
        n_biomarkers = int(self.n_metabolites * 0.1)
        biomarker_indices = np.random.choice(self.n_metabolites, size=n_biomarkers, replace=False)
        
        for i, row in metadata.iterrows():
            if row['diagnosis_label'] != 'control':
                # Cancer samples have altered metabolite levels
                cancer_effect = np.random.normal(0.5, 0.3, n_biomarkers)
                base_intensities[i, biomarker_indices] += cancer_effect
                
                # Subtype-specific effects
                if 'prostate' in row['diagnosis_label']:
                    subtype_metabs = biomarker_indices[:n_biomarkers//4]
                    base_intensities[i, subtype_metabs] += np.random.normal(0.3, 0.2, len(subtype_metabs))
                elif 'bladder' in row['diagnosis_label']:
                    subtype_metabs = biomarker_indices[n_biomarkers//4:2*n_biomarkers//4]
                    base_intensities[i, subtype_metabs] += np.random.normal(0.3, 0.2, len(subtype_metabs))
                elif 'kidney' in row['diagnosis_label']:
                    subtype_metabs = biomarker_indices[2*n_metabolites//4:3*n_metabolites//4]
                    base_intensities[i, subtype_metabs] += np.random.normal(0.3, 0.2, len(subtype_metabs))
        
        # Add age/sex effects (subtle)
        for i, row in metadata.iterrows():
            if row['age_range'] in ['61-75', '76+']:
                base_intensities[i, :] += np.random.normal(0.1, 0.05, self.n_metabolites)
            if row['sex'] == 'F':
                sex_metabs = np.random.choice(self.n_metabolites, size=50, replace=False)
                base_intensities[i, sex_metabs] += np.random.normal(0.15, 0.1, 50)
        
        # Add missing values (non-detects, ~5% missing)
        missing_mask = np.random.random((n_samples, self.n_metabolites)) < 0.05
        base_intensities[missing_mask] = np.nan
        
        # Create metabolite column names
        metab_cols = [f"metab_{i:04d}" for i in range(1, self.n_metabolites + 1)]
        
        metabolite_df = pd.DataFrame(base_intensities, columns=metab_cols)
        
        return metabolite_df
    
    def generate_dataset(self) -> pd.DataFrame:
        """Generate complete dataset with metadata and metabolites."""
        metadata = self.generate_metadata()
        metabolite_data = self.generate_metabolite_data(metadata)
        
        # Combine
        dataset = pd.concat([metadata, metabolite_data], axis=1)
        
        return dataset


def main():
    parser = argparse.ArgumentParser(
        description='Generate synthetic urine metabolomics data'
    )
    parser.add_argument(
        '--out',
        type=str,
        default='data/synthetic',
        help='Output directory'
    )
    parser.add_argument(
        '--n',
        type=int,
        default=200,
        help='Number of samples'
    )
    parser.add_argument(
        '--n_metab',
        type=int,
        default=500,
        help='Number of metabolites'
    )
    parser.add_argument(
        '--n_batches',
        type=int,
        default=3,
        help='Number of batches'
    )
    parser.add_argument(
        '--case_ratio',
        type=float,
        default=0.3,
        help='Proportion of cancer cases (0.0-1.0)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    out_path = Path(args.out)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Generate data
    print(f"Generating synthetic dataset with {args.n} samples and {args.n_metab} metabolites...")
    generator = SyntheticMetabolomicsGenerator(
        n_samples=args.n,
        n_metabolites=args.n_metab,
        n_batches=args.n_batches,
        case_ratio=args.case_ratio,
        random_seed=args.seed
    )
    
    dataset = generator.generate_dataset()
    
    # Save
    output_file = out_path / 'synthetic_urine_metabolomics.csv'
    dataset.to_csv(output_file, index=False)
    print(f"Saved to {output_file}")
    print(f"Dataset shape: {dataset.shape}")
    print(f"Cases: {(dataset['diagnosis_label'] != 'control').sum()}")
    print(f"Controls: {(dataset['diagnosis_label'] == 'control').sum()}")
    
    # Save summary statistics
    summary_file = out_path / 'synthetic_data_summary.txt'
    with open(summary_file, 'w') as f:
        f.write("Synthetic Urine Metabolomics Dataset Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total samples: {len(dataset)}\n")
        f.write(f"Total metabolites: {args.n_metab}\n")
        f.write(f"Cases: {(dataset['diagnosis_label'] != 'control').sum()}\n")
        f.write(f"Controls: {(dataset['diagnosis_label'] == 'control').sum()}\n\n")
        f.write("Diagnosis label distribution:\n")
        f.write(str(dataset['diagnosis_label'].value_counts()) + "\n\n")
        f.write("Batch distribution:\n")
        f.write(str(dataset['batch_id'].value_counts()) + "\n\n")
        f.write("Missing value percentage: ")
        missing_pct = dataset.iloc[:, 14:].isna().sum().sum() / (len(dataset) * args.n_metab) * 100
        f.write(f"{missing_pct:.2f}%\n")


if __name__ == '__main__':
    main()

