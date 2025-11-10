# Data Handling Guidelines

## Overview

This directory should contain metabolomics data files. **IMPORTANT**: Only synthetic data should be committed to this repository. Real patient data must be stored separately with proper security measures.

## Data Format

### Expected CSV Structure

The metabolomics dataset should be a CSV file with the following columns:

#### Metadata Columns

- `sample_id`: Unique sample identifier (e.g., "SAMPLE_0001")
- `patient_id_pseudonym`: Pseudonymized patient ID (e.g., "P1234")
- `collection_date_relative`: Days since study start (integer, not actual dates)
- `age_range`: Age category (e.g., "18-30", "31-45", "46-60", "61-75", "76+")
- `sex`: Biological sex ("M" or "F")
- `diagnosis_label`: Diagnosis category
  - `"control"`: No cancer
  - `"cancer_prostate"`: Prostate cancer
  - `"cancer_bladder"`: Bladder cancer
  - `"cancer_kidney"`: Kidney cancer
  - `"cancer_other"`: Other urological cancers
- `tumor_stage`: Tumor stage for cases ("T1", "T2", "T3", "T4", or "N/A" for controls)

#### Quality Control Columns

- `batch_id`: Batch identifier (e.g., "BATCH_1", "BATCH_2")
- `instrument_id`: Instrument identifier (e.g., "INST_1", "INST_2")
- `internal_standard_values`: Internal standard intensity values
- `creatinine_normalization_factor`: Creatinine normalization factor (if applicable)

#### Optional Clinical Metadata

- `urinalysis_flag`: Urinalysis results ("normal", "abnormal", "N/A")
- `smoking_status`: Smoking history ("never", "former", "current", "N/A")
- `BMI`: Body Mass Index (numeric or "N/A")

#### Metabolite Columns

- `metab_0001`, `metab_0002`, ..., `metab_0NNN`: Metabolite intensity values
  - Format: Numeric (LC-MS peak intensities or NMR bins)
  - Missing values: NaN or empty (will be imputed)
  - Units: Raw intensities (will be normalized)

### Example Data Structure

```csv
sample_id,patient_id_pseudonym,collection_date_relative,age_range,sex,diagnosis_label,tumor_stage,batch_id,instrument_id,internal_standard_values,creatinine_normalization_factor,metab_0001,metab_0002,...
SAMPLE_0001,P1234,15,46-60,M,control,N/A,BATCH_1,INST_1,10.5,1.2,1250.3,890.1,...
SAMPLE_0002,P5678,22,61-75,M,cancer_prostate,T2,BATCH_1,INST_1,10.3,1.1,1450.2,920.5,...
```

## Synthetic Data

### Generating Synthetic Data

Synthetic data for demonstration can be generated using:

```bash
python src/data/synth_generator.py --out data/synthetic --n 200 --n_metab 500
```

This creates:
- `synthetic_urine_metabolomics.csv`: Main dataset
- `synthetic_data_summary.txt`: Summary statistics

### Synthetic Data Characteristics

- Realistic metabolite intensity distributions (log-normal)
- Simulated batch effects
- Simulated disease signatures
- Missing values (~5%)
- Demographic distributions

**Note**: Synthetic data is for demonstration only and does not represent real biological patterns.

## Real Patient Data

### ⚠️ CRITICAL REQUIREMENTS

1. **IRB Approval**: Must have Institutional Review Board approval
2. **Informed Consent**: All participants must provide informed consent
3. **Secure Storage**: Store on encrypted, access-controlled servers
4. **NOT in Repository**: Real data must NEVER be committed to this repository

### Data Collection Protocol

1. **Sample Collection**:
   - Standardized time (fasting status documented)
   - Storage at -80°C
   - No anticoagulants
   - Chain of custody maintained

2. **Analysis Platform**:
   - LC-MS (Liquid Chromatography-Mass Spectrometry) or
   - NMR (Nuclear Magnetic Resonance)
   - Document platform and parameters

3. **Quality Control**:
   - Internal standards included
   - Blank samples
   - Pooled QC samples
   - Batch randomization
   - Instrument calibration records

### Data Anonymization Checklist

Before any analysis, ensure:

- [ ] Patient names removed
- [ ] Medical record numbers removed or pseudonymized
- [ ] Dates converted to relative dates (days since)
- [ ] Birth dates converted to age ranges
- [ ] Geographic identifiers removed (if small populations)
- [ ] Any other identifiers removed
- [ ] Pseudonymization mapping stored separately (encrypted)

### Data Storage

**Real patient data should be stored**:
- On secure institutional servers
- With encryption at rest and in transit
- With access controls and audit logs
- Following institutional data policies
- With regular backups

**NOT in**:
- This GitHub repository
- Public cloud storage (without proper agreements)
- Unencrypted local drives
- Email attachments

## Data Preprocessing

The preprocessing pipeline handles:

1. **Missing Value Imputation**:
   - K-NN imputation (default)
   - Half-minimum imputation
   - Median imputation

2. **Normalization**:
   - Log2 transform (default)
   - Total Ion Current (TIC) normalization
   - Creatinine normalization (if factor provided)
   - Variance Stabilizing Normalization (VSN)

3. **Batch Effect Correction**:
   - Mean centering per batch
   - (Advanced: ComBat, limma, RUV - requires additional packages)

4. **Feature Scaling**:
   - Z-score standardization (default)

5. **Feature Selection**:
   - Univariate filtering (F-test)
   - Mutual information
   - PCA, LDA, UMAP

See `src/preprocessing.py` and `src/features.py` for details.

## Data Sharing

### Public Sharing (GitHub, etc.)

✅ **Allowed**:
- Synthetic data
- Code and pipelines
- Documentation
- Aggregate statistics (properly anonymized)

❌ **NOT Allowed**:
- Real patient data
- Individual-level data
- Data that could be re-identified

### Real Data Sharing

Requires:
1. **Data Transfer Agreement (DTA)**
2. **Recipient IRB approval**
3. **Explicit participant consent** (for data sharing)
4. **Compliance with regulations** (GDPR, HIPAA, etc.)

## File Naming Conventions

- Synthetic data: `synthetic_urine_metabolomics.csv`
- Real data: Use institutional naming conventions (NOT in this repo)
- Processed data: `processed_*.csv` (if needed for intermediate steps)
- Model outputs: Store in `models/` directory

## Version Control

- Use version numbers for datasets
- Document changes in data versions
- Maintain audit trail (who/when/why)
- Tag releases appropriately

## Troubleshooting

### Common Issues

1. **Missing metabolite columns**: Ensure all `metab_XXXX` columns are present
2. **Incorrect data types**: Check that metabolite columns are numeric
3. **Excessive missing values**: Review data quality, may need different imputation
4. **Batch effects**: Ensure batch IDs are correctly labeled

### Data Quality Checks

Before analysis, verify:
- [ ] All required columns present
- [ ] No PHI in data
- [ ] Missing value percentage reasonable (<20% per feature)
- [ ] Batch IDs correctly assigned
- [ ] Diagnosis labels match expected format
- [ ] No duplicate sample IDs

## Contact

For questions about data handling:
- **Data Manager**: [Contact]
- **IRB Office**: [Contact]
- **Principal Investigator**: [Contact]

---

**Remember**: When in doubt about data handling, consult your IRB and data protection officer.

