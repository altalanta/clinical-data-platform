"""
Data loading and I/O utilities for DepMap and TCGA datasets.

This module handles:
- Automated dataset downloading with license compliance
- Data loading with proper validation
- Metadata extraction and quality checks
- Group-aware data splits without leakage
"""

import os
import hashlib
import requests
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional, List, Any
from pathlib import Path
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


class DatasetLoader:
    """Base class for dataset loaders."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data_dir = Path(config.get("data_dir", "data"))
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def download_file(self, url: str, filepath: Path, expected_hash: Optional[str] = None) -> None:
        """Download file with progress bar and hash verification."""
        if filepath.exists() and expected_hash:
            # Verify existing file
            if self._verify_hash(filepath, expected_hash):
                logger.info(f"File {filepath} already exists and hash matches")
                return
                
        logger.info(f"Downloading {url} to {filepath}")
        
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filepath, 'wb') as f, tqdm(
            desc=filepath.name,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
                    
        if expected_hash and not self._verify_hash(filepath, expected_hash):
            raise ValueError(f"Hash verification failed for {filepath}")
            
    def _verify_hash(self, filepath: Path, expected_hash: str) -> bool:
        """Verify file hash."""
        sha256_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest() == expected_hash
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
        """Load data and return (X, y, groups, tissue, batch)."""
        raise NotImplementedError


class DepMapLoader(DatasetLoader):
    """Loader for DepMap data (Cancer Dependency Map)."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.dataset_config = config.get("dataset", {})
        
    def download_depmap_data(self) -> None:
        """Download DepMap datasets if needed."""
        download_urls = self.dataset_config.get("download_urls", {})
        
        # Create DepMap data directory
        depmap_dir = self.data_dir / "depmap"
        depmap_dir.mkdir(exist_ok=True)
        
        # Download files
        files_to_download = [
            ("expression", "OmicsExpressionProteinCodingGenesTPMLogp1.csv"),
            ("gene_effect", "CRISPRGeneEffect.csv"),
            ("sample_info", "Model.csv")
        ]
        
        for key, filename in files_to_download:
            if key in download_urls:
                url = download_urls[key]
                filepath = depmap_dir / filename
                
                try:
                    self.download_file(url, filepath)
                except Exception as e:
                    logger.warning(f"Failed to download {filename}: {e}")
                    logger.info(f"Please manually download {filename} from DepMap portal")
                    
    def load_data(self) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        Load DepMap data.
        
        Returns:
            X: Gene expression features (n_samples, n_genes)
            y: Binary labels based on gene dependency
            groups: Cell line identifiers for group-aware splits
            tissue: Tissue/lineage information for slice analysis
            batch: Batch information for correction
        """
        # Download data if needed
        self.download_depmap_data()
        
        data_sources = self.dataset_config.get("data_sources", {})
        
        # Load expression data
        expr_path = Path(data_sources.get("expression"))
        if not expr_path.exists():
            raise FileNotFoundError(f"Expression data not found: {expr_path}")
            
        logger.info("Loading expression data...")
        expression_df = pd.read_csv(expr_path, index_col=0)
        
        # Load gene effect data (CRISPR dependency scores)
        effect_path = Path(data_sources.get("gene_effect"))
        if not effect_path.exists():
            raise FileNotFoundError(f"Gene effect data not found: {effect_path}")
            
        logger.info("Loading gene effect data...")
        gene_effect_df = pd.read_csv(effect_path, index_col=0)
        
        # Load sample metadata
        sample_path = Path(data_sources.get("sample_info"))
        if not sample_path.exists():
            raise FileNotFoundError(f"Sample info not found: {sample_path}")
            
        logger.info("Loading sample metadata...")
        sample_info_df = pd.read_csv(sample_path, index_col=0)
        
        # Create target labels based on gene dependency
        target_config = self.dataset_config.get("target", {})
        target_gene = target_config.get("gene_name", "TP53")
        threshold = target_config.get("threshold", -0.5)
        
        if target_gene not in gene_effect_df.columns:
            # Try with gene symbol formatting
            target_gene_formatted = f"{target_gene} ({target_gene})"
            if target_gene_formatted in gene_effect_df.columns:
                target_gene = target_gene_formatted
            else:
                available_genes = [col for col in gene_effect_df.columns if target_gene in col]
                if available_genes:
                    target_gene = available_genes[0]
                    logger.info(f"Using gene: {target_gene}")
                else:
                    raise ValueError(f"Target gene {target_gene} not found in gene effect data")
        
        # Find common samples
        common_samples = (
            set(expression_df.index) & 
            set(gene_effect_df.index) & 
            set(sample_info_df.index)
        )
        
        if len(common_samples) == 0:
            raise ValueError("No common samples found across datasets")
            
        logger.info(f"Found {len(common_samples)} common samples")
        
        # Filter to common samples
        expression_df = expression_df.loc[list(common_samples)]
        gene_effect_df = gene_effect_df.loc[list(common_samples)]
        sample_info_df = sample_info_df.loc[list(common_samples)]
        
        # Create binary labels (essential vs non-essential)
        y = (gene_effect_df[target_gene] <= threshold).astype(int)
        
        # Extract grouping variables
        grouping_config = self.dataset_config.get("grouping", {})
        
        # Groups (cell line IDs)
        group_key = grouping_config.get("key", "ModelID")
        if group_key in sample_info_df.columns:
            groups = sample_info_df[group_key].astype(str)
        else:
            groups = pd.Series(sample_info_df.index, index=sample_info_df.index, name="groups")
            
        # Tissue information
        tissue_key = grouping_config.get("tissue_key", "OncotreePrimaryDisease")
        if tissue_key in sample_info_df.columns:
            tissue = sample_info_df[tissue_key].fillna("Unknown")
        else:
            tissue = pd.Series("Unknown", index=sample_info_df.index, name="tissue")
            
        # Batch information
        batch_key = grouping_config.get("batch_key", "DepmapModelType")
        if batch_key in sample_info_df.columns:
            batch = sample_info_df[batch_key].fillna("Unknown")
        else:
            batch = pd.Series("Unknown", index=sample_info_df.index, name="batch")
        
        # Apply filters
        filters = self.dataset_config.get("filters", {})
        
        # Filter by tissue representation
        min_samples = filters.get("min_samples_per_tissue", 10)
        tissue_counts = tissue.value_counts()
        valid_tissues = tissue_counts[tissue_counts >= min_samples].index
        
        mask = tissue.isin(valid_tissues)
        
        # Filter by missing data
        max_missing = filters.get("max_missing_fraction", 0.1)
        missing_fraction = expression_df.isnull().mean(axis=1)
        mask = mask & (missing_fraction <= max_missing)
        
        logger.info(f"After filtering: {mask.sum()} samples remaining")
        
        # Apply filters
        X = expression_df.loc[mask]
        y = y.loc[mask]
        groups = groups.loc[mask]
        tissue = tissue.loc[mask]
        batch = batch.loc[mask]
        
        # Log class distribution
        logger.info(f"Class distribution: {y.value_counts().to_dict()}")
        logger.info(f"Tissue distribution: {tissue.value_counts().head(10).to_dict()}")
        
        return X, y, groups, tissue, batch


class TCGALoader(DatasetLoader):
    """Loader for TCGA data (The Cancer Genome Atlas)."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.dataset_config = config.get("dataset", {})
        
    def download_tcga_data(self) -> None:
        """Download TCGA datasets from GDC API."""
        logger.info("TCGA data download requires GDC API access")
        logger.info("Please refer to data/README.md for manual download instructions")
        
        # Note: TCGA data download is complex and requires specific GDC API calls
        # This is typically done manually or with specialized tools
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        Load TCGA data.
        
        Returns:
            X: Gene expression features (n_samples, n_genes)
            y: Binary labels based on cancer subtype
            groups: Patient identifiers for group-aware splits
            tissue: Cancer type information for slice analysis
            batch: Batch information for correction
        """
        data_sources = self.dataset_config.get("data_sources", {})
        
        # Load expression data
        expr_path = Path(data_sources.get("expression"))
        if not expr_path.exists():
            raise FileNotFoundError(f"Expression data not found: {expr_path}")
            
        logger.info("Loading TCGA expression data...")
        expression_df = pd.read_csv(expr_path, index_col=0)
        
        # Load clinical data
        clinical_path = Path(data_sources.get("clinical"))
        if not clinical_path.exists():
            raise FileNotFoundError(f"Clinical data not found: {clinical_path}")
            
        logger.info("Loading clinical data...")
        clinical_df = pd.read_csv(clinical_path, index_col=0)
        
        # Load subtype labels
        subtype_path = Path(data_sources.get("subtype_labels"))
        if not subtype_path.exists():
            raise FileNotFoundError(f"Subtype labels not found: {subtype_path}")
            
        logger.info("Loading subtype labels...")
        subtype_df = pd.read_csv(subtype_path, index_col=0)
        
        # Find common samples
        common_samples = (
            set(expression_df.index) & 
            set(clinical_df.index) & 
            set(subtype_df.index)
        )
        
        if len(common_samples) == 0:
            raise ValueError("No common samples found across TCGA datasets")
            
        logger.info(f"Found {len(common_samples)} common TCGA samples")
        
        # Filter to common samples
        expression_df = expression_df.loc[list(common_samples)]
        clinical_df = clinical_df.loc[list(common_samples)]
        subtype_df = subtype_df.loc[list(common_samples)]
        
        # Create target labels
        target_config = self.dataset_config.get("target", {})
        label_col = target_config.get("label_column", "PAM50_subtype")
        positive_class = target_config.get("positive_class", "Basal")
        
        if label_col not in subtype_df.columns:
            raise ValueError(f"Label column {label_col} not found in subtype data")
            
        y = (subtype_df[label_col] == positive_class).astype(int)
        
        # Extract grouping variables
        grouping_config = self.dataset_config.get("grouping", {})
        
        # Groups (patient IDs)
        group_key = grouping_config.get("key", "patient_id")
        if group_key in clinical_df.columns:
            groups = clinical_df[group_key].astype(str)
        else:
            groups = pd.Series(clinical_df.index, index=clinical_df.index, name="groups")
            
        # Tissue information (cancer type)
        tissue_key = grouping_config.get("tissue_key", "cancer_type")
        if tissue_key in clinical_df.columns:
            tissue = clinical_df[tissue_key].fillna("Unknown")
        else:
            tissue = pd.Series("Unknown", index=clinical_df.index, name="tissue")
            
        # Batch information
        batch_key = grouping_config.get("batch_key", "tissue_source_site")
        if batch_key in clinical_df.columns:
            batch = clinical_df[batch_key].fillna("Unknown")
        else:
            batch = pd.Series("Unknown", index=clinical_df.index, name="batch")
        
        # Apply filters
        filters = self.dataset_config.get("filters", {})
        
        # Filter by cancer types if specified
        cancer_types = filters.get("cancer_types")
        if cancer_types:
            mask = tissue.isin(cancer_types)
        else:
            mask = pd.Series(True, index=tissue.index)
            
        # Filter by tissue representation
        min_samples = filters.get("min_samples_per_type", 20)
        tissue_counts = tissue.value_counts()
        valid_tissues = tissue_counts[tissue_counts >= min_samples].index
        
        mask = mask & tissue.isin(valid_tissues)
        
        # Filter by missing data
        max_missing = filters.get("max_missing_fraction", 0.05)
        missing_fraction = expression_df.isnull().mean(axis=1)
        mask = mask & (missing_fraction <= max_missing)
        
        logger.info(f"After filtering: {mask.sum()} TCGA samples remaining")
        
        # Apply filters
        X = expression_df.loc[mask]
        y = y.loc[mask]
        groups = groups.loc[mask]
        tissue = tissue.loc[mask]
        batch = batch.loc[mask]
        
        # Log distributions
        logger.info(f"Class distribution: {y.value_counts().to_dict()}")
        logger.info(f"Cancer type distribution: {tissue.value_counts().to_dict()}")
        
        return X, y, groups, tissue, batch


def load_dataset(config: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Load dataset based on configuration.
    
    Args:
        config: Dataset configuration
        
    Returns:
        Tuple of (X, y, groups, tissue, batch)
    """
    dataset_name = config.get("dataset", {}).get("name", "depmap")
    
    if dataset_name == "depmap":
        loader = DepMapLoader(config)
    elif dataset_name == "tcga":
        loader = TCGALoader(config)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
        
    return loader.load_data()


def compute_data_hash(X: pd.DataFrame, y: pd.Series) -> str:
    """Compute hash of dataset for reproducibility tracking."""
    data_str = f"{X.shape}_{X.columns.tolist()}_{y.sum()}_{len(y)}"
    return hashlib.md5(data_str.encode()).hexdigest()[:8]