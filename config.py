#!/usr/bin/env python
# coding: utf-8

# In[1]:


# --- config.py ---
# Central configuration file for the ML Asset Pricing Pipeline.
# Edit the settings below to match your data, desired models, and analysis parameters.

import numpy as np
import os

# <<< FILE PATHS >>>
# --------------------------------------------------------------------------
# *** POINT TO THE PREPROCESSED FILE ***
DATA_FILE = "Cleaned_OSEFX_Market_Macro_Data_PREPROCESSED.csv"
BENCHMARK_FILE = None
FF_FACTOR_FILE = "Europe_4_Factors_Monthly.csv" # <--- Path to Fama-French factor CSV (optional)
PORTFOLIO_DEFS_FILE = None
OUTPUT_DIR = "ML_Pipeline_Results_Yearly_Percentile_Preprocessed" # Changed output dir name

# <<< DATA PREPARATION CONFIG >>>
# --------------------------------------------------------------------------
# --- Column Names (These should match names in the *PREPROCESSED* CSV) ---
# *** SIMPLIFIED: Only map ID/Date if they aren't standard, target is defined below ***
COLUMN_CONFIG = {
    'date': ['Date', 'date'],
    'id': ['Instrument', 'instrument'],
    # Other columns should now have clean names from preprocess_data.py
    # We don't need mappings for price, shares, rf, book_market etc. here
    # as they are either used in preprocessing or already logged/transformed.
    'EconomicSector': ['EconomicSector'] # Keep if sector dummies are needed
}

# --- Feature Engineering (Now done externally) ---
# VARS_TO_LOG = [] # Logging is done in preprocess_data.py
TARGET_VARIABLE = "TargetReturn_t+1"         # *** MATCHES PREPROCESSING SCRIPT OUTPUT ***
NEXT_RETURN_VARIABLE = "NextMonthlyReturn_t+1" # *** MATCHES PREPROCESSING SCRIPT OUTPUT ***
MARKET_CAP_ORIG_VARIABLE = "MarketCap_orig"    # *** MATCHES PREPROCESSING SCRIPT OUTPUT ***

# --- Data Cleaning & Filtering (Within Pipeline) ---
# WINSORIZE_LIMITS = [] # Winsorizing of raw returns done externally
# Imputation will happen in clean_data on features
# Dropping NaNs focuses on target/ID/next_ret/mkt_cap_orig (essentials for modeling/analysis)
ESSENTIAL_COLS_FOR_DROPNA = ['Date', 'Instrument', TARGET_VARIABLE, NEXT_RETURN_VARIABLE, MARKET_CAP_ORIG_VARIABLE]

# <<< ROLLING WINDOW CONFIG >>>
# --------------------------------------------------------------------------
INITIAL_TRAIN_YEARS = 9
VALIDATION_YEARS = 6
TEST_YEARS_PER_WINDOW = 1

# <<< MODEL CONFIGURATION >>>
# --------------------------------------------------------------------------
# --- Model Selection ---
RUN_MODELS = {
    'OLS': True,        'OLS3H': True,      'PLS': True,        'PCR': True,
    'ENET': True,       'GLM_H': True,      'RF': True,         'GBRT_H': True,
    'NN1': False,        'NN2': False,        'NN3': False,         'NN4': False,
    'NN5': False,
}

# --- Feature Sets ---
# *** THESE MUST MATCH THE COLUMN NAMES IN THE *PREPROCESSED* CSV ***
OLS3_FEATURE_NAMES = ["log_BM", "Momentum_12M", "log_MarketCap"] # Verify these exist in the preprocessed file
MODEL_FEATURE_MAP = { # Which feature set does each model use?
    'OLS': 'all_numeric', 'OLS3H': 'ols3_features', 'PLS': 'all_numeric',
    'PCR': 'all_numeric', 'ENET': 'all_numeric', 'GLM_H': 'all_numeric',
    'RF': 'all_numeric', 'GBRT_H': 'all_numeric',
    'NN1': 'all_numeric', 'NN2': 'all_numeric', 'NN3': 'all_numeric',
    'NN4': 'all_numeric', 'NN5': 'all_numeric',
}

# --- Model Hyperparameters (Keep as is, or adjust) ---
MODEL_PARAMS = {
    'OLS': {},
    'OLS3H': {'maxiter': 100, 'tol': 1e-6},
    'PLS': {'n_components_grid': [1, 3, 5, 8, 10, 15]},
    'PCR': {'n_components_grid': [1, 5, 10, 15, 20, 25]},
    'ENET': {'alphas': np.logspace(-6, 1, 8), 'l1_ratio': [0.1, 0.5, 0.9, 0.99, 1.0], 'cv_folds': 3, 'max_iter': 1000, 'tol': 0.001, 'n_jobs': -1},
    'GLM_H': {'param_grid': {'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0], 'epsilon': [1.1, 1.35, 1.5, 2.0]}, 'max_iter': 300},
    'RF': {'param_grid': {'n_estimators': [100], 'max_depth': [3, 6, 10], 'min_samples_leaf': [50, 100], 'max_features': ['sqrt', 0.33]}, 'n_jobs': -1, 'random_state': 42},
    'GBRT_H': {'param_grid': {'n_estimators': [100], 'learning_rate': [0.1], 'max_depth': [3, 5], 'min_samples_leaf': [50, 100], 'alpha': [0.9]}, 'loss': 'huber', 'random_state': 42},
    'NN_SHARED': {'param_grid': {'lambda1': [1e-5, 1e-4, 1e-3], 'learning_rate': [0.001, 0.01]}, 'epochs': 100, 'batch_size': 10000, 'patience': 5, 'ensemble_size': 10, 'random_seed_base': 42},
    'NN1': {'name': 'NN1', 'hidden_units': [32]},
    'NN2': {'name': 'NN2', 'hidden_units': [64, 32]},
    'NN3': {'name': 'NN3', 'hidden_units': [96, 64, 32]},
    'NN4': {'name': 'NN4', 'hidden_units': [128, 96, 64, 32]},
    'NN5': {'name': 'NN5', 'hidden_units': [128, 96, 64, 32, 16]},
}

# <<< ANALYSIS & REPORTING CONFIG >>>
# --------------------------------------------------------------------------
SUBSETS_TO_RUN = ['all', 'big', 'small']
BIG_FIRM_TOP_PERCENT = 30
SMALL_FIRM_BOTTOM_PERCENT = 30
# Use the specific market cap column saved for this purpose
FILTER_SMALL_CAPS_PORTFOLIO = False
ANNUALIZATION_FACTOR = 12

# --- Variable Importance ---
CALCULATE_VI = True
VI_METHOD = 'permutation_zero'
VI_PLOT_TOP_N = 20
MODEL_VI_STRATEGY = {
    'OLS': 'per_window', 'OLS3H': 'per_window', 'PLS': 'per_window',
    'PCR': 'per_window', 'ENET': 'per_window', 'GLM_H': 'per_window',
    'RF': 'last_window', 'GBRT_H': 'last_window',
    'NN1': 'last_window', 'NN2': 'last_window', 'NN3': 'last_window',
    'NN4': 'last_window', 'NN5': 'last_window',
}

# --- Complexity Plotting ---
COMPLEXITY_PARAMS_TO_PLOT = {
    'PLS': ['optim_n_components'], 'PCR': ['optim_n_components'],
    'ENET': ['optim_alpha', 'optim_l1_ratio'], 'GLM_H': ['optim_alpha', 'optim_epsilon'],
    'RF': ['optim_max_depth'],
    'GBRT_H': ['optim_max_depth'],
    'NN1': ['optim_lambda1', 'optim_learning_rate'], 'NN2': ['optim_lambda1', 'optim_learning_rate'],
    'NN3': ['optim_lambda1', 'optim_learning_rate'], 'NN4': ['optim_lambda1', 'optim_learning_rate'],
    'NN5': ['optim_lambda1', 'optim_learning_rate'],
}

# --- Seeds ---
GENERAL_SEED = 42
TF_SEED = 42

# --- Create Output Directory ---
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"Created output directory: {OUTPUT_DIR}")

print("Configuration loaded from config.py")

