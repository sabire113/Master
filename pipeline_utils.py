#!/usr/bin/env python
# coding: utf-8

# In[1]:


# --- pipeline_utils.py ---
# Shared utility functions for the ML Asset Pricing Pipeline.
# Contains functions for data loading/prep, feature definition,
# standardization, cleaning, splitting, portfolio analysis, VI, plotting, saving.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from scipy.stats.mstats import winsorize # No longer needed here
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, ElasticNet, HuberRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

import warnings
import os
import time
from collections import defaultdict
import random
import traceback
import re

# Import config AFTER it's defined
import config

# --- Suppress specific warnings ---
# (Keep suppression settings as they are)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in log")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Maximum number of iterations reached.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="divide by zero encountered.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered.*divide")
pd.options.mode.chained_assignment = None

# Set general random seed
random.seed(config.GENERAL_SEED)
np.random.seed(config.GENERAL_SEED)


# === Stage 1: Data Loading and Preparation (SIMPLIFIED) ===
def find_col(df, potential_names, default=None):
    """Helper to find the first matching column name from a list."""
    for name in potential_names:
        if name in df.columns: return name
    return default

def load_prepare_data(file_path, column_config, target_var_name, next_ret_var_name, mkt_cap_orig_var_name):
    """
    Loads the PREPROCESSED data file.
    Performs minimal checks: date conversion, finds essential columns (Date, ID, Target, NextReturn, MktCapOrig).
    Does NOT calculate returns, targets, log transforms, etc. as this is assumed done externally.
    """
    print(f"\n--- 1. Laster Forhåndsbehandlet Data ---")
    print(f"Laster data fra: {file_path}")
    try:
        # Load data, ensuring Date is parsed
        df = pd.read_csv(file_path, parse_dates=['Date']) # Assume 'Date' is the date column
        print(f"Forhåndsbehandlet data lastet inn. Form: {df.shape}")
    except FileNotFoundError:
        print(f"FEIL: Fil '{file_path}' ikke funnet."); return None
    except Exception as e:
        print(f"FEIL under lasting av forhåndsbehandlet data: {e}"); return None

    # --- 1. Find and Standardize Essential Column Names ---
    std_names_map = {
        'date': 'Date',
        'id': 'Instrument',
        # Add other potential mappings from config if needed, but keep it minimal
    }
    rename_dict = {}
    found_cols = {}

    # Find Date column
    date_col_found = find_col(df, column_config.get('date', ['Date', 'date']))
    if not date_col_found: print("FEIL: Datokolonne ikke funnet."); return None
    if date_col_found != 'Date': rename_dict[date_col_found] = 'Date'
    found_cols['date'] = 'Date'

    # Find ID column
    id_col_found = find_col(df, column_config.get('id', ['Instrument', 'instrument']))
    if not id_col_found: print("FEIL: Instrument ID kolonne ikke funnet."); return None
    if id_col_found != 'Instrument': rename_dict[id_col_found] = 'Instrument'
    found_cols['id'] = 'Instrument'

    # Apply renames if necessary
    if rename_dict:
        df = df.rename(columns=rename_dict)
        print(f"Standardiserte essensielle kolonner: {list(rename_dict.values())}")

    # --- 2. Verify Essential Columns Exist ---
    essential_cols = ['Date', 'Instrument', target_var_name, next_ret_var_name, mkt_cap_orig_var_name]
    # Check if sector column exists IF it's needed later (e.g., for dummies, although dummies should be pre-created now)
    if 'EconomicSector' in column_config:
         sector_col_cand = find_col(df, column_config['EconomicSector'])
         if sector_col_cand: essential_cols.append(sector_col_cand)

    missing_essential = [col for col in essential_cols if col not in df.columns]
    if missing_essential:
        print(f"FEIL: Essensielle kolonner mangler i forhåndsbehandlet fil: {missing_essential}")
        print(f"Tilgjengelige kolonner: {df.columns.tolist()}")
        return None
    print("Essensielle kolonner funnet.")

    # --- 3. Ensure Date is Datetime and Sort ---
    if not pd.api.types.is_datetime64_any_dtype(df['Date']):
        try:
            df['Date'] = pd.to_datetime(df['Date'])
        except Exception as e:
            print(f"FEIL: Kunne ikke konvertere 'Date' kolonne til datetime: {e}"); return None
    df = df.sort_values(by=['Instrument', 'Date']).reset_index(drop=True)
    print("Data sortert etter Instrument og Dato.")

    # --- 4. Optional: Create Sector Dummies (if not already done in preprocessing) ---
    # Check if sector column exists and if dummy columns DON'T already exist
    sector_col_std = find_col(df, column_config.get('EconomicSector', []))
    if sector_col_std and not any(col.startswith("Sector_") for col in df.columns):
         print(f"  INFO: Oppretter Sektor dummy-variabler fra '{sector_col_std}'...")
         df = pd.get_dummies(df, columns=[sector_col_std], prefix="Sector", dtype=int)
         print("  Sektor dummy-variabler opprettet.")

    print(f"Lasting og grunnleggende sjekk fullført. Form: {df.shape}")
    # print(f"Final Columns: {df.columns.tolist()}") # Uncomment for detailed debug
    return df


# === Stage 2: Feature Definition ===
def define_features(df, ols3_feature_names, base_exclusions):
    """
    Identifies numeric features from the PREPROCESSED data,
    excluding specified base columns (target, IDs, intermediate calcs that might remain).
    """
    print("\n--- 2. Definerer Features (fra forhåndsbehandlet data) ---")
    if df is None or df.empty: print(" FEIL: DataFrame tom."); return [], [], []

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    print(f"  Funnet {len(numeric_cols)} num. kolonner.")

    # Define columns to exclude from features
    cols_to_exclude = set()

    # Add columns explicitly passed for exclusion (target, next_ret, mkt_cap_orig, etc.)
    base_exclusions_present = [col for col in base_exclusions if col in df.columns]
    cols_to_exclude.update(base_exclusions_present)

    # Add standard identifiers and intermediate calculation columns that might *still* be present
    # (AdjustedReturn_t is likely still there from the preprocessor script)
    date_col = 'Date' # Should be standardized now
    id_col = 'Instrument' # Should be standardized now
    std_exclusions = [date_col, id_col, 'level_0','index','Year','MonthYear',
                      'AdjustedReturn_t', # Keep this exclusion
                      'MonthlyReturn_t', 'MonthlyRiskFreeRate_t'] # Keep these
    cols_to_exclude.update([col for col in std_exclusions if col in df.columns])

    # Add portfolio helper columns if they might exist (unlikely now)
    pf_cols=['Rank','DecileRank','Decile','ew_weights','vw_weights','rank']
    cols_to_exclude.update([c for c in pf_cols if c in df.columns])

    # Exclude original versions of logged variables IF the log version exists
    # (This check remains useful, e.g., excludes MarketCap if log_MarketCap exists)
    log_cols = {c for c in numeric_cols if c.startswith('log_')}
    # List potential raw names based on how log names are created (log_VARNAME)
    raw_names_from_logs = {c.replace('log_', '') for c in log_cols}
    for raw_name in raw_names_from_logs:
        if raw_name in df.columns and f"log_{raw_name}" in log_cols:
            cols_to_exclude.add(raw_name)
            # Specific exclusions if MarketCap or ClosePrice were logged
            if raw_name in ['MarketCap', 'ClosePrice'] and 'ClosePrice' in df.columns:
                 cols_to_exclude.add('ClosePrice') # Exclude if log exists
            if raw_name == 'MarketCap' and 'CommonSharesOutstanding' in df.columns:
                 cols_to_exclude.add('CommonSharesOutstanding') # Exclude if log exists

    # Exclude raw NorgesBank10Y, NIBOR3M if TermSpread exists
    if 'TermSpread' in df.columns:
         if 'NorgesBank10Y' in df.columns: cols_to_exclude.add('NorgesBank10Y')
         if 'NIBOR3M' in df.columns: cols_to_exclude.add('NIBOR3M')

    # Identify final numeric features
    potential_features = [c for c in numeric_cols if c not in cols_to_exclude]
    final_features = []
    for col in potential_features:
        if col not in df.columns: continue
        valid = df[col].dropna()
        # Check for variance and multiple unique values
        # Use a slightly larger tolerance for std dev check with float32
        if len(valid) > 1 and valid.nunique() > 1 and valid.std() > 1e-7:
            final_features.append(col)
        # else: print(f"  -> Dropping potential feature '{col}' due to no variance or <=1 unique value.")

    final_features = sorted(list(set(final_features)))
    print(f"  Identifisert {len(final_features)} features totalt etter ekskludering og validitetssjekk.")
    print(f"  Features sample: {final_features[:5]}...{final_features[-5:]}") # Show sample
    # print(f"  Ekskluderte kolonner: {sorted(list(cols_to_exclude.intersection(df.columns)))}") # Optional Debug

    # Check OLS3 features against the *final* feature list
    # Use the names specified in config.OLS3_FEATURE_NAMES directly
    ols3_features_final = [f for f in ols3_feature_names if f in final_features]
    missing_ols3 = [f for f in ols3_feature_names if f not in ols3_features_final]

    if missing_ols3: print(f"  ADVARSEL: OLS3 mangler features fra config: {missing_ols3}")
    if not ols3_features_final: print("  ADVARSEL: Ingen av de spesifiserte OLS3 features er gyldige endelige features.")
    elif len(ols3_features_final) < len(ols3_feature_names): print(f"  ADVARSEL: Kunne finne deler av OLS3 features: {ols3_features_final}")
    else: print(f"  Valide OLS3 features funnet: {ols3_features_final}")

    # Return all valid features, valid OLS3 features, and all valid features again
    all_needed_final = sorted(list(set(final_features)))
    return all_needed_final, ols3_features_final, all_needed_final


# === Stage 3: Standardization ===
# (Keep rank_standardize_features as is - it operates on the defined features)
def rank_standardize_features(df, features_to_standardize):
    print("\n--- 3. Rank Standardiserer Features ---");
    date_col = 'Date' # Assumes Date column exists and is named 'Date'
    if date_col not in df.columns: print(f"FEIL: Datokolonne ('{date_col}') mangler for standardisering."); return df
    features=[f for f in features_to_standardize if f in df.columns];
    if not features: print("  Ingen features funnet å standardisere."); return df
    print(f"Standardiserer {len(features)} features...")
    def rank_transform(x):
        x_num=pd.to_numeric(x,errors='coerce')
        if x_num.isnull().all(): return x_num # Return NaNs if all are NaN
        # Rank, converting ranks to [-1, 1] range
        r=x_num.rank(pct=True, na_option='keep')
        # Fill remaining NaNs (e.g., from single non-NaN value) with 0 AFTER scaling
        return (r * 2 - 1).fillna(0)

    try:
        # Group by Date and apply the rank transform to each feature column
        # Using transform should be efficient
        df[features] = df.groupby(date_col)[features].transform(rank_transform)
    except Exception as e:
        print(f" ADVARSEL under transform (prøver apply): {e}. Standardisering kan være ufullstendig.");
        # Fallback might be needed for complex cases, but transform is preferred
        try:
            df_s = df.set_index(date_col)
            for col in features:
                 df_s[col] = df_s.groupby(level=0)[col].apply(rank_transform)
            df = df_s.reset_index() # Bring Date back as a column
        except Exception as e2:
             print(f" FEIL under apply: {e2}. Standardisering kan være ufullstendig."); return df # Return potentially partially processed df

    print("Rank standardisering fullført."); return df


# === Stage 4: Data Cleaning (Post-Standardization) ===
def clean_data(df, numeric_features_to_impute, essential_cols_for_dropna, mkt_cap_orig_var):
    """
    Cleans data AFTER standardization.
    1. Replaces inf with NaN in features.
    2. Imputes NaN in FEATURE columns using the overall median.
    3. Drops rows with NaN in ESSENTIAL columns (target, IDs, next_ret, mkt_cap_orig).
    4. Drops rows with non-positive original market cap.
    """
    print("\n--- 4. Renser Data (Post-Standardisering) ---"); initial_rows=len(df)
    features=[f for f in numeric_features_to_impute if f in df.columns];

    if features:
        # Replace inf values first within feature columns
        inf_mask = df[features].isin([np.inf, -np.inf])
        if inf_mask.any().any():
            inf_cols = df[features].columns[inf_mask.any(axis=0)].tolist()
            print(f"  Erstatter +/-inf med NaN i features: {inf_cols}...")
            df[features] = df[features].replace([np.inf, -np.inf], np.nan)

        # Impute NaNs in FEATURES with OVERALL MEDIAN (robust to outliers after standardization)
        # Calculate medians ONLY for the feature columns that have NaNs
        cols_with_nan = df[features].isnull().any()
        features_to_impute_now = cols_with_nan[cols_with_nan].index.tolist()

        if features_to_impute_now:
            print(f"  Imputerer NaNs i {len(features_to_impute_now)} features med overall median...")
            medians = df[features_to_impute_now].median(skipna=True) # Calculate median for each feature column

            # Fill NaNs using the calculated medians
            df[features_to_impute_now] = df[features_to_impute_now].fillna(medians)

            # If median itself is NaN (e.g., all NaNs in a column), fill remaining NaNs with 0
            if medians.isnull().any():
                cols_nan_median = medians[medians.isnull()].index.tolist()
                print(f"  ADVARSEL: Median var NaN for features: {cols_nan_median}. Fyller resterende NaNs i disse kolonnene med 0.")
                df[cols_nan_median] = df[cols_nan_median].fillna(0)
            print(f"  NaNs i features imputert.")
        else:
            print("  Ingen NaNs funnet i features som trenger imputering.")

    # Drop rows with NaNs in ESSENTIAL columns (target, IDs, next return, original market cap)
    # These columns should have been created by the preprocessing script.
    essentials_present = [c for c in essential_cols_for_dropna if c in df.columns]
    if essentials_present:
        rows0 = len(df)
        df = df.dropna(subset=essentials_present)
        dropped_count = rows0 - len(df)
        if dropped_count > 0:
            print(f"  Fjernet {dropped_count} rader pga NaN i essensielle kolonner: {essentials_present}.")
        else:
            print(f"  Ingen rader fjernet pga NaN i essensielle kolonner ({essentials_present}).")
    else:
         print(f"  ADVARSEL: Kunne ikke sjekke NaN i essensielle kolonner (mangler): {[c for c in essential_cols_for_dropna if c not in df.columns]}")


    # Drop rows where original market cap is non-positive (already done in preprocessor, but good safety check)
    if mkt_cap_orig_var in df.columns:
        rows0 = len(df)
        df = df[df[mkt_cap_orig_var] > 0]
        dropped_count = rows0 - len(df)
        if dropped_count > 0:
            print(f"  Fjernet {dropped_count} rader der '{mkt_cap_orig_var}' <= 0 (sikkerhetssjekk).")
    else:
        print(f"ADVARSEL: Kolonne '{mkt_cap_orig_var}' ikke funnet for sjekk av positiv verdi.")

    print(f"Datarensing ferdig. Form: {df.shape}. Totalt fjernet {initial_rows-len(df)} rader i dette steget.");
    if df.empty: print("FEIL: DataFrame er tom etter rensing."); return None
    return df


# === Stage 5: Data Splitting ===
# (Keep get_yearly_rolling_splits as is)
def get_yearly_rolling_splits(df, initial_train_years, val_years, test_years):
    print("\n--- 5. Setter opp Årlige Rullerende Vinduer ---")
    date_col = 'Date' # Assumes standard name
    if date_col not in df.columns: raise ValueError(f"'{date_col}' kolonnen mangler for splitting.")

    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
         try: df[date_col] = pd.to_datetime(df[date_col])
         except Exception as e: raise ValueError(f"Kunne ikke konvertere '{date_col}' til datetime: {e}")

    df['Year']=df[date_col].dt.year; unique_years=sorted(df["Year"].unique()); n_years=len(unique_years)
    print(f"Funnet {n_years} unike år i data ({unique_years[0]}-{unique_years[-1]})")

    min_years_needed = initial_train_years + val_years + test_years
    if n_years < min_years_needed:
        df.drop(columns=['Year'],inplace=True,errors='ignore'); # Clean up temp column
        raise ValueError(f"Ikke nok år ({n_years}) for den spesifiserte splitten (trenger minst {min_years_needed}).")

    first_test_year_idx = initial_train_years + val_years
    if first_test_year_idx >= n_years:
        df.drop(columns=['Year'],inplace=True,errors='ignore');
        raise ValueError(f"Kombinasjonen av initial_train ({initial_train_years}) og validation ({val_years}) dekker alle ({n_years}) eller flere år. Ingen testår igjen.")

    first_test_year = unique_years[first_test_year_idx]
    last_test_start_year = unique_years[n_years - test_years]
    num_windows = last_test_start_year - first_test_year + 1

    if num_windows <= 0:
        df.drop(columns=['Year'],inplace=True,errors='ignore');
        raise ValueError(f"Negativt eller null antall vinduer beregnet ({num_windows}). Sjekk årskonfigurasjon. First test year: {first_test_year}, Last possible test start year: {last_test_start_year}")

    print(f"Genererer {num_windows} rullerende vinduer.")
    print(f"  Første vindu testår: {first_test_year} (slutter {first_test_year+test_years-1})")
    print(f"  Siste vindu testår: {last_test_start_year} (slutter {last_test_start_year+test_years-1})")

    splits_info=[] # Store tuples of (train_idx, val_idx, test_idx, train_dates, val_dates, test_dates)
    for i in range(num_windows):
        test_start_year = first_test_year + i
        test_end_year = test_start_year + test_years - 1
        val_end_year = test_start_year - 1
        val_start_year = val_end_year - val_years + 1
        train_end_year = val_start_year - 1
        train_start_year = unique_years[0] # Train from the beginning

        train_indices = df[(df['Year'] >= train_start_year) & (df['Year'] <= train_end_year)].index
        val_indices = df[(df['Year'] >= val_start_year) & (df['Year'] <= val_end_year)].index
        test_indices = df[(df['Year'] >= test_start_year) & (df['Year'] <= test_end_year)].index

        train_dates = df.loc[train_indices, date_col].agg(['min','max']) if not train_indices.empty else None
        val_dates = df.loc[val_indices, date_col].agg(['min','max']) if not val_indices.empty else None
        test_dates = df.loc[test_indices, date_col].agg(['min','max']) if not test_indices.empty else None

        splits_info.append((
            train_indices, val_indices, test_indices,
            train_dates, val_dates, test_dates,
            train_start_year, train_end_year, val_start_year, val_end_year, test_start_year, test_end_year
        ))

    print("\n--- Split Detaljer per Vindu ---")
    for i,split_data in enumerate(splits_info):
        tr_idx, v_idx, te_idx, tr_d, v_d, t_d, tr_s, tr_e, v_s, v_e, t_s, t_e = split_data
        print(f"  Vindu {i+1}/{num_windows}:")
        print(f"    Train: {tr_s}-{tr_e} ({len(tr_idx)} obs) [{tr_d['min'].date() if tr_d is not None else 'N/A'} -> {tr_d['max'].date() if tr_d is not None else 'N/A'}]")
        print(f"    Val:   {v_s}-{v_e} ({len(v_idx)} obs) [{v_d['min'].date() if v_d is not None else 'N/A'} -> {v_d['max'].date() if v_d is not None else 'N/A'}]")
        print(f"    Test:  {t_s}-{t_e} ({len(te_idx)} obs) [{t_d['min'].date() if t_d is not None else 'N/A'} -> {t_d['max'].date() if t_d is not None else 'N/A'}]")
        yield tr_idx, v_idx, te_idx, tr_d, v_d, t_d # Yield indices and date ranges

    df.drop(columns=['Year'],inplace=True,errors='ignore')


# === Stage 6: Model Evaluation Metrics ===
# (Keep calculate_oos_r2 and calculate_sharpe_of_predictions as is)
def calculate_oos_r2(y_true, y_pred):
    """ Calculates OOS R2 based on Gu, Kelly, Xiu (2020) definition: 1 - SSR/SST0. """
    if y_true is None or y_pred is None: return np.nan
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    if len(y_true) < 2 or len(y_pred) < 2 or len(y_true) != len(y_pred): return np.nan

    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_t = y_true[mask]; y_p = y_pred[mask]

    if len(y_t) < 2: return np.nan
    ss_res = np.sum((y_t - y_p)**2)
    ss_tot = np.sum(y_t**2) # SST0

    if ss_tot < 1e-15:
        return 1.0 if ss_res < 1e-15 else np.nan
    return 1.0 - (ss_res / ss_tot)

def calculate_sharpe_of_predictions(y_pred, annualization_factor=12):
    """ Calculates annualized Sharpe ratio of the *predictions* themselves. """
    if y_pred is None: return np.nan
    y_pred = np.asarray(y_pred)
    if len(y_pred) < 2: return np.nan

    mask = np.isfinite(y_pred)
    y_p = y_pred[mask]
    if len(y_p) < 2: return np.nan
    mean_pred = np.mean(y_p)
    std_pred = np.std(y_p)
    if std_pred < 1e-9: return np.nan
    return (mean_pred / std_pred) * np.sqrt(annualization_factor)


# === Stage 7: Portfolio Analysis ===
# (Keep perform_detailed_portfolio_analysis mostly as is, but ensure column names passed are correct)
# --> Key change: The `original_df_subset` argument will now be the main `df_clean` DataFrame
#     loaded by the pipeline, which contains the preprocessed data including
#     NEXT_RETURN_VARIABLE and MARKET_CAP_ORIG_VARIABLE created by `preprocess_data.py`.
def MDD(returns):
    """ Calculates Maximum Drawdown from a pandas Series of returns. """
    returns = pd.Series(returns).fillna(0) # Ensure it's a series and fill NaNs with 0
    if returns.empty or len(returns) < 2: return np.nan

    cumulative_returns = (1 + returns).cumprod()
    peak = cumulative_returns.cummax()
    drawdown = (cumulative_returns / peak) - 1
    max_drawdown = drawdown.min() # MDD is the minimum value in the drawdown series

    return max_drawdown * 100 if pd.notna(max_drawdown) else np.nan

def perform_detailed_portfolio_analysis(results_df, # Contains Date, Instrument, yhat_*
                                        full_preprocessed_df, # Contains Date, Instrument, Target, NextRawReturn, MktCqpOrig etc.
                                        prediction_cols,
                                        mkt_cap_orig_var, # Name of the original market cap column
                                        next_ret_var,     # Name of the NEXT month's RAW return column
                                        # monthly_rf_var, # Not strictly needed if using excess returns from results_df
                                        filter_small_caps=False, annualization_factor=12,
                                        benchmark_file=None, ff_factor_file=None):
    print("\n--- 7. Detaljert Porteføljeanalyse (Desiler) ---")
    if filter_small_caps: print(">>> Filtrering av små selskaper (basert på market cap ved t) er AKTIVERT for porteføljedanning <<<")

    # --- Input Data Validation ---
    date_col = 'Date' # Standard name
    id_col = 'Instrument' # Standard name
    target_var = config.TARGET_VARIABLE # From config, should match results_df

    required_orig = [date_col, id_col, mkt_cap_orig_var, next_ret_var] # Core needs from preprocessed data
    missing_orig = [c for c in required_orig if c not in full_preprocessed_df.columns]
    if missing_orig: print(f"FEIL: Mangler påkrevde kolonner i full_preprocessed_df: {missing_orig}."); return {},{},{}

    required_res = [date_col, id_col, target_var] + prediction_cols
    missing_res = [c for c in required_res if c not in results_df.columns]
    if missing_res: print(f"FEIL: Mangler påkrevde kolonner i results_df: {missing_res}."); return {},{},{}

    # --- Data Preparation & Merging ---
    print("Forbereder data for porteføljeanalyse...")
    # Ensure correct dtypes and standard columns before merge
    results_df[date_col] = pd.to_datetime(results_df[date_col])
    results_df[id_col] = results_df[id_col].astype(str)
    full_preprocessed_df[date_col] = pd.to_datetime(full_preprocessed_df[date_col])
    full_preprocessed_df[id_col] = full_preprocessed_df[id_col].astype(str)

    # Select necessary columns from the full preprocessed data
    # We need Date, Instrument, the original market cap (t), and the next raw return (t+1)
    df_orig_sub = full_preprocessed_df[required_orig].drop_duplicates(subset=[date_col, id_col], keep='first')

    # Select necessary columns from the results (predictions at t for portfolio formed at t)
    results_sub = results_df[[date_col, id_col] + prediction_cols + [target_var]].drop_duplicates(subset=[date_col, id_col], keep='first')

    # Merge predictions (at time t) with market cap (at t) and NEXT month's return (t+1)
    portfolio_data = pd.merge(results_sub, df_orig_sub, on=[date_col, id_col], how='inner')
    print(f"Data for analyse etter merge: {portfolio_data.shape}")

    # Rename columns for clarity in portfolio context
    # me = market equity (market cap) at time t
    # ret_t+1 = raw return realized in month t+1 (obtained from preprocessed data)
    # target = target variable (excess return t+1, used for sorting checks maybe, but ret_t+1 used for perf)
    portfolio_data = portfolio_data.rename(columns={
        mkt_cap_orig_var: 'me',
        next_ret_var: 'ret_t+1',
        target_var: 'target_ret_t+1' # Rename target for clarity
    })

    # --- Calculate Excess Returns (t+1) for analysis ---
    # We need the risk-free rate corresponding to the ret_t+1 period.
    # The easiest way is to get it from the target_ret_t+1 calculation:
    # target_ret_t+1 = ret_t+1 - rf_t+1  =>  rf_t+1 = ret_t+1 - target_ret_t+1
    if 'ret_t+1' in portfolio_data.columns and 'target_ret_t+1' in portfolio_data.columns:
         portfolio_data['rf_t+1'] = portfolio_data['ret_t+1'] - portfolio_data['target_ret_t+1']
         # Calculate excess return for t+1 (should be very close to target_ret_t+1, good check)
         portfolio_data['excess_ret_t+1'] = portfolio_data['ret_t+1'] - portfolio_data['rf_t+1']
         print("  Beregnet excess_ret_t+1 for porteføljeanalyse.")
    else:
         print("FEIL: Kan ikke beregne excess_ret_t+1 - mangler ret_t+1 eller target_ret_t+1.")
         return {}, {}, {}


    # --- Data Cleaning Post-Merge ---
    # Critical columns needed for decile sorts and return calculations
    crit_cols = prediction_cols + ['excess_ret_t+1', 'ret_t+1', 'me'] # rf_t+1 not strictly needed if excess_ret exists
    initial_rows = len(portfolio_data)
    portfolio_data = portfolio_data.dropna(subset=crit_cols)
    # Ensure valid market cap for weighting and potential filtering
    portfolio_data = portfolio_data[portfolio_data['me'] > 0]
    rows_removed = initial_rows - len(portfolio_data)
    if rows_removed > 0:
        print(f"  Fjernet {rows_removed} rader pga NaNs i kritiske kolonner ({crit_cols}) eller me <= 0.")
    if portfolio_data.empty: print("FEIL: Ingen gyldige data igjen etter sammenslåing og rensing."); return {},{},{}

    # Use Month-Year Period for grouping
    portfolio_data['MonthYear'] = portfolio_data[date_col].dt.to_period('M')

    # --- Decile Sorting and Monthly Returns Calculation ---
    # (The rest of this function remains largely the same, as it operates on the prepared portfolio_data)
    all_monthly_results = []
    monthly_weights_all = [] # To store weights for turnover calculation
    model_names_processed = [] # Track models actually processed
    hl_monthly_dfs_plotting = {} # Store H-L returns for plotting
    long_monthly_dfs_plotting = {} # Store Long-only (D10) returns for plotting

    unique_months = sorted(portfolio_data['MonthYear'].unique())
    print(f"Itererer gjennom {len(unique_months)} måneder for desil-sortering...")

    for month in unique_months:
        monthly_data_full = portfolio_data[portfolio_data['MonthYear'] == month].copy()

        if filter_small_caps:
            mc_cutoff = monthly_data_full['me'].quantile(config.SMALL_FIRM_BOTTOM_PERCENT / 100.0)
            monthly_data_filtered = monthly_data_full[monthly_data_full['me'] > mc_cutoff].copy()
            if monthly_data_filtered.empty and not monthly_data_full.empty: continue
            elif len(monthly_data_filtered) < 10: continue
            else: monthly_data = monthly_data_filtered
        else:
            monthly_data = monthly_data_full

        if len(monthly_data) < 10: continue

        for model_pred_col in prediction_cols:
            model_name = model_pred_col.replace('yhat_', '').upper().replace('_', '-')
            if model_name not in model_names_processed: model_names_processed.append(model_name)

            monthly_data_model = monthly_data.dropna(subset=[model_pred_col]).copy()
            if len(monthly_data_model) < 10: continue

            monthly_data_model['Rank'] = monthly_data_model[model_pred_col].rank(method='first')
            try:
                monthly_data_model['Decile'] = pd.qcut(monthly_data_model['Rank'], 10, labels=False, duplicates='drop') + 1
            except ValueError: continue

            if monthly_data_model['Decile'].nunique() < 2: continue

            monthly_data_model['ew_weights'] = 1 / monthly_data_model.groupby('Decile')[id_col].transform('size')
            mc_sum_decile = monthly_data_model.groupby('Decile')['me'].transform('sum')
            monthly_data_model['vw_weights'] = np.where(mc_sum_decile > 1e-9, monthly_data_model['me'] / mc_sum_decile, 0)

            # Use 'excess_ret_t+1' and 'ret_t+1' directly now
            monthly_data_model['ew_excess_ret'] = monthly_data_model['excess_ret_t+1'] * monthly_data_model['ew_weights']
            monthly_data_model['vw_excess_ret'] = monthly_data_model['excess_ret_t+1'] * monthly_data_model['vw_weights']
            monthly_data_model['ew_raw_ret'] = monthly_data_model['ret_t+1'] * monthly_data_model['ew_weights']
            monthly_data_model['vw_raw_ret'] = monthly_data_model['ret_t+1'] * monthly_data_model['vw_weights']
            monthly_data_model['ew_pred_ret'] = monthly_data_model[model_pred_col] * monthly_data_model['ew_weights']
            monthly_data_model['vw_pred_ret'] = monthly_data_model[model_pred_col] * monthly_data_model['vw_weights']

            weights_m = monthly_data_model[[id_col, 'Decile', 'ew_weights', 'vw_weights']].copy()
            weights_m['Model'] = model_name
            weights_m['MonthYear'] = month
            monthly_weights_all.append(weights_m)

            agg_results = monthly_data_model.groupby('Decile').agg(
                ew_excess_ret = ('ew_excess_ret', 'sum'),
                vw_excess_ret = ('vw_excess_ret', 'sum'),
                ew_raw_ret = ('ew_raw_ret', 'sum'),
                vw_raw_ret = ('vw_raw_ret', 'sum'),
                ew_pred_ret = ('ew_pred_ret', 'sum'),
                vw_pred_ret = ('vw_pred_ret', 'sum'),
                n_stocks = (id_col, 'size')
            ).reset_index()

            agg_results['MonthYear'] = month
            agg_results['Model'] = model_name
            all_monthly_results.append(agg_results)

    if not all_monthly_results: print("FEIL: Ingen månedsresultater ble generert."); return {},{},{}

    # --- Combine Monthly Results and Calculate Turnover ---
    # (Turnover calculation logic remains the same)
    combined_results_df = pd.concat(all_monthly_results).reset_index(drop=True)
    turnover_results = defaultdict(lambda: {'ew': np.nan, 'vw': np.nan, 'long_ew': np.nan, 'long_vw': np.nan})

    if monthly_weights_all:
        all_weights_df = pd.concat(monthly_weights_all).sort_values(['Model', 'MonthYear', id_col])
        print(f"\nBeregner porteføljeomsetning (turnover) for {len(model_names_processed)} modeller...")
        for mn in model_names_processed:
            model_weights = all_weights_df[all_weights_df['Model'] == mn].copy()
            if model_weights.empty: continue

            # H-L Turnover
            long_weights = model_weights[model_weights['Decile'] == 10]
            short_weights = model_weights[model_weights['Decile'] == 1].assign(ew_weights=lambda x: -x.ew_weights, vw_weights=lambda x: -x.vw_weights)
            hl_weights = pd.concat([long_weights, short_weights]).sort_values([id_col, 'MonthYear'])
            hl_weights['ew_w_next'] = hl_weights.groupby(id_col)['ew_weights'].shift(-1).fillna(0)
            hl_weights['vw_w_next'] = hl_weights.groupby(id_col)['vw_weights'].shift(-1).fillna(0)
            hl_weights['trade_ew'] = abs(hl_weights['ew_w_next'] - hl_weights['ew_weights'])
            hl_weights['trade_vw'] = abs(hl_weights['vw_w_next'] - hl_weights['vw_weights'])
            last_month_hl = hl_weights['MonthYear'].max()
            monthly_turnover_hl = hl_weights[hl_weights['MonthYear'] != last_month_hl].groupby('MonthYear').agg(sum_trade_ew=('trade_ew', 'sum'), sum_trade_vw=('trade_vw', 'sum'))
            if not monthly_turnover_hl.empty:
                turnover_results[mn]['ew'] = monthly_turnover_hl['sum_trade_ew'].mean() / 2
                turnover_results[mn]['vw'] = monthly_turnover_hl['sum_trade_vw'].mean() / 2

            # Long-Only Turnover
            long_only_weights = model_weights[model_weights['Decile'] == 10].sort_values([id_col, 'MonthYear'])
            if not long_only_weights.empty:
                 long_only_weights['ew_w_next'] = long_only_weights.groupby(id_col)['ew_weights'].shift(-1).fillna(0)
                 long_only_weights['vw_w_next'] = long_only_weights.groupby(id_col)['vw_weights'].shift(-1).fillna(0)
                 long_only_weights['trade_ew'] = abs(long_only_weights['ew_w_next'] - long_only_weights['ew_weights'])
                 long_only_weights['trade_vw'] = abs(long_only_weights['vw_w_next'] - long_only_weights['vw_weights'])
                 last_month_lo = long_only_weights['MonthYear'].max()
                 monthly_turnover_lo = long_only_weights[long_only_weights['MonthYear'] != last_month_lo].groupby('MonthYear').agg(sum_trade_ew=('trade_ew', 'sum'), sum_trade_vw=('trade_vw', 'sum'))
                 if not monthly_turnover_lo.empty:
                     turnover_results[mn]['long_ew'] = monthly_turnover_lo['sum_trade_ew'].mean() / 2
                     turnover_results[mn]['long_vw'] = monthly_turnover_lo['sum_trade_vw'].mean() / 2
        print("Omsetningsberegning fullført.")
    else: print("Advarsel: Ingen vektdata funnet, kan ikke beregne omsetning.")

    # --- Aggregate Performance and Generate Tables/Plots ---
    # (Calculation and formatting logic remains the same)
    decile_tables = {}
    hl_risk_tables = {}
    long_risk_tables = {}
    performance_summary_list = []

    print(f"\nGenererer ytelsestabeller for {len(model_names_processed)} modeller...")
    for model_name in model_names_processed:
        model_results = combined_results_df[combined_results_df['Model'] == model_name].copy()
        if model_results.empty: continue

        # Decile Performance
        decile_perf = model_results.groupby('Decile').agg(
            ew_pred_mean=('ew_pred_ret', 'mean'), vw_pred_mean=('vw_pred_ret', 'mean'),
            ew_excess_mean=('ew_excess_ret', 'mean'), vw_excess_mean=('vw_excess_ret', 'mean'),
            ew_raw_std=('ew_raw_ret', 'std'), vw_raw_std=('vw_raw_ret', 'std'), # Use raw std for SR
            n_months=('MonthYear', 'nunique'), avg_stocks=('n_stocks','mean')
        ).reset_index()
        decile_perf['ew_sharpe'] = (decile_perf['ew_excess_mean'] / decile_perf['ew_raw_std']) * np.sqrt(annualization_factor)
        decile_perf['vw_sharpe'] = (decile_perf['vw_excess_mean'] / decile_perf['vw_raw_std']) * np.sqrt(annualization_factor)

        # H-L Performance
        hl_stats_df = pd.DataFrame(); hl_monthly = pd.DataFrame()
        if 1 in model_results['Decile'].values and 10 in model_results['Decile'].values:
            long_monthly = model_results[model_results['Decile'] == 10].set_index('MonthYear')
            short_monthly = model_results[model_results['Decile'] == 1].set_index('MonthYear')
            common_index = long_monthly.index.intersection(short_monthly.index)
            if not common_index.empty:
                hl_monthly = pd.DataFrame({
                    'ew_excess_ret_HL': long_monthly.loc[common_index, 'ew_excess_ret'].sub(short_monthly.loc[common_index, 'ew_excess_ret'], fill_value=0),
                    'vw_excess_ret_HL': long_monthly.loc[common_index, 'vw_excess_ret'].sub(short_monthly.loc[common_index, 'vw_excess_ret'], fill_value=0),
                    'ew_raw_ret_HL': long_monthly.loc[common_index, 'ew_raw_ret'].sub(short_monthly.loc[common_index, 'ew_raw_ret'], fill_value=0),
                    'vw_raw_ret_HL': long_monthly.loc[common_index, 'vw_raw_ret'].sub(short_monthly.loc[common_index, 'vw_raw_ret'], fill_value=0),
                    'ew_pred_ret_HL': long_monthly.loc[common_index, 'ew_pred_ret'].sub(short_monthly.loc[common_index, 'ew_pred_ret'], fill_value=0),
                    'vw_pred_ret_HL': long_monthly.loc[common_index, 'vw_pred_ret'].sub(short_monthly.loc[common_index, 'vw_pred_ret'], fill_value=0)
                }).reset_index()
                hl_monthly_dfs_plotting[model_name] = hl_monthly.copy()

                ew_excess_mean_hl = hl_monthly['ew_excess_ret_HL'].mean(); vw_excess_mean_hl = hl_monthly['vw_excess_ret_HL'].mean()
                ew_raw_std_hl = hl_monthly['ew_raw_ret_HL'].std(); vw_raw_std_hl = hl_monthly['vw_raw_ret_HL'].std()
                ew_sharpe_hl = (ew_excess_mean_hl / ew_raw_std_hl) * np.sqrt(annualization_factor) if ew_raw_std_hl > 1e-9 else np.nan
                vw_sharpe_hl = (vw_excess_mean_hl / vw_raw_std_hl) * np.sqrt(annualization_factor) if vw_raw_std_hl > 1e-9 else np.nan
                mdd_ew_hl = MDD(hl_monthly['ew_excess_ret_HL']); mdd_vw_hl = MDD(hl_monthly['vw_excess_ret_HL'])
                # Factor model placeholders
                alpha_ew_hl, tstat_ew_hl, r2_ew_hl = np.nan, np.nan, np.nan
                alpha_vw_hl, tstat_vw_hl, r2_vw_hl = np.nan, np.nan, np.nan

                hl_stats_df = pd.DataFrame({
                    'ew_pred_mean': [hl_monthly['ew_pred_ret_HL'].mean()],'vw_pred_mean': [hl_monthly['vw_pred_ret_HL'].mean()],
                    'ew_excess_mean': [ew_excess_mean_hl],'vw_excess_mean': [vw_excess_mean_hl],
                    'ew_raw_std': [ew_raw_std_hl],'vw_raw_std': [vw_raw_std_hl],
                    'n_months': [len(hl_monthly)], 'ew_sharpe': [ew_sharpe_hl],'vw_sharpe': [vw_sharpe_hl],
                    'avg_stocks': [np.nan], 'Decile': ['H-L']
                })

        model_summary = pd.concat([decile_perf, hl_stats_df], ignore_index=True)
        performance_summary_list.append(model_summary)

        # Formatting Functions (keep as is)
        def format_decile_table(summary_df, weight_scheme):
            prefix = 'ew_' if weight_scheme == 'EW' else 'vw_'
            cols_map = {f'{prefix}pred_mean': 'Pred', f'{prefix}excess_mean': 'Avg Ex Ret', f'{prefix}raw_std': 'SD (Raw Ret)', f'{prefix}sharpe': 'Ann SR', 'avg_stocks': 'Avg N'}
            relevant_cols = [c for c in cols_map if c in summary_df.columns]
            if not relevant_cols or 'Decile' not in summary_df.columns: return pd.DataFrame()
            sub_df = summary_df[['Decile'] + relevant_cols].rename(columns=cols_map).copy().set_index('Decile')
            for col in ['Pred', 'Avg Ex Ret', 'SD (Raw Ret)']:
                 if col in sub_df.columns: sub_df[col] = pd.to_numeric(sub_df[col], errors='coerce') * 100
            if 'Ann SR' in sub_df.columns: sub_df['Ann SR'] = pd.to_numeric(sub_df['Ann SR'], errors='coerce')
            if 'Avg N' in sub_df.columns: sub_df['Avg N'] = pd.to_numeric(sub_df['Avg N'], errors='coerce')
            def map_idx(x): return 'Low (L)' if str(x) == '1' else ('High (H)' if str(x) == '10' else str(x))
            sub_df.index = sub_df.index.map(map_idx)
            desired_order = ['Low (L)','2','3','4','5','6','7','8','9','High (H)','H-L']
            sub_df = sub_df.reindex([i for i in desired_order if i in sub_df.index])
            final_cols = [c for c in ['Pred', 'Avg Ex Ret', 'SD (Raw Ret)', 'Ann SR', 'Avg N'] if c in sub_df.columns]
            sub_df_formatted = sub_df[final_cols].copy()
            for col in ['Pred', 'Avg Ex Ret', 'SD (Raw Ret)']:
                 if col in sub_df_formatted.columns: sub_df_formatted[col] = sub_df_formatted[col].map('{:.2f}%'.format).replace('nan%','N/A')
            if 'Ann SR' in sub_df_formatted.columns: sub_df_formatted['Ann SR'] = sub_df_formatted['Ann SR'].map('{:.2f}'.format).replace('nan','N/A')
            if 'Avg N' in sub_df_formatted.columns: sub_df_formatted['Avg N'] = sub_df_formatted['Avg N'].map('{:.0f}'.format).replace('nan','N/A')
            return sub_df_formatted[final_cols]

        def format_risk_table(data_dict, table_index):
             df_risk = pd.DataFrame(data_dict, index=table_index)
             for idx in df_risk.index:
                  is_percent = '%' in idx; num_decimals = 2 if is_percent else 3; suffix = '%' if is_percent else ''
                  try: df_risk.loc[idx] = df_risk.loc[idx].map(f'{{:.{num_decimals}f}}'.format).astype(str) + suffix
                  except (ValueError, TypeError): df_risk.loc[idx] = df_risk.loc[idx].apply(lambda x: f'{float(x):.{num_decimals}f}{suffix}' if pd.notna(x) and isinstance(x,(int,float)) else 'N/A')
                  df_risk.loc[idx] = df_risk.loc[idx].replace(['nan%', 'nan', ''], 'N/A', regex=False)
             return df_risk

        # Generate and Print Decile Tables
        ew_table = format_decile_table(model_summary, 'EW'); decile_tables[f'{model_name}_EW'] = ew_table
        vw_table = format_decile_table(model_summary, 'VW'); decile_tables[f'{model_name}_VW'] = vw_table
        print(f"\n--- Ytelsestabell (Desiler): {model_name} - EW ---"); print(ew_table)
        print(f"\n--- Ytelsestabell (Desiler): {model_name} - VW ---"); print(vw_table)

        # Generate and Print H-L Risk/Performance Table
        if not hl_stats_df.empty and not hl_monthly.empty:
            hl_res = hl_stats_df.iloc[0]
            turnover_ew_hl = turnover_results.get(model_name, {}).get('ew', np.nan)
            turnover_vw_hl = turnover_results.get(model_name, {}).get('vw', np.nan)
            max_loss_1m_ew_hl = hl_monthly['ew_excess_ret_HL'].min() * 100 if not hl_monthly.empty else np.nan
            max_loss_1m_vw_hl = hl_monthly['vw_excess_ret_HL'].min() * 100 if not hl_monthly.empty else np.nan
            risk_idx_hl = ["Mean Excess Return [%]", 'Std Dev (Raw) [%]', "Ann. Sharpe Ratio", "Max Drawdown (Excess) [%]", "Max 1M Loss (Excess) [%]", "Avg Monthly Turnover [%]", "Factor Model Alpha [%]", "t(Alpha)", "Factor Model Adj R2", "Info Ratio"]
            ew_data_hl = {f'{model_name} H-L EW': [hl_res.get('ew_excess_mean', np.nan) * 100, hl_res.get('ew_raw_std', np.nan) * 100, hl_res.get('ew_sharpe', np.nan), abs(mdd_ew_hl), max_loss_1m_ew_hl, turnover_ew_hl * 100, alpha_ew_hl, tstat_ew_hl, r2_ew_hl, np.nan]}
            vw_data_hl = {f'{model_name} H-L VW': [hl_res.get('vw_excess_mean', np.nan) * 100, hl_res.get('vw_raw_std', np.nan) * 100, hl_res.get('vw_sharpe', np.nan), abs(mdd_vw_hl), max_loss_1m_vw_hl, turnover_vw_hl * 100, alpha_vw_hl, tstat_vw_hl, r2_vw_hl, np.nan]}
            ew_chart_hl = format_risk_table(ew_data_hl, risk_idx_hl); hl_risk_tables[f'{model_name}_EW'] = ew_chart_hl
            vw_chart_hl = format_risk_table(vw_data_hl, risk_idx_hl); hl_risk_tables[f'{model_name}_VW'] = vw_chart_hl
            print(f"\n--- H-L Portefølje Risk/Performance ({model_name} EW) ---"); print(ew_chart_hl)
            print(f"\n--- H-L Portefølje Risk/Performance ({model_name} VW) ---"); print(vw_chart_hl)

        # Generate and Print Long-Only (Decile 10) Risk/Performance Table
        long_res_row = decile_perf[decile_perf['Decile'] == 10]
        if not long_res_row.empty:
            long_res = long_res_row.iloc[0]
            long_monthly = model_results[model_results['Decile'] == 10].set_index('MonthYear')
            if not long_monthly.empty: long_monthly_dfs_plotting[model_name] = long_monthly.reset_index()
            mdd_ew_long = MDD(long_monthly['ew_excess_ret']) if not long_monthly.empty else np.nan
            mdd_vw_long = MDD(long_monthly['vw_excess_ret']) if not long_monthly.empty else np.nan
            max_loss_1m_ew_long = long_monthly['ew_excess_ret'].min() * 100 if not long_monthly.empty else np.nan
            max_loss_1m_vw_long = long_monthly['vw_excess_ret'].min() * 100 if not long_monthly.empty else np.nan
            turnover_ew_long = turnover_results.get(model_name, {}).get('long_ew', np.nan)
            turnover_vw_long = turnover_results.get(model_name, {}).get('long_vw', np.nan)
            alpha_long_ew, tstat_long_ew, r2_long_ew = np.nan, np.nan, np.nan # Factor placeholders
            alpha_long_vw, tstat_long_vw, r2_long_vw = np.nan, np.nan, np.nan # Factor placeholders
            risk_idx_long = ["Mean Excess Return [%]", 'Std Dev (Raw) [%]', "Ann. Sharpe Ratio", "Max Drawdown (Excess) [%]", "Max 1M Loss (Excess) [%]", "Avg Monthly Turnover [%]", "Factor Model Alpha [%]", "t(Alpha)", "Factor Model Adj R2", "Info Ratio"]
            ew_data_long = {f'{model_name} Long EW': [long_res.get('ew_excess_mean', np.nan) * 100, long_res.get('ew_raw_std', np.nan) * 100, long_res.get('ew_sharpe', np.nan), abs(mdd_ew_long), max_loss_1m_ew_long, turnover_ew_long * 100, alpha_long_ew, tstat_long_ew, r2_long_ew, np.nan]}
            vw_data_long = {f'{model_name} Long VW': [long_res.get('vw_excess_mean', np.nan) * 100, long_res.get('vw_raw_std', np.nan) * 100, long_res.get('vw_sharpe', np.nan), abs(mdd_vw_long), max_loss_1m_vw_long, turnover_vw_long * 100, alpha_long_vw, tstat_long_vw, r2_long_vw, np.nan]}
            ew_chart_long = format_risk_table(ew_data_long, risk_idx_long); long_risk_tables[f'{model_name}_EW'] = ew_chart_long
            vw_chart_long = format_risk_table(vw_data_long, risk_idx_long); long_risk_tables[f'{model_name}_VW'] = vw_chart_long
            print(f"\n--- Long-Only (D10) Risk/Performance ({model_name} EW) ---"); print(ew_chart_long)
            print(f"\n--- Long-Only (D10) Risk/Performance ({model_name} VW) ---"); print(vw_chart_long)
        else: print(f"  Advarsel: Ingen data for Desil 10 funnet for modell {model_name}.")

    # Plotting Cumulative Returns (keep as is)
    fig_hl, ax_hl = plt.subplots(figsize=(14, 7)); plotted_hl = 0
    sorted_models_hl = sorted(hl_monthly_dfs_plotting.keys())
    for model_name in sorted_models_hl:
        df_hl = hl_monthly_dfs_plotting[model_name]
        if 'MonthYear' in df_hl.columns and not df_hl.empty:
             df_hl['PlotDate'] = df_hl['MonthYear'].dt.to_timestamp() if pd.api.types.is_period_dtype(df_hl['MonthYear']) else pd.to_datetime(df_hl['MonthYear'])
             df_hl = df_hl.set_index('PlotDate').sort_index()
             if 'ew_excess_ret_HL' in df_hl.columns:
                  ret_ew = df_hl['ew_excess_ret_HL'].dropna()
                  if not ret_ew.empty: (1 + ret_ew).cumprod().plot(ax=ax_hl, label=f'{model_name} H-L EW'); plotted_hl += 1
             if 'vw_excess_ret_HL' in df_hl.columns:
                  ret_vw = df_hl['vw_excess_ret_HL'].dropna()
                  if not ret_vw.empty: (1 + ret_vw).cumprod().plot(ax=ax_hl, label=f'{model_name} H-L VW', linestyle='--'); plotted_hl += 1
    if plotted_hl > 0:
        ax_hl.set_title('Kumulativ Excess Avkastning (H-L Portefølje, t+1)'); ax_hl.set_ylabel('Kumulativ Verdi (Log Skala)'); ax_hl.set_xlabel('Dato'); ax_hl.set_yscale('log'); ax_hl.legend(loc='center left', bbox_to_anchor=(1, 0.5)); ax_hl.grid(True, which='both', linestyle='--', linewidth=0.5); fig_hl.tight_layout(rect=[0, 0, 0.85, 1]); plt.show()
    else: plt.close(fig_hl)

    fig_long, ax_long = plt.subplots(figsize=(14, 7)); plotted_long = 0
    sorted_models_long = sorted(long_monthly_dfs_plotting.keys())
    for model_name in sorted_models_long:
        df_long = long_monthly_dfs_plotting[model_name]
        if 'MonthYear' in df_long.columns and not df_long.empty:
             df_long['PlotDate'] = df_long['MonthYear'].dt.to_timestamp() if pd.api.types.is_period_dtype(df_long['MonthYear']) else pd.to_datetime(df_long['MonthYear'])
             df_long = df_long.set_index('PlotDate').sort_index()
             if 'ew_excess_ret' in df_long.columns:
                  ret_ew = df_long['ew_excess_ret'].dropna()
                  if not ret_ew.empty: (1 + ret_ew).cumprod().plot(ax=ax_long, label=f'{model_name} Long EW'); plotted_long += 1
             if 'vw_excess_ret' in df_long.columns:
                  ret_vw = df_long['vw_excess_ret'].dropna()
                  if not ret_vw.empty: (1 + ret_vw).cumprod().plot(ax=ax_long, label=f'{model_name} Long VW', linestyle='--'); plotted_long += 1
    if plotted_long > 0:
        ax_long.set_title('Kumulativ Excess Avkastning (Long-Only Portefølje [D10], t+1)'); ax_long.set_ylabel('Kumulativ Verdi (Log Skala)'); ax_long.set_xlabel('Dato'); ax_long.set_yscale('log'); ax_long.legend(loc='center left', bbox_to_anchor=(1, 0.5)); ax_long.grid(True, which='both', linestyle='--', linewidth=0.5); fig_long.tight_layout(rect=[0, 0, 0.85, 1]); plt.show()
    else: plt.close(fig_long)

    print("--- Detaljert Porteføljeanalyse Fullført ---")
    return decile_tables, hl_risk_tables, long_risk_tables


# === Stage 8: Variable Importance ===
# (Keep calculate_variable_importance as is)
def calculate_variable_importance(model_name, fitted_model, X_eval, y_eval, features, base_r2_is, vi_method='permutation_zero', model_params=None):
    start_vi = time.time()
    if vi_method != 'permutation_zero': print(f"    FEIL: VI metode '{vi_method}' støttes ikke."); return pd.DataFrame()
    if fitted_model is None: print(f"    FEIL: Ingen modell for VI."); return pd.DataFrame()
    if pd.isna(base_r2_is): print(f"    ADVARSEL: Basis IS R2 NaN."); return pd.DataFrame({'Feature': features, 'Importance': 0.0})
    if len(features)==0 or X_eval.shape[0]==0 or y_eval.shape[0]==0 or X_eval.shape[1]!=len(features): print(f"    FEIL: Ugyldige VI data dim."); return pd.DataFrame({'Feature': features, 'Importance': 0.0})

    importance_results = {}
    y_eval_finite = y_eval[np.isfinite(y_eval)]
    ss_tot_zero = np.sum(y_eval_finite**2)
    if ss_tot_zero < 1e-15: return pd.DataFrame({'Feature': features, 'Importance': 0.0})

    params_retrain = model_params if model_params else {}
    if model_name == 'ENET' and hasattr(fitted_model, 'alpha_'): params_retrain = {'alpha': fitted_model.alpha_, 'l1_ratio': fitted_model.l1_ratio_, **config.MODEL_PARAMS.get('ENET', {})}
    elif model_name == 'PLS' and hasattr(fitted_model, 'n_components'): params_retrain = {'n_components': fitted_model.n_components, 'scale': False}
    elif model_name == 'PCR' and hasattr(fitted_model, 'named_steps'):
        try: params_retrain = {'n_components': fitted_model.named_steps['pca'].n_components_}
        except KeyError: params_retrain = {'n_components': 1}
    elif model_name in ['GLM_H', 'RF', 'GBRT_H'] and hasattr(fitted_model, 'get_params'):
        params_retrain = fitted_model.get_params()
        if model_params: params_retrain.update(model_params) # Use optimal params if provided
    elif model_name == 'OLS3H': params_retrain = {k: v for k, v in config.MODEL_PARAMS.get('OLS3H', {}).items() if k != 'M'}

    for idx, feat_name in enumerate(features):
        X_permuted = X_eval.copy(); X_permuted[:, idx] = 0
        permuted_model = None; permuted_preds = None; permuted_r2 = -np.inf
        try:
            if model_name == 'OLS': permuted_model = LinearRegression(fit_intercept=True).fit(X_permuted, y_eval)
            elif model_name == 'OLS3H' and sm:
                 X_perm_c = sm.add_constant(X_permuted)
                 permuted_model_rlm = sm.RLM(y_eval, X_perm_c, M=sm.robust.norms.HuberT())
                 permuted_model = permuted_model_rlm.fit(**params_retrain)
                 permuted_preds = permuted_model.predict(X_perm_c)
            elif model_name == 'PLS': permuted_model = PLSRegression(**params_retrain).fit(X_permuted, y_eval)
            elif model_name == 'PCR': permuted_model = Pipeline([('pca', PCA(n_components=params_retrain.get('n_components', 1))), ('lr', LinearRegression())]).fit(X_permuted, y_eval)
            elif model_name == 'ENET': permuted_model = ElasticNet(**params_retrain, fit_intercept=True).fit(X_permuted, y_eval)
            elif model_name == 'GLM_H': permuted_model = HuberRegressor(**params_retrain, fit_intercept=True).fit(X_permuted, y_eval)
            elif model_name == 'RF': permuted_model = RandomForestRegressor(**params_retrain).fit(X_permuted, y_eval)
            elif model_name == 'GBRT_H': permuted_model = GradientBoostingRegressor(**params_retrain).fit(X_permuted, y_eval)

            if permuted_model and model_name != 'OLS3H': permuted_preds = permuted_model.predict(X_permuted).flatten()
            if permuted_preds is not None:
                preds_finite = permuted_preds[np.isfinite(y_eval)]
                if len(preds_finite) == len(y_eval_finite) and np.all(np.isfinite(preds_finite)):
                    ss_res_permuted = np.sum((y_eval_finite - preds_finite)**2)
                    permuted_r2 = 1.0 - (ss_res_permuted / ss_tot_zero)
        except Exception as e: print(f"    ADVARSEL: Unntak VI '{feat_name}' i {model_name}: {e}")
        reduction = base_r2_is - permuted_r2
        importance_results[feat_name] = max(0, reduction) if pd.notna(reduction) else 0.0

    if not importance_results: return pd.DataFrame({'Feature': features, 'Importance': 0.0})
    importance_df = pd.DataFrame(importance_results.items(), columns=['Feature', 'R2_Reduction'])
    total_reduction = importance_df['R2_Reduction'].sum()
    importance_df['Importance'] = importance_df['R2_Reduction'] / total_reduction if total_reduction > 1e-9 else 0.0
    return importance_df[['Feature', 'Importance']]


# === Stage 9: Complexity Plotting ===
# (Keep plot_time_varying_complexity as is)
def plot_time_varying_complexity(model_metrics, complexity_params_to_plot):
     print("\n--- 9. Plotter Tidsvarierende Modellkompleksitet ---")
     plotted_any = False
     for model_name, param_keys in complexity_params_to_plot.items():
         if model_name not in model_metrics: continue
         print(f"\n  --- Modell: {model_name} ---")
         model_data = model_metrics[model_name]
         for param_key in param_keys:
             if param_key in model_data:
                 values = model_data[param_key]
                 valid_data = [(i + 1, v) for i, v in enumerate(values) if v is not None and pd.notna(v)]
                 if valid_data:
                     plotted_any = True
                     windows, param_values = zip(*valid_data)
                     param_label = param_key.replace('optim_', '').replace('_', ' ').title()
                     data_table = pd.DataFrame({param_label: param_values}, index=pd.Index(windows, name='Vindu Nr.'))
                     print(f"    Optimal {param_label} per Vindu:"); print(data_table.round(4))
                     plt.figure(figsize=(10, 5))
                     plt.plot(windows, param_values, marker='o', linestyle='-')
                     plot_title = f"Tidsvariasjon i Optimal {param_label} for {model_name}"
                     y_axis_label = f"Optimal {param_label}"
                     if 'alpha' in param_key.lower() or 'lambda' in param_key.lower():
                          if all(v > 1e-9 for v in param_values): plt.yscale('log'); y_axis_label += " (Log Skala)"
                          else: print(f"    Advarsel: Kan ikke bruke log-skala for {param_label}.")
                     plt.xlabel("Vindu Nr."); plt.ylabel(y_axis_label); plt.title(plot_title); plt.grid(True, which='both', linestyle='--', linewidth=0.5); plt.tight_layout(); plt.show()
                 else: print(f"    Ingen gyldige verdier funnet for '{param_label}' for {model_name}.")
             else: print(f"    Metrikk '{param_key}' ikke funnet for {model_name}.")
     if not plotted_any: print("  Ingen kompleksitetsparametere å plotte.")


# === Stage 10: Reporting & Saving ===
# (Keep create_summary_table and save_results as is)
def create_summary_table(model_metrics, annualization_factor=12):
    print("\n--- 10a. Lager Oppsummerende Resultattabell ---")
    summary_data = []
    model_order = ['OLS','OLS3H','PLS','PCR','ENET','GLM_H','RF','GBRT_H','NN1','NN2','NN3','NN4','NN5']
    models_in_results = list(model_metrics.keys())
    models_sorted = [m for m in model_order if m in models_in_results] + \
                    [m for m in sorted(models_in_results) if m not in model_order]
    if not models_sorted: print("Ingen modelldata funnet."); return pd.DataFrame()

    for model_name in models_sorted:
        metrics = model_metrics[model_name]
        avg_is_r2 = np.nanmean(metrics.get('is_r2_train_val', [])) if metrics.get('is_r2_train_val') else np.nan
        avg_oos_r2 = np.nanmean(metrics.get('oos_r2', [])) if metrics.get('oos_r2') else np.nan
        avg_oos_sharpe = np.nanmean(metrics.get('oos_sharpe', [])) if metrics.get('oos_sharpe') else np.nan
        overall_oos_r2_gu = metrics.get('oos_r2_overall_gu', np.nan)
        avg_optim_params_str = ""
        optim_parts = []
        for k, v in metrics.items():
            if k.startswith('optim_') and v:
                 numeric_v = [item for item in v if isinstance(item, (int, float, np.number)) and pd.notna(item)]
                 if numeric_v:
                     try:
                          mean_val = np.nanmean(numeric_v)
                          if not np.isnan(mean_val): optim_parts.append(f"{k.replace('optim_', '')}={mean_val:.2g}")
                     except Exception as e_mean: print(f"  Advarsel: mean calc error for {k} {model_name}: {e_mean}.")
        avg_optim_params_str = ", ".join(optim_parts)

        summary_data.append({
            'Modell': model_name,
            'Avg IS R² (%)': avg_is_r2 * 100 if pd.notna(avg_is_r2) else np.nan,
            'Avg Window OOS R² (%)': avg_oos_r2 * 100 if pd.notna(avg_oos_r2) else np.nan,
            'Overall OOS R² (%)': overall_oos_r2_gu * 100 if pd.notna(overall_oos_r2_gu) else np.nan,
            'Avg Pred Sharpe (OOS)': avg_oos_sharpe if pd.notna(avg_oos_sharpe) else np.nan,
            'Avg Optim Params': avg_optim_params_str
        })
    if not summary_data: print("Ingen data å inkludere."); return pd.DataFrame()
    summary_df = pd.DataFrame(summary_data).set_index('Modell')
    print("\n--- Oppsummeringstabell ---")
    print(summary_df.to_string(float_format=lambda x: f"{x:.3f}" if pd.notna(x) else "N/A", na_rep="N/A"))
    return summary_df

def save_results(output_dir, subset_label, results_dict):
    print(f"\n--- 10b. Lagrer Resultater for Subset: {subset_label} ---")
    subset_dir = os.path.join(output_dir, subset_label)
    try:
        if not os.path.exists(subset_dir): os.makedirs(subset_dir); print(f"  Opprettet mappe: {subset_dir}")
    except OSError as e: print(f"  FEIL: Kunne ikke opprette mappe {subset_dir}: {e}"); return

    for name, data in results_dict.items():
        base_filename = os.path.join(subset_dir, f"{name}")
        try:
            if isinstance(data, pd.DataFrame):
                if not data.empty: filename = f"{base_filename}.csv"; data.to_csv(filename); print(f"  -> Lagret DataFrame: {filename}")
                else: print(f"  -> Hoppet over tom DataFrame: {name}")
            elif isinstance(data, dict):
                saved_sub = False
                for sub_name, sub_data in data.items():
                     if isinstance(sub_data, pd.DataFrame):
                         if not sub_data.empty: sub_filename = f"{base_filename}_{sub_name}.csv"; sub_data.to_csv(sub_filename); print(f"  -> Lagret Dict->DataFrame: {sub_filename}"); saved_sub = True
                     elif isinstance(sub_data, dict): print(f"  -> Hoppet over Dict->Dict: {name}_{sub_name}")
                     else: print(f"  -> Hoppet over ukjent datatype i dict: {name}_{sub_name} (Type: {type(sub_data)})")
            else: print(f"  -> Hoppet over ukjent datatype: {name} (Type: {type(data)})")
        except Exception as e: print(f"  FEIL under lagring av '{name}' til '{subset_dir}': {e}"); traceback.print_exc(limit=1)
    print(f"--- Lagring for {subset_label} fullført (eller forsøkt) ---")

