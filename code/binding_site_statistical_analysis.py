
import pandas as pd
import numpy as np
from scipy.stats import fisher_exact, chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 11

print("="*80)
print("Binding Site Statistical Analysis")
print("="*80)

# --- Load data ---
df = pd.read_csv("./data/output_from_KNIME/binding-site.csv")
print(f"\nDataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# --- Data preparation ---
df_filtered = df.copy()
df_filtered = df_filtered[df_filtered["minimum_bound_fraction"].notna()]
df_filtered = df_filtered[df_filtered["binding_site"].notna()]
df_filtered["selectivity-label"] = df_filtered["selectivity-label"].astype(str)

print(f"\nFiltered dataset shape: {df_filtered.shape}")

# Using the same threshold as the original notebook
strong_selective = df_filtered[
    (df_filtered["minimum_bound_fraction"] > 0.4) &
    (df_filtered["selectivity-label"] == "selective")
]

others = df_filtered[~(
    (df_filtered["minimum_bound_fraction"] > 0.4) &
    (df_filtered["selectivity-label"] == "selective")
)]

print(f"\n--- Compound Categories ---")
print(f"Strong & Selective: n = {len(strong_selective)}")
print(f"Others: n = {len(others)}")
print(f"Total: n = {len(df_filtered)}")

# --- Check all binding sites ---
binding_sites = df_filtered["binding_site"].unique()
print(f"\n--- Binding Sites ---")
print(f"Unique binding sites: {sorted(binding_sites)}")

# Count distribution
print(f"\n--- Distribution in Strong & Selective ---")
for site in sorted(binding_sites):
    count = (strong_selective["binding_site"] == site).sum()
    pct = count / len(strong_selective) * 100 if len(strong_selective) > 0 else 0
    print(f"  {site}: {count} ({pct:.1f}%)")

print(f"\n--- Distribution in Others ---")
for site in sorted(binding_sites):
    count = (others["binding_site"] == site).sum()
    pct = count / len(others) * 100 if len(others) > 0 else 0
    print(f"  {site}: {count} ({pct:.1f}%)")

# --- Statistical tests for each binding site ---
print("\n" + "="*80)
print("Fisher's Exact Test for Each Binding Site")
print("="*80)

results = []

for site in sorted(binding_sites):
    # Create 2x2 contingency table
    # Rows: Strong&Selective vs Others
    # Cols: This site vs Other sites

    strong_this_site = (strong_selective["binding_site"] == site).sum()
    strong_other_sites = (strong_selective["binding_site"] != site).sum()

    others_this_site = (others["binding_site"] == site).sum()
    others_other_sites = (others["binding_site"] != site).sum()

    table = np.array([
        [strong_this_site, strong_other_sites],
        [others_this_site, others_other_sites]
    ])

    # Fisher's exact test
    oddsratio, p_value = fisher_exact(table)

    # Effect size: Odds Ratio interpretation
    if oddsratio > 1:
        effect_direction = "enriched in strong&selective"
    elif oddsratio < 1:
        effect_direction = "depleted in strong&selective"
    else:
        effect_direction = "no difference"

    # Odds ratio interpretation
    if oddsratio == 0 or np.isinf(oddsratio):
        or_interpretation = "extreme"
    elif oddsratio > 3 or oddsratio < 0.33:
        or_interpretation = "large"
    elif oddsratio > 2 or oddsratio < 0.5:
        or_interpretation = "moderate"
    elif oddsratio > 1.5 or oddsratio < 0.67:
        or_interpretation = "small"
    else:
        or_interpretation = "negligible"

    print(f"\n{site}:")
    print(f"  Contingency table:")
    print(f"                   {site:15s} Other sites")
    print(f"  Strong&Selective {strong_this_site:5d}           {strong_other_sites:5d}")
    print(f"  Others           {others_this_site:5d}           {others_other_sites:5d}")
    print(f"  Odds Ratio: {oddsratio:.3f}")
    print(f"  P-value: {p_value:.6f}")
    print(f"  Direction: {effect_direction}")
    print(f"  Effect size: {or_interpretation}")

    if p_value <= 0.001:
        sig_label = "*** (p ≤ 0.001)"
    elif p_value <= 0.01:
        sig_label = "** (p ≤ 0.01)"
    elif p_value <= 0.05:
        sig_label = "* (p ≤ 0.05)"
    else:
        sig_label = "n.s. (p > 0.05)"
    print(f"  Significance: {sig_label}")

    results.append({
        'Binding_Site': site,
        'Strong_Selective_n': strong_this_site,
        'Others_n': others_this_site,
        'Strong_Selective_pct': strong_this_site / len(strong_selective) * 100 if len(strong_selective) > 0 else 0,
        'Others_pct': others_this_site / len(others) * 100 if len(others) > 0 else 0,
        'Odds_Ratio': oddsratio,
        'P_value': p_value,
        'Effect_direction': effect_direction,
        'Effect_size': or_interpretation,
        'Significance': sig_label
    })

# --- Overall Chi-square test ---
print("\n" + "="*80)
print("Overall Chi-Square Test (All Binding Sites)")
print("="*80)

# Create contingency table for all sites
site_counts_strong = [
    (strong_selective["binding_site"] == site).sum()
    for site in sorted(binding_sites)
]
site_counts_others = [
    (others["binding_site"] == site).sum()
    for site in sorted(binding_sites)
]

overall_table = np.array([site_counts_strong, site_counts_others])

chi2, p_overall, dof, expected = chi2_contingency(overall_table)

print(f"\nContingency table:")
print(f"{'Binding Site':<20} Strong&Selective  Others")
for i, site in enumerate(sorted(binding_sites)):
    print(f"{site:<20} {site_counts_strong[i]:6d}          {site_counts_others[i]:6d}")

print(f"\nChi-square statistic: {chi2:.3f}")
print(f"Degrees of freedom: {dof}")
print(f"P-value: {p_overall:.6f}")

if p_overall <= 0.001:
    print("Significance: *** (p ≤ 0.001)")
elif p_overall <= 0.01:
    print("Significance: ** (p ≤ 0.01)")
elif p_overall <= 0.05:
    print("Significance: * (p ≤ 0.05)")
else:
    print("Significance: n.s. (p > 0.05)")

print("\nInterpretation:")
if p_overall < 0.05:
    print("The distribution of binding sites differs significantly between")
    print("strong&selective compounds and others.")
else:
    print("No significant overall difference in binding site distribution.")

# --- Save results ---
results_df = pd.DataFrame(results)
output_path = "./data/output_from_code/S1_binding_site_test_results.csv"
results_df.to_csv(output_path, index=False)

print("\n" + "="*80)
print(f"Results saved to: {output_path}")
print("="*80)

# --- Summary table ---
print("\n\nSUMMARY TABLE:")
print("="*80)
summary_cols = ['Binding_Site', 'Strong_Selective_n', 'Strong_Selective_pct',
                'Odds_Ratio', 'P_value', 'Effect_size', 'Significance']
print(results_df[summary_cols].to_string(index=False))

# --- Identify significant sites ---
print("\n\nSIGNIFICANT FINDINGS (p < 0.05):")
print("="*80)
sig_results = results_df[results_df['P_value'] < 0.05]
if len(sig_results) > 0:
    for _, row in sig_results.iterrows():
        print(f"\n{row['Binding_Site']}:")
        print(f"  Odds Ratio: {row['Odds_Ratio']:.3f}")
        print(f"  P-value: {row['P_value']:.6f}")
        print(f"  {row['Effect_direction']}")
        print(f"  Effect size: {row['Effect_size']}")
else:
    print("No significant differences found (all p > 0.05)")

print("\n" + "="*80)
print("Analysis completed!")
print("="*80)
