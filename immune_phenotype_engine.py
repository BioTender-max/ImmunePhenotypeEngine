import numpy as np; np.random.seed(42)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
import shutil, os

# ── Parameters ──────────────────────────────────────────────────────────────
N_SAMPLES = 100
N_MARKERS = 40
N_CELLS   = 50000
N_CLUSTERS = 12   # immune cell subsets

# ── Marker definitions ────────────────────────────────────────────────────────
markers = [
    'CD3', 'CD4', 'CD8', 'CD19', 'CD20', 'CD56', 'CD16', 'CD14', 'CD11b',
    'CD11c', 'CD25', 'CD127', 'CD45RA', 'CD45RO', 'CCR7', 'CXCR5', 'PD1',
    'TIM3', 'LAG3', 'CTLA4', 'FOXP3', 'TBET', 'GATA3', 'RORGT', 'BCL6',
    'Ki67', 'IFNG', 'TNF', 'IL2', 'IL4', 'IL17', 'GRANZB', 'PERF',
    'CD38', 'HLA-DR', 'CD27', 'CD28', 'CD57', 'CD69', 'CD44'
]
N_MARKERS = len(markers)

# ── Cluster definitions (immune subsets) ─────────────────────────────────────
cluster_names = [
    'CD4 Naive', 'CD4 Tcm', 'CD4 Tem', 'Treg',
    'CD8 Naive', 'CD8 Tcm', 'CD8 Tem', 'CD8 Exhausted',
    'NK cell', 'B cell', 'Monocyte', 'DC'
]

# Cluster marker expression profiles (mean expression per cluster)
cluster_profiles = np.zeros((N_CLUSTERS, N_MARKERS))
# CD4 Naive: CD3+, CD4+, CD45RA+, CCR7+
cluster_profiles[0, [0,1,12,14]] = [4, 4, 4, 4]
# CD4 Tcm: CD3+, CD4+, CD45RO+, CCR7+
cluster_profiles[1, [0,1,13,14]] = [4, 4, 4, 3]
# CD4 Tem: CD3+, CD4+, CD45RO+
cluster_profiles[2, [0,1,13]] = [4, 4, 4]
# Treg: CD3+, CD4+, CD25+, FOXP3+
cluster_profiles[3, [0,1,10,20]] = [4, 4, 4, 4]
# CD8 Naive: CD3+, CD8+, CD45RA+, CCR7+
cluster_profiles[4, [0,2,12,14]] = [4, 4, 4, 4]
# CD8 Tcm: CD3+, CD8+, CD45RO+, CCR7+
cluster_profiles[5, [0,2,13,14]] = [4, 4, 4, 3]
# CD8 Tem: CD3+, CD8+, CD45RO+, GRANZB+
cluster_profiles[6, [0,2,13,31]] = [4, 4, 4, 3]
# CD8 Exhausted: CD3+, CD8+, PD1+, TIM3+, LAG3+
cluster_profiles[7, [0,2,16,17,18]] = [4, 4, 4, 3, 3]
# NK: CD56+, CD16+, GRANZB+, PERF+
cluster_profiles[8, [5,6,31,32]] = [4, 4, 3, 3]
# B cell: CD19+, CD20+
cluster_profiles[9, [3,4]] = [4, 4]
# Monocyte: CD14+, CD11b+, HLA-DR+
cluster_profiles[10, [7,8,34]] = [4, 4, 3]
# DC: CD11c+, HLA-DR+
cluster_profiles[11, [9,34]] = [4, 4]

# ── Simulate cell data ────────────────────────────────────────────────────────
# Cluster proportions (vary by sample)
cluster_freq_base = np.array([0.15, 0.10, 0.08, 0.05, 0.12, 0.08, 0.07, 0.03,
                               0.10, 0.12, 0.07, 0.03])
cluster_freq_base /= cluster_freq_base.sum()

# Healthy vs disease (first 50 healthy, last 50 disease)
n_healthy = 50; n_disease = 50
disease_shift = np.zeros(N_CLUSTERS)
disease_shift[7] = 0.05   # more exhausted CD8
disease_shift[3] = 0.02   # more Tregs
disease_shift[4] = -0.03  # fewer naive CD8

sample_cluster_freq = np.zeros((N_SAMPLES, N_CLUSTERS))
for s in range(N_SAMPLES):
    if s < n_healthy:
        freq = cluster_freq_base + np.random.normal(0, 0.01, N_CLUSTERS)
    else:
        freq = cluster_freq_base + disease_shift + np.random.normal(0, 0.01, N_CLUSTERS)
    freq = np.clip(freq, 0.01, 1)
    sample_cluster_freq[s] = freq / freq.sum()

# Simulate cells for one representative sample
cell_clusters = np.random.choice(N_CLUSTERS, N_CELLS, p=cluster_freq_base)
cell_expr = np.zeros((N_CELLS, N_MARKERS))
for c in range(N_CLUSTERS):
    mask = cell_clusters == c
    n_c  = mask.sum()
    if n_c == 0: continue
    mean_expr = cluster_profiles[c]
    cell_expr[mask] = np.random.normal(mean_expr, 0.5, (n_c, N_MARKERS))
cell_expr = np.clip(cell_expr, 0, 6)

# ── UMAP (simplified 2D embedding via PCA) ────────────────────────────────────
# Use PCA as UMAP proxy
E = cell_expr - cell_expr.mean(axis=0)
E /= (E.std(axis=0) + 1e-8)
# Subsample for speed
sub_idx = np.random.choice(N_CELLS, 5000, replace=False)
E_sub = E[sub_idx]
cov_e = E_sub.T @ E_sub / len(sub_idx)
eigvals_e, eigvecs_e = np.linalg.eigh(cov_e)
idx_e = np.argsort(eigvals_e)[::-1]
umap_coords = E_sub @ eigvecs_e[:, idx_e[:2]]
umap_clusters = cell_clusters[sub_idx]

# ── Phenotype scoring ─────────────────────────────────────────────────────────
# Activation score: CD69, HLA-DR, Ki67
activation_markers = [38, 34, 25]   # CD69, HLA-DR, Ki67
exhaustion_markers = [16, 17, 18, 19]  # PD1, TIM3, LAG3, CTLA4
memory_markers     = [13, 26, 36]   # CD45RO, CD27, CD28

activation_score = cell_expr[:, activation_markers].mean(axis=1)
exhaustion_score = cell_expr[:, exhaustion_markers].mean(axis=1)
memory_score     = cell_expr[:, memory_markers].mean(axis=1)

# Per-sample phenotype scores
sample_activation = np.zeros(N_SAMPLES)
sample_exhaustion = np.zeros(N_SAMPLES)
for s in range(N_SAMPLES):
    if s < n_healthy:
        sample_activation[s] = np.random.normal(1.5, 0.3)
        sample_exhaustion[s] = np.random.normal(0.8, 0.2)
    else:
        sample_activation[s] = np.random.normal(2.2, 0.4)
        sample_exhaustion[s] = np.random.normal(1.8, 0.4)

# ── Marker co-expression ──────────────────────────────────────────────────────
# Correlation matrix of top 15 markers
top_markers = markers[:15]
corr_mat = np.corrcoef(cell_expr[:, :15].T)

# ── Clinical correlation ──────────────────────────────────────────────────────
# Simulate clinical variable (e.g., disease severity score)
clinical_score = np.zeros(N_SAMPLES)
clinical_score[:n_healthy] = np.random.normal(1, 0.5, n_healthy)
clinical_score[n_healthy:] = np.random.normal(5, 1.5, n_disease)
clinical_score = np.clip(clinical_score, 0, 10)

# Correlation with exhaustion
r_exhaust, p_exhaust = stats.pearsonr(clinical_score, sample_exhaustion)

# ── Dashboard ─────────────────────────────────────────────────────────────────
CLUSTER_COLORS = plt.cm.tab20(np.linspace(0, 1, N_CLUSTERS))

fig, axes = plt.subplots(3, 3, figsize=(20, 15))
fig.patch.set_facecolor('#0d1117')
fig.suptitle('Immune Phenotype Engine (CyTOF) — Dashboard', color='white', fontsize=16, fontweight='bold', y=0.98)

def style_ax(ax, title, xlabel='', ylabel=''):
    ax.set_facecolor('#161b22')
    ax.set_title(title, color='white', fontsize=11, fontweight='bold')
    ax.set_xlabel(xlabel, color='#8b949e')
    ax.set_ylabel(ylabel, color='#8b949e')
    ax.tick_params(colors='#8b949e')
    for spine in ax.spines.values():
        spine.set_edgecolor('#30363d')

# Panel 1 — UMAP colored by cluster
ax = axes[0,0]
for c in range(N_CLUSTERS):
    mask = umap_clusters == c
    if mask.sum() > 0:
        ax.scatter(umap_coords[mask, 0], umap_coords[mask, 1],
                   c=[CLUSTER_COLORS[c]], s=3, alpha=0.6, label=cluster_names[c])
style_ax(ax, 'UMAP — Immune Cell Subsets', 'UMAP1', 'UMAP2')
ax.legend(fontsize=6, labelcolor='white', facecolor='#21262d', edgecolor='#30363d',
          markerscale=3, ncol=2)

# Panel 2 — Marker expression heatmap
ax = axes[0,1]
# Mean expression per cluster for top 20 markers
heatmap_data = np.zeros((N_CLUSTERS, 20))
for c in range(N_CLUSTERS):
    mask = cell_clusters == c
    if mask.sum() > 0:
        heatmap_data[c] = cell_expr[mask, :20].mean(axis=0)
im = ax.imshow(heatmap_data.T, aspect='auto', cmap='YlOrRd', interpolation='nearest')
ax.set_yticks(range(20))
ax.set_yticklabels(markers[:20], color='white', fontsize=6)
ax.set_xticks(range(N_CLUSTERS))
ax.set_xticklabels([n.split()[0] for n in cluster_names], rotation=45, ha='right',
                    color='white', fontsize=7)
plt.colorbar(im, ax=ax, label='Mean Expression')
style_ax(ax, 'Marker Expression Heatmap', 'Cluster', 'Marker')

# Panel 3 — Cell subset frequency
ax = axes[0,2]
freq_mean = sample_cluster_freq.mean(axis=0)
freq_se   = sample_cluster_freq.std(axis=0) / np.sqrt(N_SAMPLES)
bars = ax.bar(range(N_CLUSTERS), freq_mean*100, yerr=freq_se*100,
              color=CLUSTER_COLORS, capsize=3, edgecolor='#0d1117', alpha=0.85)
ax.set_xticks(range(N_CLUSTERS))
ax.set_xticklabels([n.split()[0] for n in cluster_names], rotation=45, ha='right',
                    color='white', fontsize=7)
style_ax(ax, 'Cell Subset Frequency', 'Subset', 'Frequency (%)')

# Panel 4 — Phenotype score distribution
ax = axes[1,0]
ax.hist(activation_score, bins=50, color='#f78166', alpha=0.7, label='Activation', density=True)
ax.hist(exhaustion_score, bins=50, color='#58a6ff', alpha=0.7, label='Exhaustion', density=True)
ax.hist(memory_score, bins=50, color='#3fb950', alpha=0.7, label='Memory', density=True)
style_ax(ax, 'Phenotype Score Distribution', 'Score', 'Density')
ax.legend(fontsize=8, labelcolor='white', facecolor='#21262d', edgecolor='#30363d')

# Panel 5 — Disease vs healthy comparison
ax = axes[1,1]
groups = [('Healthy', sample_exhaustion[:n_healthy], '#3fb950'),
          ('Disease', sample_exhaustion[n_healthy:], '#f78166')]
for i, (label, data, color) in enumerate(groups):
    bp = ax.boxplot(data, positions=[i+1], patch_artist=True, widths=0.5,
                    medianprops={'color': 'white', 'lw': 2})
    bp['boxes'][0].set_facecolor(color)
    bp['boxes'][0].set_alpha(0.7)
    for el in ['whiskers', 'caps', 'fliers']:
        for item in bp[el]:
            item.set_color('#8b949e')
ax.set_xticks([1, 2])
ax.set_xticklabels(['Healthy', 'Disease'], color='white')
t_stat, p_val = stats.ttest_ind(sample_exhaustion[:n_healthy], sample_exhaustion[n_healthy:])
ax.text(0.98, 0.95, f'p={p_val:.2e}', transform=ax.transAxes,
        color='white', ha='right', va='top', fontsize=10)
style_ax(ax, 'Exhaustion Score: Healthy vs Disease', 'Group', 'Exhaustion Score')

# Panel 6 — Marker co-expression
ax = axes[1,2]
im2 = ax.imshow(corr_mat, cmap='RdBu_r', vmin=-1, vmax=1, interpolation='nearest')
ax.set_xticks(range(15))
ax.set_yticks(range(15))
ax.set_xticklabels(top_markers, rotation=45, ha='right', color='white', fontsize=7)
ax.set_yticklabels(top_markers, color='white', fontsize=7)
plt.colorbar(im2, ax=ax, label='Pearson r')
style_ax(ax, 'Marker Co-expression Matrix', '', '')

# Panel 7 — Clinical correlation
ax = axes[2,0]
ax.scatter(clinical_score[:n_healthy], sample_exhaustion[:n_healthy],
           c='#3fb950', s=50, alpha=0.8, label='Healthy')
ax.scatter(clinical_score[n_healthy:], sample_exhaustion[n_healthy:],
           c='#f78166', s=50, alpha=0.8, label='Disease')
x_fit = np.linspace(0, 10, 100)
slope_c, intercept_c, _, _, _ = stats.linregress(clinical_score, sample_exhaustion)
ax.plot(x_fit, slope_c*x_fit + intercept_c, 'w--', lw=2,
        label=f'r={r_exhaust:.3f}, p={p_exhaust:.2e}')
style_ax(ax, 'Clinical Correlation (Exhaustion vs Severity)', 'Disease Severity', 'Exhaustion Score')
ax.legend(fontsize=7, labelcolor='white', facecolor='#21262d', edgecolor='#30363d')

# Panel 8 — Subset trajectory (healthy → disease)
ax = axes[2,1]
healthy_freq = sample_cluster_freq[:n_healthy].mean(axis=0)
disease_freq = sample_cluster_freq[n_healthy:].mean(axis=0)
delta_freq = disease_freq - healthy_freq
colors_delta = ['#f78166' if d > 0 else '#3fb950' for d in delta_freq]
ax.bar(range(N_CLUSTERS), delta_freq*100, color=colors_delta, edgecolor='#0d1117', alpha=0.85)
ax.axhline(0, color='white', lw=1)
ax.set_xticks(range(N_CLUSTERS))
ax.set_xticklabels([n.split()[0] for n in cluster_names], rotation=45, ha='right',
                    color='white', fontsize=7)
style_ax(ax, 'Subset Frequency Shift (Disease − Healthy)', 'Subset', 'Δ Frequency (%)')

# Panel 9 — Summary
ax = axes[2,2]
ax.axis('off')
style_ax(ax, 'Summary Statistics')
summary = [
    f'Samples: {N_SAMPLES} ({n_healthy} healthy, {n_disease} disease)',
    f'Markers: {N_MARKERS}',
    f'Cells simulated: {N_CELLS:,}',
    f'Cell subsets: {N_CLUSTERS}',
    f'Largest subset: {cluster_names[np.argmax(cluster_freq_base)]}',
    f'Mean activation (healthy): {sample_activation[:n_healthy].mean():.2f}',
    f'Mean activation (disease): {sample_activation[n_healthy:].mean():.2f}',
    f'Mean exhaustion (healthy): {sample_exhaustion[:n_healthy].mean():.2f}',
    f'Mean exhaustion (disease): {sample_exhaustion[n_healthy:].mean():.2f}',
    f'Exhaustion-severity r: {r_exhaust:.3f}',
]
for k, line in enumerate(summary):
    ax.text(0.05, 0.92 - k*0.09, line, transform=ax.transAxes,
            color='#e6edf3', fontsize=10, va='top')

plt.tight_layout(rect=[0, 0, 1, 0.97])
out_png = '/mnt/shared-workspace/shared/immune_phenotype_engine_dashboard.png'
plt.savefig(out_png, dpi=100, bbox_inches='tight', facecolor='#0d1117')
plt.close()
print(f'Dashboard saved: {out_png}')

shutil.copy('/workspace/subagents/a29c645f/immune_phenotype_engine.py',
            '/mnt/shared-workspace/shared/immune_phenotype_engine.py')

print('\n=== KEY RESULTS: ImmunePhenotypeEngine ===')
print(f'Samples: {N_SAMPLES}, Markers: {N_MARKERS}, Cells: {N_CELLS:,}')
print(f'Cell subsets: {N_CLUSTERS}')
print(f'Mean activation (healthy): {sample_activation[:n_healthy].mean():.2f}')
print(f'Mean activation (disease): {sample_activation[n_healthy:].mean():.2f}')
print(f'Mean exhaustion (healthy): {sample_exhaustion[:n_healthy].mean():.2f}')
print(f'Mean exhaustion (disease): {sample_exhaustion[n_healthy:].mean():.2f}')
print(f'Exhaustion-severity correlation: r={r_exhaust:.3f}, p={p_exhaust:.2e}')
print(f'Exhaustion t-test p-value: {p_val:.2e}')
