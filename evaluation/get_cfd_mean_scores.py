import pandas as pd

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv('cfd_scores.csv', header=0)
df.columns = ['image_id', 'result_idx', 'result_path',
              'cfd', 'd_context', 'd_hallucination', 'is_best']

df['cfd']             = pd.to_numeric(df['cfd'],             errors='coerce')
df['d_context']       = pd.to_numeric(df['d_context'],       errors='coerce')
df['d_hallucination'] = pd.to_numeric(df['d_hallucination'], errors='coerce')
df['is_best']         = df['is_best'].astype(str).str.strip().str.lower() == 'true'

# Extract scene prefix (e.g. "V11" from "V11_00005")
df['scene'] = df['image_id'].str.extract(r'^([^_]+)')

# Sheet 1: Average per pair
# All 4 results averaged
avg_per_pair_all = (
    df.groupby('image_id')[['cfd', 'd_context', 'd_hallucination']]
    .mean()
    .round(4)
    .reset_index()
    .rename(columns={
        'cfd':             'cfd_mean_all',
        'd_context':       'd_context_mean_all',
        'd_hallucination': 'd_hallucination_mean_all'
    })
)

# Best result only
best_per_pair = (
    df[df['is_best']]
    .groupby('image_id')[['cfd', 'd_context', 'd_hallucination']]
    .mean()
    .round(4)
    .reset_index()
    .rename(columns={
        'cfd':             'cfd_best',
        'd_context':       'd_context_best',
        'd_hallucination': 'd_hallucination_best'
    })
)

avg_per_pair = avg_per_pair_all.merge(best_per_pair, on='image_id', how='left')
avg_per_pair.insert(1, 'scene', avg_per_pair['image_id'].str.extract(r'^([^_]+)')[0])

# Sheet 2: Average per scene
# All results averaged
avg_per_scene_all = (
    df.groupby('scene')[['cfd', 'd_context', 'd_hallucination']]
    .mean()
    .round(4)
    .reset_index()
    .rename(columns={
        'cfd':             'cfd_mean_all',
        'd_context':       'd_context_mean_all',
        'd_hallucination': 'd_hallucination_mean_all'
    })
)

# Best results only averaged per scene
avg_per_scene_best = (
    df[df['is_best']]
    .groupby('scene')[['cfd', 'd_context', 'd_hallucination']]
    .mean()
    .round(4)
    .reset_index()
    .rename(columns={
        'cfd':             'cfd_best',
        'd_context':       'd_context_best',
        'd_hallucination': 'd_hallucination_best'
    })
)

# Pair count per scene
pair_count = df.groupby('scene')['image_id'].nunique().reset_index()
pair_count.columns = ['scene', 'num_pairs']

avg_per_scene = avg_per_scene_all \
    .merge(avg_per_scene_best, on='scene', how='left') \
    .merge(pair_count, on='scene', how='left')

# Reorder: scene, num_pairs, then scores
cols = ['scene', 'num_pairs',
        'cfd_mean_all', 'd_context_mean_all', 'd_hallucination_mean_all',
        'cfd_best', 'd_context_best', 'd_hallucination_best']
avg_per_scene = avg_per_scene[cols]

# Write to Excel
output_path = 'cfd_scores_aggregated.xlsx'

with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
    df.to_excel(writer, sheet_name='Raw Data', index=False)
    avg_per_pair.to_excel(writer, sheet_name='Avg per Pair', index=False)
    avg_per_scene.to_excel(writer, sheet_name='Avg per Scene', index=False)

print(f"Saved: {output_path}")
print(f"  Raw rows:   {len(df)}")
print(f"  Pairs:      {len(avg_per_pair)}")
print(f"  Scenes:     {len(avg_per_scene)}")