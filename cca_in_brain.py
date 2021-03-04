"""
Map a TSV of receptors by CCA components to a TSV of regions by CCA components
"""
import os
import argparse
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--region_by_expression_csv', default='./abagen_42receptors_yeo17net_FINAL_transpose.csv',
                        help='Folder of text dumps of testimonials, one drug per file.')
    parser.add_argument('--receptor_cca_tsv', default='tsvs/receptor_cca_8_on_psychedelics_mdma.tsv',
                        help='Folder of text dumps of testimonials, one drug per file.')
    return parser.parse_args()


args = parse_args()


gene_names_to_receptors = {
    'HTR1A': '5HT1A',
    'HTR2A': '5HT2A',
    'HTR1B': '5HT1B',
    'HTR1D': '5HT1D',
    'HTR1E': '5HT1E',
    'HTR2B': '5HT2B',
    'HTR2C': '5HT2C',
    'HTR5A': '5HT5A',
    'HTR6': '5HT6',
    'HTR7': '5HT7',
    'DRD1': 'D1',
    'DRD2': 'D2',
    'DRD3': 'D3',
    'DRD4': 'D4',
    'DRD5': 'D5',
    'ADRA1A': 'Alpha1A',
    'ADRA2A': 'Alpha2A',
    'ADRA1B': 'Alpha1B',
    'ADRA2B': 'Alpha2B',
    'ADRA2C': 'Alpha2C',
    'ADRB1': 'Beta1',
    'ADRB2': 'Beta2',
    'SLC6A4': 'SERT',
    'SLC6A3': 'DAT',
    'SLC6A2': 'NET',
    'NISCH': 'Imidazoline1',
    'SIGMAR1': 'Sigma1',
    'TMEM97': 'Sigma2',
    'OPRD1': 'DOR',
    'OPRK1': 'KOR',
    'OPRM1': 'MOR',
    'CHRM1': 'M1',
    'CHRM2': 'M2',
    'CHRM3': 'M3',
    'CHRM4': 'M4',
    'CHRM5': 'M5',
    'HRH1': 'H1',
    'HRH2': 'H2',
    'CNR1': 'CB1',
    'CNR2': 'CB2',
    'CACNA1C': 'Ca+Channel',
    'GRIN1': 'NMDA',
}

receptors_to_gene_names = {v: k for k, v in gene_names_to_receptors.items()}
#
df_receptor_ccas = pd.read_csv(args.receptor_cca_tsv, sep='\t')
df_region_by_expression = pd.read_csv(args.region_by_expression_csv)
#print(f'axis  mean: {np.nanmean(df_region_by_expression.astype(np.float32, errors="ignore").to_numpy())} \n axis  std: {np.nanstd(df_region_by_expression.astype(np.float32, errors="ignore").to_numpy())} ')
print(f'axis  min: {df_region_by_expression.min(numeric_only=True).min()} ')
receptor_names_to_index = {name.upper(): i for i, name in enumerate(df_receptor_ccas.columns)}
df_region_by_expression.info()
print(receptor_names_to_index)
df_region_ccas = pd.DataFrame(data=np.zeros((len(df_region_by_expression.columns)-1, len(df_receptor_ccas.index))))
df_region_ccas.index = df_region_by_expression.columns[:-1]
gene_name_to_index = {g: i for i, g in enumerate(df_region_by_expression.gene)}
df_region_ccas.columns = [f'cca_component_{i}' for i in range(len(df_receptor_ccas.index))]
for roi_index, region in enumerate(df_region_by_expression.columns[:-1]):
    count_neg = 0
    count_pos = 0
    for cca_index in range(len(df_receptor_ccas.index)):
        for gene_name in gene_names_to_receptors:
            receptor_name = gene_names_to_receptors[gene_name].upper()
            gene_index = gene_name_to_index[gene_name]
            expression = df_region_by_expression.iloc[gene_index, roi_index]
            if receptor_name not in receptor_names_to_index:
                #print(f'Could not find receptor {receptor_name}')
                continue
            try:
                float(expression)
            except:
                print(f'Bad expression receptor {receptor_name} {roi_index} with gene {gene_name} {gene_index} expression: {expression}')
                continue

            #expression += 4
            if expression < 0:
                count_neg += expression
            elif expression > 0:
                count_pos += expression
            cca_load = df_receptor_ccas.iloc[cca_index, receptor_names_to_index[receptor_name]]
            df_region_ccas.iloc[roi_index, cca_index] += (expression * cca_load) / 3
    print(f'Region: {region} Roi Index {roi_index} Negative expression {count_neg} Positive: {count_pos}')
cca_df = df_region_ccas.loc[df_region_ccas[f'cca_component_0'] != 0]
for cca_index in range(len(df_receptor_ccas.index)):
    print(f'df_region_ccas mean {cca_df.iloc[:, cca_index].mean()}')
    cca_df.iloc[:, cca_index] -= cca_df.iloc[:, cca_index].mean()
    print(f'df_region_ccas mean {cca_df.iloc[:, cca_index].mean()}')
region_by_components_file = f"tsvs/cca_{os.path.basename(args.receptor_cca_tsv).replace('.tsv', '')}_loadings_by_brain_region.csv"
cca_df.to_csv(region_by_components_file)
print(f'Done! Wrote region by components file at: {region_by_components_file}')


# def append_expressions(df_region_by_expression):
#     htr2b = pd.read_csv('~/Downloads/the_last_four_pieces_labels_and_labels2.csv')
#
#     for gene in ['SLC6A4', 'ADRA2B', 'SLC6A2', 'SLC6A20']:
#         print(f"Gene:{gene} has mean {htr2b[gene].mean():.3f} std {htr2b[gene].std():.3f}")
#         htr2b[gene] -= htr2b[gene].mean()
#         htr2b[gene] /= htr2b[gene].std()
#         print(f"After Z scoring: mean {htr2b[gene].mean():.3f} std {htr2b[gene].std():.3f}")
#     dft = df_region_by_expression.transpose()
#     print(f'LAB2:\n{htr2b.labels2}\n\n 0:\n{dft[[1]]}')
#     merge = pd.merge(htr2b, df_region_by_expression.transpose(), left_on='labels2', right_on=0)
#     merge.info()
#     merge.to_csv(f'./full_expressions_z_scored.csv')
# print(f'{list(gene_names_to_receptors.keys())} {len(gene_names_to_receptors)}')
# append_expressions(df_region_by_expression)
