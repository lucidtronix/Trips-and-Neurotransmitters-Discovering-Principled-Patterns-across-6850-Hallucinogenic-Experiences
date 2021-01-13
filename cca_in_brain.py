"""
Map a TSV of receptors by CCA components to a TSV of regions by CCA components
"""
import os
import argparse
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--region_by_expression_csv', default='schaeffer200_17Net_expr_mat.csv',
                        help='Folder of text dumps of testimonials, one drug per file.')
    parser.add_argument('--receptor_cca_tsv', default='tsvs/receptor_cca_8_on_psychedelics_mdma.tsv',
                        help='Folder of text dumps of testimonials, one drug per file.')
    return parser.parse_args()


args = parse_args()
#5ht2a,5ht2c,5ht2b,5ht1a,5ht1b,5ht1d,5ht1e,5ht5a,5ht6,5ht7,D1,D2,D3,D4,D5,Alpha1A,Alpha1B,Alpha2A,Alpha2B,Alpha2C,Beta1,Beta2,SERT,DAT,NET,Imidazoline1,Sigma1,Sigma2,DOR,KOR,MOR,M1,M2,M3,M4,M5,H1,H2,CB1,CB2,Ca+Channel,NMDA

gene_names_to_receptors = {
    'HTR1A': '5HT1A',
    'HTR2A': '5HT2A',
    'HTR1B': '5HT1B',
    'HTR1D': '5HT1D',
    'HTR1E': '5HT1E',
    #'HTR2B': '5HT2B',
    'HTR2C': '5HT2C',
    'HTR5A': '5HT5A',
    'HTR6': '5HT6',
    'HTR7': '5HT7',
    'DRD1': 'D1',
    'DRD2': 'D2',
    'DRD3': 'D3',
    'DRD4': 'D4',
    'DRD5': 'D5',
    'GABRA1': 'Alpha1A',
    'GABRA2': 'Alpha2A',
    'GABBR1': 'Alpha1B',
    'GABBR2': 'Alpha2B',
    'ITGB1': 'Beta1',
    'ADRB2': 'Beta2',
    'SERTAD1': 'SERT',
    'SLC6A3': 'DAT',
    'SLC6A5': 'NET',
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

            expression += 4
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
region_by_components_file = f"tsvs/cca_{os.path.basename(args.receptor_cca_tsv).replace('.tsv', '')}_loadings_by_brain_region_schaeffer.csv"
cca_df.to_csv(region_by_components_file)
print(f'Done! Wrote region by components file at: {region_by_components_file}')
