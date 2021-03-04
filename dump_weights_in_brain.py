import os
import argparse
import pandas as pd
import nibabel as nib
from nilearn import datasets
import numpy as np
from nilearn.input_data import NiftiLabelsMasker


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cca_components', default=8, type=int,
                        help='Number of CCA components to find in reduced word count matrix')
    parser.add_argument('--cca_by_brain_region_csv', default='tsvs/cca_max_128_pca_128_None_cca_5_on_psychedelics_mdma_loadings_by_brain_region_schaeffer.csv',
                        help='Folder of text dumps of testimonials, one drug per file.')
    return parser.parse_args()


args = parse_args()

df = pd.read_csv(args.cca_by_brain_region_csv)

yeo_atlas = datasets.fetch_atlas_schaefer_2018(n_rois=200, yeo_networks=17)

print(len(yeo_atlas.labels))
print(yeo_atlas.labels)

print(yeo_atlas.maps)

masker = NiftiLabelsMasker(yeo_atlas.maps)

masker.fit()

yeo_nii = nib.load(yeo_atlas.maps)

print(np.unique(yeo_nii.get_data().ravel()))

# print this to see whether order of regions is matched in CSV sheet
# and in the nilearn atlas
list(zip(df['Unnamed: 0'].values, yeo_atlas.labels))

for i_component in range(args.cca_components):
    comp_weights = df['cca_component_%i' % i_component].values
    out_nii = masker.inverse_transform(comp_weights[None, :])
    out_nii.to_filename(os.path.join('niftis', f"cca_{i_component}_of_{os.path.basename(args.cca_by_brain_region_csv).replace('.csv', '')}.nii.gz"))
