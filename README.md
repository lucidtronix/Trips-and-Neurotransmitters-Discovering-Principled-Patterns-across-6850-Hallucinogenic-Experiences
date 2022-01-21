# Trips-and-Neurotransmitters-Discovering-Principled-Patterns-across-6850-Hallucinogenic-Experiences
This code accompanies 
[our paper available on BioRxiv](https://doi.org/10.1101/2021.07.13.452263), 
analyzing subjective reports of psychedelic experiences and mapping the discovered patterns to the brain.
See for example, the strongest canonical component:
![image](component0.png)


Create the python environment:
`conda env create -f brain_environment.yml`

Activate it:
`conda activate brain`

Then to run the CCA analysis:

```
python cca_on_erowid.py 
    --drug_folder final_drugs_sansdxm
    --limit 300 
    --pca_components 800 
    --cca_components 8 
    --id example_run 
    --seed 19376
```
Put the components in brain regions using receptor expressions:
```
python cca_in_brain.py 
    --receptor_cca_tsv tsvs/tester_max_300_pca_800_None_cca_8_on_final_drugs_sansdxm.tsv
```
Plot each component as a NIFTI file.
```
python dump_weights_in_brain.py 
    --cca_components 8 
    --cca_by_brain_region_csv tsvs/cca_tester_max_300_pca_800_None_cca_8_on_final_drugs_sansdxm_loadings_by_brain_region_schaeffer.csv
```