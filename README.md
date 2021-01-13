# Trips-and-Neurotransmitters-Discovering-Principled-Patterns-across-6850-Hallucinogenic-Experiences

`conda env create -f brain_environment.yml`

`conda activate brain`

Then to run the CCA analsis:

```
python cca_on_erowid.py 
    --drug_folder final_drugs_sansdxm
    --limit 300 
    --pca_components 800 
    --cca_components 8 
    --id example_run 
    --seed 19376
```
