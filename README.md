# COVID-19 Metabolomics
In partnership with the [Petri Lab](https://med.virginia.edu/petri-lab/) I investigated functional metabolic shifts between severe (hospitalized) and non-severe (out-patient) COVID-19 patients, using serum metabolomics. 

## [Differential Metabolism & Associated Pathways](https://github.com/lrd3uu/COVID19_metabolomics/blob/main/Data%20Analysis%20Severe%20vs%20Non-acute.py)
I first identified differential metabolites between the two conditions, and the associated metabolic pathways using [MetaboAnalyst](https://www.metaboanalyst.ca/)

Using the identified differential metabolites, along with [RIPTiDe](https://github.com/mjenior/riptide) I created [contextualized models](https://github.com/lrd3uu/COVID19_metabolomics/blob/main/RiPTIDe.py) for both severe and non-acute SARS-CoV2 infection.

## [Flux Balance Analysis](https://github.com/lrd3uu/COVID19_metabolomics/blob/main/Rf_NMDS_UpSet_R_covid.Rmd)
After creating contextualized models, I ran flux balance analysis followed by random forest and NMDS, to identify key differential metabolic pathways between models. Once again using Metaboanalyst, I identified signficantly associated metabolic pathways based on those identified via random forest. NMDS showed signficant clustering based on the conserved reaction flux values. 
