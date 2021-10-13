#!/usr/bin/env python
# coding: utf-8

# In[1]:


from cobra import Model, Reaction, Metabolite
from riptide import *
import pandas as pd
from copy import *
from cobra.medium import minimal_medium
import copy
import numpy as np


# In[2]:


#recon 3d https://www.nature.com/articles/nbt.4072
model = cobra.io.read_sbml_model("Recon3D.xml")


# In[3]:


#metabolites that are signficantly higher in the non-acute disease state (relative down in the severe disease state)
down_formula = ["C29H48O2",
                "C22H40O4",
                "C9H10N2O4",
                "C13H24N2O",
                "C24H50NO7P",
                "C11H21NO2",
                "C26H50NO7P",
                "C11H15N5O4",
                "C26H52NO7P",
                "C21H44NO7P",
                "C14H31N",
                "C6H15NO2",
                "C8H5NO",
                "C6H15NO",
                "C5H8O3",
                "C13H27NO2",
                "C16H30O",
                "C10H10N2O3S",
                "C21H37NO4",
                "C15H21NO",
                "C21H23NO2",
                "C7H7NO3",
                "C4H8O3",
                "C16H29NO4",
                "C6H6O8",
                "C14H26O3",
                "C8H15N3O3",
                "C11H12N2O3",
                "C11H15N5O3S",
                "C5H7NO3",
                "C5H8O2",
                "C9H13N5O4",
                "C9H21NO12P2",
                "C7H7NO2",
                "C6H13NO2",
                "C5H10O6",
                "C4H8O4",
                "C8H17NO2",
                "C22H43NO",
                "C2H8NO4P",
                "C10H17N3O6",
                "C20H32N6O12S2",
                "C3H9O6P",
                "C3H7N3O2",
                "C8H15NO3",
                "C7H16N4O2",
                "C10H9NO",
                "C6H14N4O2",
                "C6H9O9P",
                "C7H16N2O2",
                "C6H13N3O3",
                "C5H8N2O9",
                "C4H9NO5S",
                "C10H18O2",
                "C25H45NO4",
                "C4H6O5",
                "C11H12N2O2",
                "C13H16N2O2",
                "C12H23NO4",
                "C9H18N2O4S",
                "C8H15NO4S",
                "C8H13NO5",
                "C8H16N2O3",
                "C12H17N5O5",
                "C9H20N2O2",
                "C8H11N3O3",
                "C11H15NO9P",
                "C8H8N2O3",
                "C11H17N3O6",
                "O3V",
                "C31H46O2",
                "C6H11NO2",
                "C8H20NO6P",
                "C10H26N4",
                "C11H9NO2",
                "C12H31NO3Si3",
                "C20H21N3O3",
                "C4H4N2O2",
                "C4H5N3O3",
                "C9H12N2O6"
                   ]
down = ["(24R,24'R)-Fucosterolepoxide", #
        "(2Z)-4-(Octadecyloxy)-4-oxo-2-butenoic acid", #
        "N-Carbamoyl-2-amino-2-(4-hydroxyphenyl)acetic acid", #
        "1,3-Dicyclohexylurea", #
        "1-palmitoleoyl-glycerophosphocholine", #
        "11-Nitro-1-undecene", #
        "1-Linoleoylglycerophosphocholine", #
        "1-Methyladenosine", #
        "1-Oleoylglycerophosphocholine", #
        "1-Palmitoyl-2-hydroxy-sn-glycero-3-PE", #
        "1-Tetradecylamine", #
        "2,2'-Iminodipropan-1-ol", #
        "2-cyanobenzaldehyde", #
        "2-Methylcholine", #
        "2-Oxovaleric acid", #
        "2S-Amino-tridecanoic acid", #
        "2-Hexadecenal", #
        "3-(4-Hydroxy-1,3-benzothiazol-6-yl)alanine", #
        "3, 5-Tetradecadiencarnitine", #
        "3,5-ditert-butyl-4-hydroxybenzonitrile", #
        "3,6-Diacetyl-9-isoamyl carbazole", #
        "3-Hydroxyanthranilic acid", #
        "3-Hydroxybutyric acid", #
        "3-Hydroxy-N-(2-oxotetrahydro-3-furanyl)dodecanamide", #
        "3-Oxalomalate", #
        "3-Oxotetradecanoic acid", #
        "5-(N,N-Dimethylcarbamimidamido)-2-oxopentanoic acid", #
        "5-Hydroxy-L-tryptophan", #
        "5'-Methylthioadenosine", #
        "5-Oxoproline", #
        "5-Valerolactone", #
        "7,8-Dihydroneopterin", #
        "alanyl-poly(glycerol phosphate)", #
        "Anthranilate", #
        "beta-Leucine", #
        "D-Arabinonate", #
        "D-Erythrose", #
        "DL-α-Aminocaprylic acid", #
        "Erucamide", #
        "ETHANOLAMINE PHOSPHATE", #
        "gamma-Glu-gln", #
        "glutathione disulfide", #
        "GLYCEROL 3-PHOSPHATE", #
        "Guanidineacetic acid", #
        "Hexanoylglycine", #
        "Homoarginine", #
        "INDOLEACETALDEHYDE", #
        "L-Arginine", #
        "L-Ascorbate 6-phosphate", #
        "R-Carnitinamide", #
        "L-Citrulline", #
        "L-Cystine", #
        "L-Homocysteic acid", #
        "Limonene-1,2-diol", #
        "Linoleyl carnitine", #
        "L-Malic acid", #
        "L-Tryptophan", #
        "Melatonin", #
        "N-(tert-butoxycarbonyl)-L-leucinate", #
        "met-thr", #
        "N-[(2S)-2-Hydroxy propanoyl]methionine", #
        "N2-Acetyl-L-aminoadipate", #
        "N2-Acetyl-L-lysine", #
        "N2-Dimethylguanosine", #
        "N6,N6,N6-Trimethyl-L-lysine", #
        "N-Acetyl-L-histidine", #
        "Nicotinate D-ribonucleotide", #
        "Nicotinuric acid", #
        "n-Ribosylhistidine", #
        "Vanadium", #
        "Phylloquinone", #
        "Pipecolinic acid", #
        "sn-glycero-3-Phosphocholine", #
        "Spermine", #
        "Indoleacrylic acid", #
        "Trimethylsilyl N,O-bis(trimethylsilyl)serinate", #
        "Trp-Phe", #
        "Uracil", #
        "Uramil", #
        "Uridine"#
        ]

# 4-(1,2-dihydroxyethyl)benzene-1,2-diol not found in metabolomics data set

up_formula = ["C12H12N2O2",
              "C12H23NO5",
              "C6H8S2",
              "C6H8N2O3",
              "C4H6O3",
              "C12H16N2O4",
              "C5H10N2O3S",
              "C4H5N3O",
              "C6H12O7",
              "C13H25NO4",
              "C13H25NO5",
              "C6H6N2O3",
              "C9H16N2O5",
              "C8H16N2O4S2",
              "C6H8N2O2",
              "C7H12NO8P",
              "C12H17NO5",
              "C13H23NO6",
              "C11H20N2O3",
              "C5H11NO2S"
             ]
#metabolites that are signficantly higher in the severe disease state 
up = ["2,3,4,9-Tetrahydro-1H-beta-carboline-3-carboxylic acid", #
        "3-hydroxyisovalerylcarnitine", #
        "3-Vinyl-1,2-dithiacyclohex-5-ene", #
        "4-Imidazolone-5-propanoate", #
        "9,12-Hexadecadienoylcarnitine", #
        "Acetoacetate", #
        "Alanyltyrosine", #
        "CYS-GLY", #
        "Cytosine", #
        "D-Galactonate", #
        "Hexanoylcarnitine", #
        "Hydroxyhexanoycarnitine", #
        "Imidazol-5-yl-pyruvate", #
        "L-gamma-Glutamyl-L-leucine", #
        "L-Homocystine", #
        "Methylimidazoleacetic acid", #
        "N-Acetyl-L-glutamate 5-phosphate", #
        "N-D-Glucosylarylamine", #
        "O-3-methylglutarylcarnitine", #
        "Prolylleucine",
        "L-Methionine"#
        ]


# In[4]:


# find metabolite ids of metabolites that are higher in severe COVID-19 samples
higher_metabolite_ids = []
for metabolite in model.metabolites:
    if metabolite.formula in up_formula:
        higher_metabolite_ids.append(metabolite.id)

# find metabolite ids of metabolites that are lower in severe COVID-19 samples
lower_metabolite_ids = []
for metabolite in model.metabolites:
    if metabolite.formula in down_formula:
        lower_metabolite_ids.append(metabolite.id)


# In[5]:


len(set(lower_metabolite_ids))


# In[6]:


# find existing exchange reactions for higher metabolites
higher_exchange_rxn_id = []
higher_metabolite_w_exchange_rxn = []
for rxn in model.exchanges:
    for metabolite in higher_metabolite_ids:
        if metabolite in rxn.reaction:
            higher_exchange_rxn_id.append(rxn.id)
            higher_metabolite_w_exchange_rxn.append(metabolite)

# find existing exchange reactions for lower metabolites
lower_exchange_rxn_id = []
lower_metabolite_w_exchange_rxn = []
for rxn in model.exchanges:
    for metabolite in lower_metabolite_ids:
        if metabolite in rxn.reaction:
            lower_exchange_rxn_id.append(rxn.id)
            lower_metabolite_w_exchange_rxn.append(metabolite)


# In[7]:


#identify relatively higher metabolites that exist in model but do NOT have exchange reaction

#strip compartment id from higher metabolites that have exchange reaction
higher_stripped = []
for metabolite in higher_metabolite_w_exchange_rxn:
    higher_stripped.append(metabolite[:-2])

#strip compartment id from ALL higher metabolites
higher_metabolite_stripped = []
for metabolite in higher_metabolite_ids:
    higher_metabolite_stripped.append(metabolite[:-2])

#identify ids that are found in higher metabolite list but not found in metabolite exchange reaction
higher_no_exchange = list(set(higher_metabolite_stripped) - set(higher_stripped))

#all higher metabolites have corresponding exchange reaction
print(len(higher_no_exchange))


# In[8]:


#identify relatively lower metabolites that exist in model but do NOT have exchange reaction

#strip compartment id from lower metabolites that have exchange reaction
lower_stripped = []
for metabolite in lower_metabolite_w_exchange_rxn:
    lower_stripped.append(metabolite[:-2])

#strip compartment id from ALL lower metabolites
lower_metabolite_stripped = []
for metabolite in lower_metabolite_ids:
    lower_metabolite_stripped.append(metabolite[:-2])

#identify ids that are found in lower metabolite list but not found in metabolite exchange reaction
lower_no_exchange = list(set(lower_metabolite_stripped) - set(lower_stripped))

#6 metabolites exist in the model but do NOT have exchange reactions
print(len(lower_no_exchange))
print(lower_no_exchange)


# In[9]:


#create extracellular metabolite ids for the missing exchange reaction metabolites
new_extracellular = []
for cpd in lower_no_exchange:
    new_cpd = deepcopy(model.metabolites.get_by_id(cpd+"_c"))
    new_cpd.id = cpd + '_e'
    new_cpd.compartment = 'e'
    model.add_metabolites([new_cpd])
    new_extracellular.append(new_cpd.id)
    
#add missing lower exchange reactions for new extracellular metabolites 
for metabolite in new_extracellular:
    reaction = Reaction('EX_' + metabolite)
    reaction.name = model.metabolites.get_by_id(metabolite).name + 'exchange'
    reaction.subsystem = 'exchange'
    reaction.lower_bound = 0 
    reaction.upper_bound = 1000
    reaction.add_metabolites({model.metabolites.get_by_id(metabolite):-1.0})
    model.add_reactions([reaction])
    lower_exchange_rxn_id.append(reaction.id)


# In[10]:


#create transport reactions for new extracellular metabolites [e] --> [c]
for metabolite in new_extracellular:
    reaction = Reaction('transport_' + metabolite)
    reaction.name = model.metabolites.get_by_id(metabolite).name + ' Transport'
    reaction.subsystem = 'Transport'
    reaction.lower_bound = 0 
    reaction.upper_bound = 1000
    cytoplasm = metabolite.split('_')[0] + '_c'
    space = metabolite.split('_')[0] + '_e'
    reaction.add_metabolites({model.metabolites.get_by_id(space):-1.0,
                                          model.metabolites.get_by_id(cytoplasm):1.0})
    model.add_reactions([reaction])
    print(reaction)


# In[11]:


max_growth = model.slim_optimize()
df = pd.DataFrame(minimal_medium(model, max_growth))
essential_exchange_rxns = list(df.index.values)


# In[12]:


# function that changes the physiological constraints of the model by altering the bounds of the appropriate exchange reactions
    # Inputs:
        # model: a cobrapy model
        # media: the media condition to change the environment to
            # 1 - blood media
        # limEX: a list of exchanges ('EX_cpd#####(e)') that are added as limited exchanges to minimal media
    # Outputs:
        # modelOutput: a cobrapy model in the new media condition (with adjusted exchanges)
        
def changeMedia(model, media, limEX=[]):
    modelOutput = deepcopy(model)
    
    # Set the new media conditions
    for ex in modelOutput.exchanges:
        ex.upper_bound = 1000
        ex.lower_bound = 0
        
    # Set the media conditions to be minimal media as the non-acute baseline
    if media == 1:
        for exchange in modelOutput.exchanges:
            if exchange.id in essential_exchange_rxns:
                exchange.lower_bound = -1000
        
    #Change bounds to consume metabolites that are higher in non-acute, and produce metabolites that are higher in severe
    if media == 2:
        for exchange in modelOutput.exchanges:
            if exchange.id in lower_exchange_rxn_id:
                exchange.lower_bound = -1000
                exchange.upper_bound = -0.0000001
        for exchange in modelOutput.exchanges:
            if exchange.id in higher_exchange_rxn_id:
                exchange.lower_bound = 0.0000001
                exchange.upper_bound = 1000
    return(modelOutput) 


# In[13]:


# change conditions to mimic non-acute or severe environment
nonacute_model = changeMedia(model, media= 1)
severe_model = changeMedia(model, media = 2)


# In[14]:


#bounds are open to only essential exchange reactions for minimal media
nonacute_contextualization = riptide.contextualize(model = nonacute_model,
                                                    fraction = 0.1,
                                                    conservative = True)


# In[16]:


riptide.save_output(riptide_obj = nonacute_contextualization, path = "nonacute_minmedia")


# In[17]:


#severe model
#force consumption of metabolites that are higher in non acute, force production of metabolites higher in severe
severe_contextualization = riptide.contextualize(model = severe_model,
                                                    fraction = 0.1,
                                                    conservative = True)


# In[18]:


riptide.save_output(riptide_obj = severe_contextualization, path = "severe")


# In[19]:


#opened all exchange reactions --> positive control
open_contextualization = riptide.contextualize(model=model,
                                                fraction = 0.1,
                                                conservative = True,
                                                open_exchanges = True)


# In[20]:


riptide.save_output(riptide_obj = open_contextualization, path = "open")


# In[21]:


#3-model analysis
#Start with riptide models loaded as baseline_mean_rip
#Pull flux sampling data
nonacute = nonacute_contextualization.flux_samples
severe = severe_contextualization.flux_samples
open_e = open_contextualization.flux_samples

#Find consensus rxns for entire set (NMDS) and abtgc & muc5b (randomForest)
nonacute_rxns = nonacute.columns.tolist()
severe_rxns = severe.columns.tolist()
open_rxns = open_e.columns.tolist()

common_rxns = set(nonacute_rxns) & set(severe_rxns) & set(open_rxns)#<-- shared reactions

#Create data frame with flux samples of abtgc & muc5b consensus rxns for randomForest
nonacute_rf = nonacute[common_rxns]
severe_rf = severe[common_rxns]
open_rf = open_e[common_rxns]

#Create RF data set
shared_rxns = pd.concat([open_rf,nonacute_rf,severe_rf])

#Save rf to excel so that it can be read into R - need to add model names manually due to error
shared_rxns.to_excel("commonrxn.xlsx")


# In[22]:


#find reactions that are shared between at least two models
non_severe_common = list(set(nonacute_rxns) & set(severe_rxns))
non_open_common =  list(set(nonacute_rxns) & set(open_rxns))
severe_open_common = list(set(severe_rxns) & set(open_rxns))

#create a consensus list of those reactions
all_common_rxns = set(non_severe_common + non_open_common + severe_open_common)

#remove reactions that are present in ALL three models
common_diff_rxns = [i for i in all_common_rxns if i not in list(common_rxns)]

#find reactions that are common in each individual model
nonacute_shared = [i for i in nonacute_rxns if i in common_diff_rxns]
severe_shared = [i for i in severe_rxns if i in common_diff_rxns]
open_shared = [i for i in open_rxns if i in common_diff_rxns]


# In[23]:


#Severe vs Non-Acute Analysis
#Start with riptide models loaded as baseline_mean_rip (abtgc model) and muc5b_mean_rip (muc5b model)
#Pull flux sampling data
nonacute = nonacute_contextualization.flux_samples
severe = severe_contextualization.flux_samples

#Find consensus rxns for nonacute vs severe (NMDS) 
nonacute_rxns2 = nonacute.columns.tolist()
severe_rxns2 = severe.columns.tolist()
common_rxns2 = set(nonacute_rxns) & set(severe_rxns) #<-- shared reactions

#Create data frame with flux samples of abtgc & muc5b consensus rxns for randomForest
nonacute_rf2 = nonacute[common_rxns2]
severe_rf2 = severe[common_rxns2]

#Create RF data set
shared_rxns = pd.concat([nonacute_rf2,severe_rf2])

#Save rf to excel so that it can be read into R - need to add model names manually due to error
shared_rxns.to_excel("commonrxn_nonacutevssevere.xlsx")


# In[24]:


#create list of metabolite ids for core rxns shared between nonacute and severe models
rxn_metabolites = []       
for rxn in common_rxns2:
    for metabolite in model.reactions.get_by_id(rxn).metabolites:
        rxn_metabolites.append(metabolite.name)
        
len(set(rxn_metabolites))


# In[25]:



import csv

f=open('commonrxn_riptide_metabolites.txt','w')
for element in set(rxn_metabolites):
    f.write(element+'\n')

f.close()


# In[26]:


#create list of metabolite ids for non-shared reactions 
nonacute_diff_rxn = [i for i in nonacute_rxns2 if i not in list(common_rxns2)]
severe_diff_rxn = [i for i in severe_rxns2 if i not in list(common_rxns2)]


# In[27]:


#create list of metabolite ids for rxns found ONLY in nonacute models
nonacute_rxn_metabolites = []       
for rxn in nonacute_diff_rxn:
    for metabolite in model.reactions.get_by_id(rxn).metabolites:
        nonacute_rxn_metabolites.append(metabolite.name)

#create list of metabolite ids for rxns found ONLY in severe models
severe_rxn_metabolites = []       
for rxn in severe_diff_rxn:
    for metabolite in model.reactions.get_by_id(rxn).metabolites:
        severe_rxn_metabolites.append(metabolite.name)


# In[28]:



import csv

f=open('nonacute_riptide_metabolites.txt','w')
for element in set(nonacute_rxn_metabolites):
    f.write(element+'\n')

f.close()

import csv

f=open('severe_riptide_metabolites.txt','w')
for element in set(severe_rxn_metabolites):
    f.write(element+'\n')

f.close()

