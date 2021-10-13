#!/usr/bin/env python
# coding: utf-8

# In[1]:


#data analysis for ALL metabolomic samples based on COVID status
#severe status is for all HOSPITALIZED COVID patients --> some are vented, some are not 


# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectFromModel
from matplotlib.collections import LineCollection
from sklearn import manifold
from sklearn.metrics import euclidean_distances, pairwise_distances
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from skbio import DistanceMatrix
from skbio.stats import ordination
import sklearn
import scipy
from scipy import stats
from scipy.stats import mannwhitneyu
from numpy.random import seed
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import plot_roc_curve
from statsmodels.stats import multitest


# In[3]:


#run random forest on all metabolites and look at model prediction accuracy

#read in data
data = pd.read_csv("severe_vs_nonacute.csv")
data.groupby(['group']).size()

#select proper data
X = data.drop("group", axis=1)
y = data["group"]

#standardize the dataset
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# split into train and test set
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, stratify=y, test_size=0.3, random_state=42
)

# create the classifier
classifier = RandomForestClassifier(n_estimators=1500)

# Train the model using the training sets
classifier.fit(X_train, y_train)

# prediction on the test set
y_pred = classifier.predict(X_test)

# Calculate Model Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))


# In[4]:


#utilize man whitney u-test followed by fdr to adjust for multiple hypotheses correction to identify differential metabolites

#read in data
df = pd.read_csv("severe_vs_nonacute.csv")
data.groupby(['group']).size()

# convert disease variable to binary : severe covid = true, nonacute covid = false
df['COVID_binary'] = np.where(df['group'] == 'severe', True, False)
dx = 'COVID_binary' # Name of disease state column

#id metabolites significantly elevated in severe COVID disease state
severe_mets = [] # short for candidate metabolites 
severe_metabolite = []
severe_pvals = []

for met_name in df.columns[1:-1]: # You will need to adjust the indexing to accommodate your df (try: df.columns[1:-1])
    test_var = met_name
    if np.median(df.loc[df[dx] == True, test_var]) > np.median(df.loc[df[dx] == False, test_var]): # you can remove this if statement entirely if you want all significantly different metabolites
        # One of my mets could not be tested, the try/except commands account for that issue by skipping it.
        try:
            # Run the test, this assumes non-normal data
            stat_temp, p_temp = mannwhitneyu(df.loc[df[dx] == True, test_var], df.loc[df[dx] == False, test_var])
            severe_pvals.append(p_temp)
            severe_metabolite.append(test_var)
        except:
            continue
            
# multiple test corrections and adjusted pvals 
# Benjamini/Hochberg 
fdr = multitest.multipletests(severe_pvals, alpha=0.05, method='fdr_bh',is_sorted=False, returnsorted=False)
pvals = list(fdr[1])
pvals_metabolites = zip(severe_metabolite, pvals)
for x,y in pvals_metabolites:
    if y <=0.05:
        severe_mets.append(x)


# In[5]:


#id metabolites significantly elevated in non-acute COVID disease state 
non_acute_mets = [] 
non_acute_metabolite = []
non_acute_pvals = []
for met_name in df.columns[1:-1]:
    test_var = met_name
    if np.median(df.loc[df[dx] == True, test_var]) < np.median(df.loc[df[dx] == False, test_var]): # you can remove this if statement entirely if you want all significantly different metabolites
        try:
            # Run the test, this assumes non-normal data
            stat_temp, p_temp = mannwhitneyu(df.loc[df[dx] == True, test_var], df.loc[df[dx] == False, test_var])
            non_acute_pvals.append(p_temp)
            non_acute_metabolite.append(test_var)
        except:
            continue

#multiple test corrections and adjusted pvals 
# Benjamini/Hochberg 
fdr = multitest.multipletests(non_acute_pvals, alpha=0.05, method='fdr_bh',is_sorted=False, returnsorted=False)
pvals = list(fdr[1])
pvals_metabolites = zip(non_acute_metabolite, pvals)
for x,y in pvals_metabolites:
    if y <=0.05:
        non_acute_mets.append(x)


# In[6]:


# run random forest on all signficantly differential metabolites identified in severe vs non acute

keep = non_acute_mets + severe_mets
print(len(keep))
print(len(non_acute_mets))
print(len(severe_mets))

#re-read in data
data = pd.read_csv("severe_vs_nonacute.csv")
data.groupby(['group']).size()

#select only metabolites that are differential
X = data[keep]
y = data["group"]

#standardize the dataset
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# split into train and test set
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, stratify=y, test_size=0.3, random_state=42
)

# create the classifier
classifier = RandomForestClassifier(n_estimators=1500)

# Train the model using the training sets
classifier.fit(X_train, y_train)

# prediction on the test set
y_pred = classifier.predict(X_test)

# Calculate Model Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))


# In[7]:


# metabolites identified as non-endogenous based upon literature search
non_endogenous_metabolites =["Precocene II", #constituent of essential oils https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0135031
                        "hydralazine" #pharmaceutical https://www.genome.jp/dbget-bin/www_bget?cpd:C07040,
                        "allylcysteine",#from garlic https://www.sciencedirect.com/topics/nursing-and-health-professions/s-allylcysteine
                        "DICHLORODIFLUOROMETHANE", #pharmaceutical https://www.genome.jp/dbget-bin/www_bget?dr:D03789
                        "Cinnamaldehyde", #phytochemical compound https://www.genome.jp/dbget-bin/www_bget?cpd:C00903
                        "15-Crown-5", #synthetic compound https://pubchem.ncbi.nlm.nih.gov/compound/Benzo-15-crown-5#section=Names-and-Identifiers
                        "2_4-Dinitrophenylhydrazine", #used to qualitatively detect the carbonyl functionality of a ketone or aldehyde functional group
                        "Tilnoprofen arbamel", #pharmaceutical https://drugs.ncats.io/drug/HY45T0EF6G
                        "Pimilprost", #pharmaceutical 
                        "intercept",
                        "Monoolein", # emulsifying agent https://www.tandfonline.com/doi/full/10.1081/DDC-100101304?scroll=top&needAccess=true
                        "Theobromine", # cocoa derivative compound https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4335269/
                        "Indole-3-acetaldoxime", #plant auxin https://www.pnas.org/content/106/13/5430
                        "PEG-4", #laxative component and cosmetic ingredient https://pubmed.ncbi.nlm.nih.gov/17090481/
                        "BIM-1", #Yeast specific https://www.yeastgenome.org/locus/S000000818 
                        "Sinapylalcohol", #plant metabolite https://pubchem.ncbi.nlm.nih.gov/compound/Sinapyl-alcohol
                        "Acetochlor ESA", #herbicide https://en.wikipedia.org/wiki/Acetochlor
                        "Zearalenone", #mycotoxin https://www.genome.jp/dbget-bin/www_bget?cpd:C09981
                        "Tetraacetylethylenediamine", #detergent https://www.fishersci.com/shop/products/n-n-n-n-tetraacetylethylenediamine-99/AAL0435318
                        "4-Amino-3-hydroxybenzoic acid", #pharmaceutical prep https://www.fishersci.com/shop/products/4-amino-3-hydroxybenzoic-acid-98-6/AAA1020706
                        "Sulfonylbismethane", #dietary supplement https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5372953/
                        "Moxisylyte",#pharmaceutical https://www.genome.jp/dbget-bin/www_bget?dr:D08239
                        "Quininone", #chromatography https://www.sigmaaldrich.com/catalog/product/usp/1597504?lang=en&region=US
                        "3-Hydroxy-2-methyl-4-pyrone", #ferric maltol is used as an anti-anemic drug https://www.genome.jp/dbget-bin/www_bget?dr:D10833
                        "2_6-Dihydroxypseudooxynicotine", #soil bacteria enzyme https://www.uniprot.org/uniprot/C1B568
                        "Ectoine", #bacteria byproduct https://www.sciencedirect.com/topics/medicine-and-dentistry/ectoine
                        "N-acetylmuramate6-phosphate", #e.coli metabolite https://pubchem.ncbi.nlm.nih.gov/compound/N-Acetylmuramic-acid-6-phosphate#:~:text=N%2Dacetylmuramic%20acid%206%2Dphosphate%20is%20a%20member%20of%20muramic,N%2Dacetylmuramate%206%2Dphosphate.
                        "Triethylene glycol", #synthetic chemical platiciser https://pubchem.ncbi.nlm.nih.gov/compound/Triethylene-glycol  
                        "Lotaustralin", #phytochemical https://www.genome.jp/dbget-bin/www_bget?cpd:C08334
                        "Cuminaldehyde", #essential oils https://pubchem.ncbi.nlm.nih.gov/compound/4-Isopropylbenzaldehyde
                        "Dinitrosopentamethylenetetramine", #carcinogen for rubber production https://www.genome.jp/dbget-bin/www_bget?cpd:C19409
                        "Shikimate", #plant and microbial byproduct https://www.sciencedirect.com/topics/chemistry/shikimate
                        "Metoprolol", #pharmaceutical https://www.genome.jp/dbget-bin/www_bget?dr:D02358
                        "Butabarbital", #pharmaceutical https://www.genome.jp/dbget-bin/www_bget?dr:D03180
                        "Olomoucine", #synthetic CDK inhibtor https://www.sciencedirect.com/topics/pharmacology-toxicology-and-pharmaceutical-science/olomoucine
                        "Phenazone", #pharmaceutical https://www.genome.jp/dbget-bin/www_bget?dr:D01776
                        "2-PCB", #synthetic chemical http://www.t3db.ca/toxins/T3D0390
                        "Eperisone", #pharmaceutical https://www.genome.jp/dbget-bin/www_bget?dr:D07898
                        "Isoamyl cyanide", #non-endogenous
                        "Sulforaphane", #phytochemical https://www.sciencedirect.com/topics/medicine-and-dentistry/sulforaphane#:~:text=Sulforaphane%20is%20an%20isothiocyanate%20that,reviewed%20in%20Ho%20et%20al.)
                        "Sulfamethoxazole", #pharmaceutical https://www.genome.jp/dbget-bin/www_bget?dr:D00447
                        "Imagabalin", #pharmaceutical https://www.genome.jp/dbget-bin/www_bget?dr:D09627
                        "AICAR", #pharmaceutical https://www.usada.org/spirit-of-sport/education/aicar-and-other-prohibited-amp-activated-protein-kinase-activators/
                        "Xanthone", #phytochemical https://www.sciencedirect.com/topics/agricultural-and-biological-sciences/xanthone#:~:text=Xanthones%20are%20a%20unique%20class,%2C%20antiviral%2C%20and%20antioxidant%20properties.
                        "(E)-3,4,5-Trimethoxycinnamic acid", #allergen https://pubchem.ncbi.nlm.nih.gov/compound/3_4_5-Trimethoxycinnamic-acid#:~:text=3%2C4%2C5%2Dtrimethoxycinnamic%20acid%20is%20a%20methoxycinnamic%20acid,%2C4%2C5%2Dtrimethoxycinnamate.
                        "DL-Stachydrine", #phytochemical https://www.genome.jp/dbget-bin/www_bget?cpd:C10172
                        "Triphenylphosphine oxide", #synthetic chemical https://www.caymanchem.com/product/9000289/triphenylphosphine-oxide
                        "Aniline", #synethic chemical https://www.atsdr.cdc.gov/toxfaqs/tf.asp?id=449&tid=79
                        "oxiramide", #pharmaceutical https://www.genome.jp/dbget-bin/www_bget?drug+D05303
                        "Vestitol", #phytochemical https://www.genome.jp/dbget-bin/www_bget?cpd:C10540
                        "Dichloromethane", #non-endogenous solvent https://www.genome.jp/dbget-bin/www_bget?cpd:C02271
                        "hypaphorine", #plant metabolite https://pubchem.ncbi.nlm.nih.gov/compound/Hypaphorine
                        "Doxylamine", #anti-histamine https://www.genome.jp/dbget-bin/www_bget?cpd:C19414
                        "Oxycodone", #pharmaceutical https://www.genome.jp/dbget-bin/www_bget?cpd:C08018
                        "Carnidazole", #pharmaceutical https://drugs.ncats.io/drug/RH5KI819JG
                        "4-Methylaminoantipyrine", #pharmaceutical https://pubchem.ncbi.nlm.nih.gov/compound/Noramidopyrine
                        "Metronidazole-OH", #pharmaceutical https://www.sigmaaldrich.com/catalog/product/sial/34007?lang=en&region=US#:~:text=Metronidazole%2DOH%20is%20a%20genotoxic,protozoal%20diseases%20in%20farm%20animals.
                        "Spiroxamine", #pesticide https://www.genome.jp/dbget-bin/www_bget?cpd:C11124
                        "Theophylline", #pharmaceutical https://medlineplus.gov/druginfo/meds/a681006.html#:~:text=Theophylline%20is%20used%20to%20prevent,making%20it%20easier%20to%20breathe.
                        "cyclandelate", #pharmaceutical https://go.drugbank.com/drugs/DB04838
                        "trimethadione", #pharmaceutical https://medlineplus.gov/druginfo/meds/a601127.html
                        "Pyrrolidine, 1-oleoyl-", #non-endogenous chemical https://brumer.com/paper/pyrrolidine
                        "Anhydroecgonine", #related to cocaine https://pubmed.ncbi.nlm.nih.gov/28914428/
                        "Retinol", #non-native vitamin https://www.genome.jp/dbget-bin/www_bget?cpd:C00473
                        "Bergaptol", #phytochemical https://www.genome.jp/dbget-bin/www_bget?cpd:C00758
                        "Clindamycin", #pharmaceutical https://www.genome.jp/dbget-bin/www_bget?cpd:C06914
                        "pronetalol", #synthetic chemical https://pubchem.ncbi.nlm.nih.gov/compound/Naphthalene
                        "3,7,3',4'-Tetramethylquercetin", #plant metabolite https://pubchem.ncbi.nlm.nih.gov/compound/Retusin
                        "Metalaxyl", #fungicide https://pubchem.ncbi.nlm.nih.gov/compound/Metalaxyl
                        "D-Apiitol", #plant metabolite https://hmdb.ca/metabolites/HMDB0029941
                        "3-Chloro-4-nitro-1,2-oxazole", #non-endogenous chemical
                        "L-Hypoglycin", #plant metabolite https://www.genome.jp/dbget-bin/www_bget?cpd:C08287
                        "SAT", #toxin https://pubmed.ncbi.nlm.nih.gov/12117966/
                        "N-(6-Aminohexanoyl)-6-aminohexanoate", #microbial metabolism https://www.genome.jp/dbget-bin/www_bget?cpd:C01255
                        "12-Aminododecanoic acid", #bacterial metabolite https://pubchem.ncbi.nlm.nih.gov/compound/12-Aminododecanoic-acid
                        "Tazobactam" #pharmaceutical https://www.genome.jp/dbget-bin/www_bget?cpd:C07771
                        "allylcysteine", #plant metabolite https://www.sciencedirect.com/topics/nursing-and-health-professions/s-allylcysteine
                        "Carbaryl", #pesticides https://www.genome.jp/dbget-bin/www_bget?cpd:C07491
                        "Imafen", #unknown
                        "Buflomedil", #pharmaceutical https://pubmed.ncbi.nlm.nih.gov/3297620/
                        "Hexylresorcinol", #pharmaceutical https://www.rxlist.com/consumer_sucrets_hexylresorcinol/drugs-condition.htm
                        "Adaprolol", #pharmaceutical https://drugs.ncats.io/drug/2I8RV6WL9A
                        "Carvedilol", #chemical therapeutic https://www.genome.jp/dbget-bin/www_bget?cpd:C06875
                        "5-Fluoro-3,5-AB-PFUPPYCA", #sythetic cannabinoid https://www.caymanchem.com/product/17181
                        "4,6-Dimethyl-2(1H)-pyrimidinone", #pesticide https://pubchem.ncbi.nlm.nih.gov/compound/4_6-Dimethyl-2-hydroxypyrimidine#section=Transformations
                        "Diaveridine", #pharmaceutical https://drugs.ncats.io/drug/7KVX81XA87
                        "Flurandrenolide", #pharmaceutical https://medlineplus.gov/druginfo/meds/a601055.html
                        "N4-Acetylsulfamethoxazole", #pharmaceutical metabolite https://hmdb.ca/metabolites/HMDB0013854
                        "Ferulate", #plant biosynthesis byproduct https://www.sciencedirect.com/topics/food-science/phenylpropanoid
                        "Morphine", #pain killer
                        "Hypoxanthine", #phytochemical https://www.genome.jp/dbget-bin/www_bget?cpd:C00262
                        "Elaeokanine C", #phytochemical https://www.genome.jp/dbget-bin/www_bget?cpd:C10592
                        "Buphedrine", #synthetic https://www.lipomed.com/index.php?section=mediadir&cmd=specification&eid=425
                        "Piperine", #phytochemical https://www.genome.jp/dbget-bin/www_bget?cpd:C03882
                        "Zalcitabine", #carcinogen https://www.genome.jp/dbget-bin/www_bget?cpd:C07207
                        "Butoxytriglycol", #pesticide https://www.dow.com/en-us/pdp.butoxytriglycol-btg.85163z.html
                        "Sulfafurazole", #pharmaceutical https://pubchem.ncbi.nlm.nih.gov/compound/sulfisoxazole
                        "L-alpha-lysophosphatidylcholine", #egg yolk product https://www.sigmaaldrich.com/catalog/product/sigma/l4129?lang=en&region=US
                        "Isoniazid pyruvate", #drug metabolism https://www.genome.jp/dbget-bin/www_bget?cpd:C16624
                        "Etodolac", #pharmaceutical https://www.genome.jp/dbget-bin/www_bget?cpd:C06991
                        "Methohexital", #pharmaceutical https://www.genome.jp/dbget-bin/www_bget?cpd:C07844
                        "Brassylic acid", #synthetic http://www.chemspider.com/Chemical-Structure.10026.html
                        "Formononetin", #flavanoid https://www.genome.jp/dbget-bin/www_bget?cpd:C00858
                        "4-Aminopyridine", #pharmaceutical https://pubchem.ncbi.nlm.nih.gov/compound/4-aminopyridine
                        "N-Methyl-2-pyrrolidone", #synthetic https://www.eastman.com/Pages/ProductHome.aspx?product=71103627&pn=N-Methyl-2-Pyrrolidone+(NMP)
                        "1-Methylxanthine", #phystochemical https://www.genome.jp/dbget-bin/www_bget?cpd:C16358
                        "2-Amino-6-hydroxyaminopurine", #mutagen https://pubchem.ncbi.nlm.nih.gov/compound/2-Amino-N6-hydroxyadenine
                        "4-Coumarylalcohol(4-Hydroxycoumarin)", #aromatic alcohol https://hmdb.ca/metabolites/HMDB0003654
                        "1-(beta-D-Ribofuranosyl)-1_4-dihydronicotinamide", #plant metabolism https://www.genome.jp/dbget-bin/www_bget?cpd:C03741
                        "2-Hydroxyhippuric acid", #pharmaceutical https://pubmed.ncbi.nlm.nih.gov/3888490/
                        "4-Acetamidoantipyrine", #pharmaceutical byproduct https://pubchem.ncbi.nlm.nih.gov/compound/4-Acetamidoantipyrine
                        "4-Aminohippuricacid", #non-endogenous https://pubchem.ncbi.nlm.nih.gov/compound/Aminohippuric-acid
                        "8-Amino-7-oxononanoate", #biotin metabolism https://www.genome.jp/dbget-bin/www_bget?cpd:C01092
                        "8-Hydroxyquinoline", #synthetic compound https://www.sigmaaldrich.com/catalog/product/sial/252565?lang=en&region=US
                        "Acetonecyanohydrin", #toxic https://www.genome.jp/dbget-bin/www_bget?cpd:C02659
                        "Anhydrotetracycline", #pharmaceutical https://www.takarabio.com/products/transfection-and-cell-culture/antibiotic-selection/anhydrotetracycline
                        "Beta-Alanine", #non-native amino acid https://www.webmd.com/vitamins/ai/ingredientmono-1222/beta-alanine
                        "Cycloheptene", #pharmaceutical https://www.sciencedirect.com/topics/pharmacology-toxicology-and-pharmaceutical-science/cycloheptene
                        "Ecgonine", #coca leaf derivative https://pubchem.ncbi.nlm.nih.gov/compound/Ecgonine
                        "Kanosamine", #antibiotic https://biocyc.org/META/NEW-IMAGE?type=PATHWAY&object=PWY-5978
                        "L-Threonic acid", #plant metabolite https://pubchem.ncbi.nlm.nih.gov/compound/L-Threonic-acid
                        "Methotrexate", #carcinogen https://www.genome.jp/dbget-bin/www_bget?cpd:C01937
                        "N-Nitrosoguvacoline", #carcinogen https://www.genome.jp/dbget-bin/www_bget?cpd:C19482
                        "Pitavastatin", #pharmaceutical https://medlineplus.gov/druginfo/meds/a610018.html
                        "Porphobilinogen", #plant compound https://www.genome.jp/dbget-bin/www_bget?cpd:C00931
                        "tebuthiuron", #pesticides https://www.genome.jp/dbget-bin/www_bget?cpd:C18436
                        "Xanthohumol", #flavanoid https://www.sciencedirect.com/topics/medicine-and-dentistry/xanthohumol
                        "12-Hydroxydihydrochelirubine", #antifungal derivative https://pubchem.ncbi.nlm.nih.gov/compound/sanguinarine
                        "2-(tert-butylamino)-1-(3-chlorophenyl)propan-1-ol", #pharmaceutical https://www.sciencedirect.com/topics/medicine-and-dentistry/hydroxybupropion
                        "2-Naphthylamine", #carcinogen https://www.genome.jp/dbget-bin/www_bget?cpd:C02227
                        "4-Aminophenol", #microbial metabolism https://www.genome.jp/dbget-bin/www_bget?cpd:C02372
                        "4-HQN", #parp inhibitor https://www.adooq.com/4-hqn.html
                        "4-styrenesulfonic acid", #synthetic compound https://www.sigmaaldrich.com/catalog/product/aldrich/328596?lang=en&region=US#:~:text=4%2DStyrenesulfonic%20acid%20sodium%20salt%20hydrate%20is%20a%20water%20soluble,homopolymers%20with%20other%20vinyl%20monomers.
                        "6-Methylindole", #pharmaceutical building block https://www.goldbio.com/product/13683/6-methylindole
                        "Aceclidine", #pharmaceutical https://pubchem.ncbi.nlm.nih.gov/compound/Aceclidine
                        "Acycloguanosine", #synthetic https://www.goldbio.com/product/13165/acycloguanosine
                        "Alprenolol", #pharmaceutical https://go.drugbank.com/drugs/DB00866
                        "Baclofen", #muscle relaxant https://medlineplus.gov/druginfo/meds/a682530.html
                        "Benzyl butyl phthalate", #synthetic https://www.genome.jp/dbget-bin/www_bget?cpd:C14211
                        "Caffeine", #non-endogenous 
                        "Caprolactam", #synthetic https://pubchem.ncbi.nlm.nih.gov/compound/Caprolactam
                        "Carbamazepine 10,11-epoxide", #drug metabolism https://www.genome.jp/dbget-bin/www_bget?cpd:C07496
                        "cis-Tramadol", #opioid https://www.caymanchem.com/product/15919/cis-tramadol-(hydrochloride)
                        "Coniferylalcohol", #plant metabolite https://www.genome.jp/dbget-bin/www_bget?cpd:C00590
                        "Coniferylaldehyde", #plant metabolite https://www.genome.jp/dbget-bin/www_bget?cpd:C02666
                        "Coumarone", #synthetic http://www.chemspider.com/Chemical-Structure.8868.html
                        "Cyclohexylamine", #bacterial  https://www.genome.jp/dbget-bin/www_bget?cpd:C00571
                        "Dethiobiotin", #biotin metabolism https://www.genome.jp/dbget-bin/www_bget?cpd:C01909
                        "Dextromethorphan", #cough medicine https://www.genome.jp/dbget-bin/www_bget?cpd:C06947
                        "Dibekacin", #antibiotic https://pubchem.ncbi.nlm.nih.gov/compound/Dibekacin
                        "Dihydroxycarbamazepine", #drug metabolism genome.jp/dbget-bin/www_bget?cpd:C07495
                        "Diisobutylphthalate", #synthetic https://pubchem.ncbi.nlm.nih.gov/compound/Diisobutyl-phthalate
                        "Diphenhydramine", #pharmaceutical https://www.genome.jp/dbget-bin/www_bget?cpd:C06960
                        "Dithranol", #carcinogens https://www.genome.jp/dbget-bin/www_bget?cpd:C06831
                        "Escitalopram", #antidepressant 
                        "Gabapentin", #antidepressant https://www.genome.jp/dbget-bin/www_bget?cpd:C07018
                        "Gluconic acid", #synthetic https://www.sciencedirect.com/topics/agricultural-and-biological-sciences/gluconic-acid
                        "GLUCONOLACTONE", #food additive 
                        "Haplopine", #phytochemical https://www.genome.jp/dbget-bin/www_bget?cpd:C10694
                        "hydralazine", #carcinogen https://www.genome.jp/dbget-bin/www_bget?cpd:C07040
                        "Inosine", #pharmaceutical https://www.genome.jp/dbget-bin/www_bget?cpd:C00294
                        "Iohexol", #pharmaceutical https://www.rxlist.com/omnipaque-drug.htm
                        "Mannitol", #synthetic sugar 
                        "Metronidazole", #pharmaceutical https://www.webmd.com/drugs/2/drug-6426/metronidazole-oral/details
                        "Nisoxetine", #synthetic https://www.sciencedirect.com/topics/neuroscience/nisoxetine
                        "Nitrosoheptamethyleneimine", #carcinogen https://academic.oup.com/jnci/article-abstract/69/5/1127/943819
                        "Norketamine", #ketamine metabolite https://pubmed.ncbi.nlm.nih.gov/9311667/
                        "Omeprazole sulphone", #pharmaceutical derivative https://www.caymanchem.com/product/18882/omeprazole-sulfone#:~:text=Omeprazole%20sulfone%20is%20the%20major,14880).&text=Omeprazole%20sulfone%20has%20been%20shown,IC50%20%3D%2018%20%C2%B5M).
                        "Ondansetron", #pharmaceutical https://www.genome.jp/dbget-bin/www_bget?cpd:C07325
                        "Oseltamivir", #pharmaceutical https://www.genome.jp/dbget-bin/www_bget?cpd:C08092
                        "Quinoline", #pharmaceutical https://pubmed.ncbi.nlm.nih.gov/9719345/#:~:text=Abstract,mainstays%20of%20chemotherapy%20against%20malaria.&text=Chloroquine%2C%20a%20dibasic%20drug%2C%20is,fold%20in%20the%20food%20vacuole.
                        "Ropivacaine", # pharmaceutical https://www.genome.jp/dbget-bin/www_bget?cpd:C07532
                        "Tolperisone", #pharmaceutical https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4133921/
                        "tranexamic acid", #pharmaceutical https://www.genome.jp/dbget-bin/www_bget?cpd:C12535
                        "Venlafaxine", #pharmaceutical https://www.genome.jp/dbget-bin/www_bget?cpd:C07187
                        "Thiamin", #plant derivative vitamin 
                        "N-Benzylformamide", #non-endogenous 
                        "2-Methylthiazolidine", #drug related
                        "Oxohongdenafil", #non-endogenous https://www.researchgate.net/publication/264096924_Elucidation_of_new_anti-impotency_analogue_in_food
                        "Propionylcarnitine", #drug https://pubchem.ncbi.nlm.nih.gov/compound/Propionylcarnitine
                        "N3,N4-Dimethyl-L-arginine", #drug https://hmdb.ca/metabolites/HMDB0003334
                        "2-Hydroxyethanesulfonate", #surfactant https://hmdb.ca/metabolites/HMDB0003903
                        "1-palmitoylglycerol;MAG(16:0)", #plant metabolite https://www.ebi.ac.uk/chebi/searchId.do?chebiId=CHEBI:69081
                        "allylcysteine", #nonendogenous https://www.sciencedirect.com/topics/neuroscience/s-allyl-cysteine
                        "6-Hydroxypseudooxynicotine", #non endogenous nicotine derivative 
                        "D-Fructose", #non endogenous dietary intake based https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5893377/
                        "5-Acetylamino-6-amino-3-methyluracil", #caffeine metabolite breakdown byproduct https://hmdb.ca/metabolites/HMDB0004400
                        "meso-Tartaric acid", #non endogenous fruit byproduct https://hmdb.ca/metabolites/HMDB0000956
                        "(-)-trans-Carveol", #plant metabolite https://pubchem.ncbi.nlm.nih.gov/compound/trans-Carveol
                        "Nicotianamine", #plant metabolites https://www.nature.com/articles/s41598-020-57598-3
                        "(S)-4-Amino-5-oxopentanoate", #non endogenous https://pubchem.ncbi.nlm.nih.gov/compound/S_-4-Amino-5-oxopentanoate   
                        "Dihydropteroate", #non endogenous https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3531234/
                        "Dibenzoylmethane", #non endogenous https://pubmed.ncbi.nlm.nih.gov/11867200/
                        "2-linoleoyl-sn-glycero-3-phosphoethanolamine", #plant based metabolite derived from linoleic acid https://pubchem.ncbi.nlm.nih.gov/compound/linoleic%20acid
                        "Thiolactomycin", #non endogenous antibacterial https://www.sciencedirect.com/topics/pharmacology-toxicology-and-pharmaceutical-science/thiolactomycin#:~:text=Thiolactomycin%2C%20a%20potent%20inhibitor%20of,biosynthesis%2C%20inhibits%20growth%20of%20P.&text=Sulfonamides%20are%20inhibitors%20of%20H2,and%20trimethoprim%20inhibit%20dihydrofolate%20reductase.
                        "Cyclohexanethiol", # non endogenous liquid https://pubchem.ncbi.nlm.nih.gov/compound/Cyclohexanethiol 
                        "O-Phospho-L-homoserine", #bacterial metabolite https://pubchem.ncbi.nlm.nih.gov/compound/o-Phosphohomoserine
                        "2-Hydroxypyridine", #non endogenous used for peptide synthesis https://hmdb.ca/metabolites/HMDB0013751
                        "Quinic acid", #plant extract, bitter taste of coffee https://pubchem.ncbi.nlm.nih.gov/compound/Quinic-acid 
                        "3-CYSTEINYLACETAMINOPHEN", #acetaminophen stereoisomer
                        "Phenylisocyanate", #synthetic chemical for cellulose paper preparation https://www.sigmaaldrich.com/catalog/product/aldrich/185353?lang=en&region=US
                        "2-Aminoacetophenone;O-Acetylaniline", #found in food https://hmdb.ca/metabolites/HMDB0032630
                        "2-Hydroxyquinoline", #quinoline breakdown byproduct 
                        "Cerulenin", #anti-fungal antibiotic
                        "Guvacoline", #plant metabolite https://pubchem.ncbi.nlm.nih.gov/compound/Guvacoline
                        "Maraniol", #synthetic chemical
                        "NDA",
                        "BiochaninA"]


# In[8]:


#identify signficantly differential metabolites for both groups that are endogenous
endogenous_features = [i for i in keep if i not in non_endogenous_metabolites]
severe_end = [i for i in severe_mets if i not in non_endogenous_metabolites]
nonacute_end = [i for i in non_acute_mets if i not in non_endogenous_metabolites]

print(str(len(endogenous_features))+ ' Total Endogenous Metabolites')
print(str(len(severe_end)) + ' Total Severe Metabolites ')
print(str(len(nonacute_end)) + ' Non-acute Metabolites')


# In[9]:


'''
# write text file of endogenous severe metabolites 
import csv

f=open('severe_endogenous.txt','w')
for element in severe_end:
    f.write(element+'\n')

f.close()

# write text file of endogenous non-acute metabolites 
import csv

f=open('nonacute_endogenous.txt','w')
for element in nonacute_end:
    f.write(element+'\n')

f.close()
'''


# In[9]:


# Run random forest using ONLY endogenous differential metabolites 

#read in data 
data = pd.read_csv("severe_vs_nonacute.csv")
data.groupby(['group']).size()

#select data 
X = data[endogenous_features]
y = data["group"]

#standardize the dataset
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# split into train and test set
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, stratify=y, test_size=0.3, random_state=42
)

# create the classifier
classifier = RandomForestClassifier(n_estimators=1500)

# Train the model using the training sets
classifier.fit(X_train, y_train)

# prediction on the test set
y_pred = classifier.predict(X_test)

# Calculate Model Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))


# In[16]:


# export groups as csv
y.to_csv("groups.csv", index=False)

#export endogenous metabolomics to csv
X.to_csv("endogenous_metabolomics.csv", index=False)


# In[19]:


# check Important features
feature_importances_df = pd.DataFrame(
    {"feature": list(X.columns), "importance": classifier.feature_importances_}
).sort_values("importance", ascending=False)

# visualize important featuers
fig = plt.figure(figsize = (8,8))

# Creating a bar plot
sns.scatterplot(x=feature_importances_df.feature.head(20), y=feature_importances_df.importance.head(20), color = "darkslategray", s = 50, marker = 's')

# Add labels to your plot
plt.xlabel("Features", fontsize = 20)
plt.ylabel("Feature Importance Score", fontsize = 20)
plt.title("Random Forest Top 20 Features", fontsize = 20)
plt.xticks(
    rotation=45, horizontalalignment="right", fontweight="light", fontsize= 15
)

plt.show()
fig.savefig('randomforest_severevsnonacute.png', bbox_inches='tight')


# In[20]:


# ROC curves for comparing number of features to prediction acuracy 
# https://scikit-learn.org/stable/visualizations.html

fig = plt.figure(figsize = (8,8))

rfc1 = RandomForestClassifier(n_estimators=5, random_state=42)
rfc1.fit(X_train, y_train)
ax = plt.gca()
rfc1_disp = plot_roc_curve(rfc1, X_test, y_test, ax=ax, alpha=0.8)

rfc = RandomForestClassifier(n_estimators=10, random_state=42)
rfc.fit(X_train, y_train)
ax = plt.gca()
rfc_disp = plot_roc_curve(rfc, X_test, y_test, ax=ax, alpha=0.8)

rfc2 = RandomForestClassifier(n_estimators=25, random_state=42)
rfc2.fit(X_train, y_train)
ax = plt.gca()
rfc2_disp = plot_roc_curve(rfc2, X_test, y_test, ax=ax, alpha=0.8)

L=plt.legend(loc = 4, fontsize = 15)
L.get_texts()[0].set_text('Top 5 Features (AUC = 0.80)')
L.get_texts()[1].set_text('Top 10 Features (AUC = 0.87)')
L.get_texts()[2].set_text('Top 25 Features (AUC = 0.84)')

plt.xlabel("Specificity", fontsize = 20)
plt.ylabel("Sensitivity", fontsize = 20)
plt.title("Random Forest ROC Curve", fontsize = 20)

plt.show()
fig.savefig('ROC.png', bbox_inches='tight')


# In[ ]:


#NMDS analysis 
#bray curtis dissimilarity matrix --> designed for ecology so better for lots of measures 

similarities = pairwise_distances(X_scaled, metric = 'braycurtis')

mds = manifold.MDS(n_components=2, max_iter=3000, eps=1e-9,
                   dissimilarity="precomputed", n_jobs=1)
pos = mds.fit(similarities).embedding_

nmds = manifold.MDS(n_components=2, metric=False, max_iter=3000, eps=1e-12,
                    dissimilarity="precomputed", n_jobs=1,
                    n_init=1)
npos = nmds.fit_transform(similarities, init=pos)


# In[ ]:


#Create NMDS plot with labels based on covid status
finalDf = pd.concat([pd.DataFrame(npos), pd.DataFrame(y)], axis = 1)
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_title('COVID-19 Disease Status NMDS', fontsize = 20)
groups = finalDf.groupby("group")
for name, group in groups:
    plt.plot(group[0], group[1], marker="o", linestyle="", label=name)
#plt.plot([], [], ' ', label="p-value: 0.002")
plt.legend(bbox_to_anchor=(1.37, 1), fontsize = 12)
L=plt.legend()
L.get_texts()[0].set_text('Non-Acute COVID-19')
L.get_texts()[1].set_text('Severe COVID-19')
fig.savefig('NMDS_severevsnonacute.png', bbox_inches='tight')

#PERMANOVA run using vegan package adonis function with following line of code --> p-value: 0.002
# adon.results<-adonis(df.mat ~ group, method="bray",perm=999)
# df. mat = endogenous metabolomics data 
# group = COVID severity group 


# In[ ]:


#pathway analysis 

#Severe covid: KEGG database
#histidine metabolism: p-value 0.01 --> -Imidazolone-5-propionic acid, Methylimidazoleacetic acid
#Synthesis and degradation of ketone bodies: p-value 0.0484 --> Acetoacetic acid

#non-acute covid: KEGG database
#tryptophan metabolism: p-value 0.00313 --> L-Tryptophan, Melatonin, 5-Hydroxy-L-tryptophan,  3-Hydroxyanthranilic acid,
#Indoleacetaldehyde, 2-Aminobenzoic acid

#glutathione metabolism: p-value 0.0807 --> Oxidized glutathione, Pyroglutamic acid, Spermine

