import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.feature_extraction import FeatureHasher
from vars_local import *

def prepare_features(ds):
  ds  = ds.drop('Tumor_Sample_ID', axis = 1)

  variant_data     = ds.Variant_Type.values;
  gene_name_data   = ds.Gene_Name.values;
  reference_allele = ds.Reference_Allele.values;
  tumor_Allele     = ds.Tumor_Allele.values;
  chromosome       = ds.Chromosome.values;
  start_position   = ds.Start_Position.values;

  ds = ds.drop('Chromosome', axis = 1)
  ds = ds.drop('Gene_Name', axis = 1)
  ds = ds.drop('Variant_Type', axis = 1)
  ds = ds.drop('Reference_Allele', axis = 1)
  ds = ds.drop('Tumor_Allele', axis = 1)

  l_encoder  = LabelEncoder()
  OH_encoder = OneHotEncoder(sparse_output = False)
  hasher     = FeatureHasher(n_features = 100, input_type = 'string')

  # Diagnosis values are strings. Changing them into numerical values using LabelEncoder.
  ds = pd.concat([ds, pd.DataFrame(OH_encoder.fit_transform(l_encoder.fit_transform(variant_data).reshape(-1,1)))], axis=1)
  ds_chrome = pd.DataFrame(OH_encoder.fit_transform(l_encoder.fit_transform(chromosome).reshape(-1,1)))
  ds = pd.concat([ds, ds_chrome], axis=1)
  pos = pd.cut(start_position, bins=START_POSITION_BINS, labels=range(START_POSITION_BINS))
  ds = pd.concat([ds, pd.DataFrame(OH_encoder.fit_transform(np.asarray(pos).reshape(-1,1)))], axis=1)
  ds = pd.concat([ds, pd.DataFrame(hasher.transform([[g] for g in gene_name_data.astype(str)]).toarray())], axis=1)

  return ds