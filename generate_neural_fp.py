#!/usr/bin/env python

import sys
import pandas as pd
from NGF.models import *
from glob import glob
from rdkit import Chem

def valence_ok(smiles):
    mol = Chem.MolFromSmiles(smiles)
    for atm in mol.GetAtoms():
        if atm.GetTotalDegree() > 4:
            return False
    return True

#neural_fp throws and exception when valence is >4. Remove any SMILES with hypervalent atoms
def validate_smiles(smiles_list):
    return [valence_ok(smi) for smi in smiles_list]


for infile_name in glob("data/*.smi"):
    print(infile_name)
    df = pd.read_csv(infile_name,sep=" ",header=None)
    df.columns = ["SMILES","Name","Act"]
    print(df.shape)
    df['valid'] = validate_smiles(list(df.SMILES))
    df = df.query("valid == True")
    NG_fp = get_ngf(list(df.SMILES), model_type='1',fp_length=248)
    NG_df = pd.DataFrame(NG_fp)
    _,cols = NG_df.shape
    NG_df.columns = ["D%03d" % x for x in range(0,cols)]
    NG_df.insert(0,"Act",df.Act)
    NG_df.insert(0,"Name",df.Name)
    NG_df.insert(0,"SMILES",df.SMILES)
    NG_df.to_csv(infile_name.replace("smi","csv").replace("data","ngfp_data"),index=False)
    




             

    
