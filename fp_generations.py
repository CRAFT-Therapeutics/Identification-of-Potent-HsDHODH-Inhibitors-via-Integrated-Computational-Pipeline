import os
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.MACCSkeys import GenMACCSKeys
from rdkit.Chem import rdMolDescriptors
import pandas as pd

def generate_cfps(mol, fp_type, diameter, nbits):
    if fp_type == 'ecfp':
        return AllChem.GetMorganFingerprintAsBitVect(mol, radius=diameter, nBits=nbits, useFeatures=False)
    elif fp_type == 'fcfp':
        return AllChem.GetMorganFingerprintAsBitVect(mol, radius=diameter, nBits=nbits, useFeatures=True)
    else:
        raise ValueError("Invalid fingerprint type.")

def generate_maccs(mol):
    return GenMACCSKeys(mol)

def generate_fp(input_folder='input', output_base='fp-gen'):
    os.makedirs(output_base, exist_ok=True)
    for file in os.listdir(input_folder):
        file_path = os.path.join(input_folder, file)
        outcome = []
        if file.lower().endswith('.sdf'):
            suppl = Chem.SDMolSupplier(file_path)
            mols = [mol for mol in suppl if mol is not None]
            # Generate SMILES instead of IDs
            smiles = [Chem.MolToSmiles(mol) for mol in mols]
            # Tenta ler Outcome do campo SDF
            for mol in mols:
                if mol is not None and mol.HasProp('Outcome'):
                    outcome.append(mol.GetProp('Outcome'))
                else:
                    outcome.append(None)
        elif file.lower().endswith('.csv'):
            df_csv = pd.read_csv(file_path)
            smiles_input = df_csv['SMILES'].tolist()
            mols = [Chem.MolFromSmiles(smi) for smi in smiles_input]
            mols = [mol for mol in mols if mol is not None]
            # Keep SMILES for valid molecules
            smiles = [Chem.MolToSmiles(mol) for mol in mols]
            # Ajusta Outcome para corresponder aos smiles válidos
            import numpy as np
            valid_idx = [i for i, mol in enumerate([Chem.MolFromSmiles(smi) for smi in df_csv['SMILES']]) if mol is not None]
            outcome = df_csv['Outcome'].iloc[np.array(valid_idx)].tolist() if 'Outcome' in df_csv.columns else [None]*len(mols)
        else:
            continue

        # Generate fingerprints for both bit sizes and different diameters
        for fp_type in ['ecfp', 'fcfp']:
            for diameter in [1, 2, 3]:
                for nbits in [1024, 2048]:
                    fps = [generate_cfps(mol, fp_type, diameter, nbits) for mol in mols]
                    # Convert fingerprint to bit string
                    fps_bits = [list(fp.ToBitString()) for fp in fps]
                    df = pd.DataFrame(fps_bits)
                    df.insert(0, 'SMILES', smiles)  # Changed back to SMILES
                    df.insert(1, 'Outcome', outcome)
                    out_dir = os.path.join(output_base, fp_type)
                    os.makedirs(out_dir, exist_ok=True)
                    out_file = os.path.join(out_dir, f"{os.path.splitext(file)[0]}_{fp_type}_diam{diameter*2}_{nbits}bits.csv")
                    df.to_csv(out_file, index=False)

        # MACCS fingerprints (always 167 bits - fixed size)
        maccs_fps = [generate_maccs(mol) for mol in mols]
        maccs_bits = [list(fp.ToBitString()) for fp in maccs_fps]
        df_maccs = pd.DataFrame(maccs_bits)
        df_maccs.insert(0, 'SMILES', smiles)  # Changed back to SMILES
        df_maccs.insert(1, 'Outcome', outcome)
        out_dir = os.path.join(output_base, 'maccs')
        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(out_dir, f"{os.path.splitext(file)[0]}_maccs.csv")
        df_maccs.to_csv(out_file, index=False)

if __name__ == "__main__":
    generate_fp()