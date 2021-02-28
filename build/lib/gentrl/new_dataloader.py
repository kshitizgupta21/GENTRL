import torch
from torch.utils.data import TensorDataset
from gentrl.tokenizer import encode
import pandas as pd
import numpy as np


# so the plan here is to have 
class NewMolecularDataset:
    def __init__(self, device, sources=[], props=['logIC50', 'BFL', 'pipeline'],
                 with_missings=False):
        self.num_sources = len(sources)
        self.device = device
        self.source_smiles = []
        self.source_props = []
        self.source_missings = []
        self.source_probs = []

        self.with_missings = with_missings

        self.len = 0
        # this whole for loop is useless because we only have 1 source (i.e)
        # only 1 source dict
        for source_descr in sources:
            cur_df = pd.read_csv(source_descr['path'])
            cur_smiles = list(cur_df[source_descr['smiles']].values)
            num_smiles = len(cur_smiles)
            num_props = len(props)
            cur_props = torch.zeros(num_smiles, num_props, device=self.device)
            cur_missings = torch.zeros(num_smiles, num_props, dtype=torch.int64, device=self.device)

            for i, prop in enumerate(props):
                if prop in source_descr:
                    if isinstance(source_descr[prop], str):
                        cur_props[:, i] = torch.from_numpy(
                            cur_df[source_descr[prop]].values)
                        # so this is where we read plogp from dataframe
                        # and set it to cur_props
                        # these are our labels
                        # currently cur_props is tensor on CPU
                    else:
                        cur_props[:, i] = torch.from_numpy(
                            cur_df[source_descr['smiles']].map(
                                source_descr[prop]).values)
                else:
                    cur_missings[:, i] = 1
            

            self.source_smiles.append(cur_smiles)
            self.source_props.append(cur_props)
            self.source_missings.append(cur_missings)
            self.source_probs.append(source_descr['prob'])

            self.len = max(self.len, int(num_smiles / source_descr['prob']))

        self.source_probs = np.array(self.source_probs).astype(np.float)

        self.source_probs /= self.source_probs.sum()
        
        
    def create_tensor_dataset(self):    
        trial = np.random.random()

        s = 0
        # here self.num_sources =1 
        for i in range(self.num_sources):
            # here self.source_probs = np.array([1])
            # so self.source_probs[0] = 1
            # so for s = 0 this if condition will always be true
            if (trial >= s) and (trial <= s + self.source_probs[i]):
                # here bin_len is same as num_smiles
                #bin_len = len(self.source_smiles[i])
                
                # here sm is just idx_th SMILE string
                #sm = self.source_smiles[i][idx % bin_len]
                sm_list = self.source_smiles[i]
                # here self.source_props[0] is cur_props
                # so props is the cur_prop value corresponding to 
                # idx_th SMILE string
                # props = self.source_props[i][idx % bin_len]
                props = self.source_props[i]
                # here self.source_missings[0] is cur_missings
                # so miss is the cur_missings value corresponding to 
                # idx_th SMILE string
                # miss = self.source_missings[i][idx % bin_len]
                miss = self.source_missings[i]
                y = props
                if self.with_missings:
                    y = torch.concat([props, miss])
            # so getitem just returns (idx_th SMILE string, idx_th prop value)
            s += self.source_probs[i]
        
        # so now we have sm_list and their labels (y values)
        # we need to tokenize the SMILES string in sm_list and convert the tokens
        # to indices
        
        tokens, smile_lens = encode(sm_list, device=self.device)

        dataset = TensorDataset(tokens, smile_lens, y)
        
        return dataset
        
        
            

