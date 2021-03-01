#!/usr/bin/env python
# coding: utf-8

# In[17]:


import torch
from torch.utils.data import DataLoader
import gentrl
import gentrl.lp
from gentrl.tokenizer import encode, get_vocab_size
from gentrl.new_dataloader import NewMolecularDataset
import time
import torch.multiprocessing as mp


# ## Set GPU

# In[5]:


torch.cuda.set_device(3)


# In[6]:


RANDOM_SEED = 42


# In[7]:


enc = gentrl.RNNEncoder(latent_size=50)
dec = gentrl.DilConvDecoder(latent_input_size=50)
model = gentrl.GENTRL(enc, dec, latent_descr=50 * [('c', 20)], feature_descr=[('c', 20)], beta=0.001)
# this moves the model to GPU
model.to('cuda');


# In[8]:


device = torch.device('cuda')


# In[9]:


md = NewMolecularDataset(device=device,
    sources=[
    {'path':'train_subset_100_000.csv',
     'smiles': 'SMILES',
     'prob': 1,
     'plogP' : 'plogP',
    }], 
    props=['plogP'])


# In[10]:


gpu_dataset = md.create_tensor_dataset()


# In[18]:


def main():
    BATCH_SIZE = 512
    LR = 1e-4
    NUM_EPOCHS = 1
    NUM_WORKERS = 4
    PIN_MEMORY= False
    train_loader = DataLoader(gpu_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=True,
                          num_workers=NUM_WORKERS,
                          pin_memory=PIN_MEMORY,
                          drop_last=True)
    start_time = time.time()
    model.train_as_vaelp(train_loader, lr=LR, num_epochs=NUM_EPOCHS)
    end_time = time.time()
    duration = end_time - start_time
    print(f"Time Taken is {duration/60} min")
    
if __name__=='__main__':
    mp.set_start_method('spawn')
    main()


# In[ ]:


# from moses.metrics import mol_passes_filters, QED, SA, logP
# from moses.metrics.utils import get_n_rings, get_mol


# def get_num_rings_6(mol):
#     r = mol.GetRingInfo()
#     return len([x for x in r.AtomRings() if len(x) > 6])


# def penalized_logP(mol_or_smiles, masked=False, default=-5):
#     mol = get_mol(mol_or_smiles)
#     if mol is None:
#         return default
#     reward = logP(mol) - SA(mol) - get_num_rings_6(mol)
#     if masked and not mol_passes_filters(mol):
#         return default
#     return reward


# In[ ]:


# df = pd.read_csv('dataset_v1.csv')
# df = df[df['SPLIT'] == 'train']
# df['plogP'] = df['SMILES'].apply(penalized_logP)
# df.to_csv('train_plogp_plogpm.csv', index=None)


# In[ ]:


# ! mkdir -p saved_gentrl


# In[ ]:


# model.save('./saved_gentrl/')

