import strip_markdown
import os
import pickle
import requests
import numpy as np

#On convertit tout le dossier markdowns vers texts. Pour rajouter des textes de loi, il suffit de mettre un nouveau .md dans markdowns et run les lignes commentées

md_dir = os.getcwd()+"/data/gendp/markdowns"
txt_dir = os.getcwd()+"/data/gendp/texts"

""" Uncomment pour remplir le dossier texts à partir du dossier markdowns
for md_filename in os.listdir(md_dir):
    input_file_path = os.path.join(md_dir, md_filename)
    str = strip_markdown.strip_markdown_file(input_file_path, text_fn= os.getcwd()+"/data/gendp/texts") """

#concatenate every txt to one file, which will be the input.txt
    
total_input = []    
for txt_filename in os.listdir(txt_dir):
    if txt_filename.endswith(".txt"):
        total_input.append(os.path.join(txt_dir, txt_filename))

with open("data/gendp/input.txt","w", encoding="utf-8") as outputfile:
    for name in total_input:
        outputfile.write("\n\n") #Deux sauts entre chaque texte
        with open(name, encoding="utf-8", errors='ignore') as inputfile:
            for line in inputfile:
                outputfile.write(line)

#En s'inspirant de nanoGPT : 
"""
Prepare the law dataset for character-level language modeling.
So instead of encoding with GPT-2 BPE tokens, we just map characters to ints.
Will save train.bin, val.bin containing the ids, and meta.pkl containing the
encoder and decoder and some other related info.
"""                                

text_path = os.path.join(os.path.dirname(__file__), 'input.txt')
with open(text_path, 'r') as f:
    data = f.read()
print(f"length of dataset in characters: {len(data):,}")

# get all the unique characters that occur in this text
chars = sorted(list(set(data)))
vocab_size = len(chars)
print("all the unique characters:", ''.join(chars))
print(f"vocab size: {vocab_size:,}")

# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
def encode(s):
    return [stoi[c] for c in s] # encoder: take a string, output a list of integers
def decode(l):
    return ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# create the train and test splits
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

# encode both to integers
train_ids = encode(train_data)
val_ids = encode(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# save the meta information as well, to help us encode/decode later
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

#Paramètres du dataset de droit :
# train.bin is ~ 250 MB, val.bin is ~ 28 MB
#length of dataset in characters: 141,595,341
#all the unique characters: 
# !"#%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[]^_abcdefghijklmnopqrstuvwxyz{|} ¡¢£¤¥¦§¨©ª«¬­®¯°±²³´µ¶·¸¹º»¼½¾¿ÂÃÄÅÊËÌÎÏÐÑáâïŒœŠŸŽƒˆ˜–—‘’‚“”„†‡•…‰‹›€™
#vocab size: 163
#train has 127,435,806 tokens
#val has 14,159,535 tokens             
    
#Ces paramètres sont assez conséquents en taille, donc on a intérêt à faire un bon modèle.