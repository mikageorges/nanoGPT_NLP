"""
Sample from a trained model
"""
import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
import sys
sys.path.insert(0,"..")
from model import GPTConfig, GPT
import sqlite3
from flask import Flask, render_template, request, url_for, flash, redirect
from werkzeug.exceptions import abort

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your secret key'

def get_db_connection():
    conn = sqlite3.connect('database.db')
    conn.row_factory = sqlite3.Row
    return conn

def get_sample(sample_id):
    conn = get_db_connection()
    post = conn.execute('SELECT * FROM gen_samples WHERE id = ?',(sample_id,)).fetchone()
    conn.close()
    if post is None:
        abort(404)
    return post

@app.route('/')
def index():
    sample = get_sample(1) #first one
    return render_template('index.html', sample=sample)

@app.route('/model', methods=["GET", "POST"])
def model():
    
    return render_template('model.html')

@app.route('/generate', methods=["GET", "POST"])
def generate_sample(init_from = 'resume', out_dir = 'out-gendp', start ="\n", num_samples = 1,
                     max_new_tokens = 500, temperature = 0.8, top_k = 200, seed = 1337, device = 'cuda', dtype='bfloat16',
                     compile = False):
    # -----------------------------------------------------------------------------
    """ init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
    out_dir = 'out-gendp' # ignored if init_from is not 'resume'
    start = "\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
    num_samples = 10 # number of samples to draw
    #Comment if not using flask for presentation
    #TODO : num_samples equals user input on index.html
    max_new_tokens = 500 # number of tokens generated in each sample
    temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
    top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
    seed = 1337
    device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
    compile = False # use PyTorch 2.0 to compile the model to be faster """
    #exec(open('configurator.py').read()) # overrides from command line or config file
    # -----------------------------------------------------------------------------
    samples=[]
    # -----------------------------------------------------------------------------
    if request.method == 'POST':
        samples = []
        num_samples = request.form['num_samples']
        max_new_tokens = request.form['size_samples']

        if not num_samples or not max_new_tokens:
            flash('Informations manquantes, veuillez réessayer.')
        else:

            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
            torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
            device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
            ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
            ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

            # model
            if init_from == 'resume':
                # init from a model saved in a specific directory
                os.chdir("..")
                ckpt_path = os.path.join(out_dir, 'ckpt.pt')
                checkpoint = torch.load(ckpt_path, map_location=device)
                gptconf = GPTConfig(**checkpoint['model_args'])
                model = GPT(gptconf)
                state_dict = checkpoint['model']
                unwanted_prefix = '_orig_mod.'
                for k,v in list(state_dict.items()):
                    if k.startswith(unwanted_prefix):
                        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
                model.load_state_dict(state_dict)
            elif init_from.startswith('gpt2'):
                # init from a given GPT-2 model
                model = GPT.from_pretrained(init_from, dict(dropout=0.0))

            model.eval()
            model.to(device)
            if compile:
                model = torch.compile(model) # requires PyTorch 2.0 (optional)

            # look for the meta pickle in case it is available in the dataset folder
            load_meta = True
            if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']: # older checkpoints might not have these...
                meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
                load_meta = os.path.exists(meta_path)
            if load_meta:
                print(f"Loading meta from {meta_path}...")
                with open(meta_path, 'rb') as f:
                    meta = pickle.load(f)
                # TODO want to make this more general to arbitrary encoder/decoder schemes
                stoi, itos = meta['stoi'], meta['itos']
                encode = lambda s: [stoi[c] for c in s]
                decode = lambda l: ''.join([itos[i] for i in l])
            else:
                # ok let's assume gpt-2 encodings by default
                print("No meta.pkl found, assuming GPT-2 encodings...")
                enc = tiktoken.get_encoding("gpt2")
                encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
                decode = lambda l: enc.decode(l)

            # encode the beginning of the prompt
            if start.startswith('FILE:'):
                with open(start[5:], 'r', encoding='utf-8') as f:
                    start = f.read()
            start_ids = encode(start)
            x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

            os.chdir('flask')
            # run generation
            conn = get_db_connection()
            #clean db except for first sample
            conn.execute('DELETE FROM gen_samples WHERE id != 1')
            conn.commit()
            with torch.no_grad():
                with ctx:
                    for k in range(int(num_samples)):
                        y = model.generate(x, int(max_new_tokens), temperature=temperature, top_k=top_k)
                        sample = decode(y[0].tolist())
                        print(sample)
                        print('---------------')
            
                        conn.execute('INSERT INTO gen_samples (content) VALUES (?)',
                                            (sample,))
                        conn.commit()
            samples = conn.execute('SELECT * FROM gen_samples WHERE id!=1').fetchall()
            conn.close()
            #return redirect(url_for('generate'))

    return render_template('generate.html', samples=samples)