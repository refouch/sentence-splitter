"""
Taking as inupt: full text with <EOS> tokens
We want to obtain:
    - Tokenized sequence of the same text (X vector for our encoder)
    - A label vector Y pointing out which tokens ends a Sentence
    
Tasks to carry out:
    - Divide te text into words 
    - Delete the <EOS> and mark its position in the vector
    - Tokenise the clean text (without EOS)
    - Find a way to map the obtained token sequence to our original position vector"""

import re
from transformers import AutoTokenizer
from typing import Dict, List
from torch.utils.data import Dataset
import torch

from pathlib import Path

def load_raw_data(split: str, data_dir="datasets") -> List[str]:
    """Function to recursively load all the text in the .sent_split files for a specific split
    
        Args:
            - split: specify which split is to be loaded
                -> Choose from 'train', 'dev', or 'test'
    """

    root_dir = Path(__file__).resolve().parent
    
    data_path = root_dir / data_dir
    
    raw_texts = []

    # Search recursively in all folders the files corresponding to the correct data split
    for file in data_path.rglob(f"*{split}*.sent_split"):
        print(f"Reading : {file.name}")
        
        with open(file, 'r', encoding='utf-8') as f:
            content = f.read()
            if content.strip():
                raw_texts.append(content)
                
    return raw_texts


def prepare_text(full_text: str, tokenizer: AutoTokenizer) -> Dict[str, List]:
    """Function to transform raw text from the datasets into tokenized vector + label vector
        
        Input:
            - full_text: raw string containing <EOS> tokens
            - tokenizer: the tokenizer object used for splitting the sentences
        
        Output: A dictionnary containing:
            - input_ids:  List of tokens ids representing the full text
            - attention_mask: Useful if we need padding
            - labels: List of prediction goals: 1 if the token ends a sentence, else 0
            - offset_mapping: List of character offset created by tokenizer. Useful later to compare with spacy"""
    
    # 1. find the position of all <EOS> and delete them
    eos_pos = set()
    clean_text = ""

    sentences = re.split(r'<EOS>', full_text)

    for sentence in sentences:
        clean_text += sentence
        eos_pos.add(len(clean_text))


    # 2. Tokenize the full text
    encoding = tokenizer(
        clean_text,
        return_offsets_mapping=True,
        truncation=True,
        max_length=512,
    )
    

    # 3. Use the offsets computed by the tokenizer to map the real sentece endings
    offsets = encoding["offset_mapping"]
    labels = [
        1 if end > 0 and end in eos_pos else 0 # 1 if the ending character of a token is in the previously marked EOS positions
        for (start, end) in offsets
    ]

    return {
        "input_ids": encoding["input_ids"],
        "attention_mask": encoding["attention_mask"],
        "labels": labels,
        "offset_mapping": encoding['offset_mapping']
    }


class EOSDataset(Dataset):
    """Child class of regular Pytorch dataset to feed our optimization loop.
        Mainly useful to split our big text into chunks of tokens manageable for BERT"""

    def __init__(self, raw_texts: List[str], tokenizer: AutoTokenizer, max_length=512, stride=256):
        self.samples = []

        for text in raw_texts:
            encoded = prepare_text(text, tokenizer)

            input_ids = encoded["input_ids"]
            attention_mask = encoded["attention_mask"]
            labels = encoded["labels"]

            # Need to split up the whole sequence as BERT only proceses a maximum of 512 tokens
            # We add padding + stride -> if the cut happens in the middle of a sentence this creates overlap so that each batch has the full context
            for start in range(0, len(input_ids), stride):
                end = start + max_length
                
                window_ids   = input_ids[start:end]
                window_mask  = attention_mask[start:end]
                window_labels = labels[start:end]

                window_labels[0]  = -100 # Ignoring both [CLS] and [SEP] tokens
                window_labels[-1] = -100

                # We add some padding if the previous window is too short
                pad_len = max_length - len(window_ids)
                if pad_len > 0:
                    window_ids    = window_ids    + [tokenizer.pad_token_id] * pad_len
                    window_mask   = window_mask   + [0] * pad_len
                    window_labels = window_labels + [-100] * pad_len
                    # -100 are ignored by the loss

                self.samples.append({
                    "input_ids":      torch.tensor(window_ids),
                    "attention_mask": torch.tensor(window_mask),
                    "labels":         torch.tensor(window_labels),
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

if __name__ == "__main__":

    # Playing with function for debugging

    tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased') # Multilingual tokenizer

    sample = """From the AP comes this story :<EOS> 

    President Bush on Tuesday nominated two individuals to replace retiring jurists
    on federal courts in the Washington area.<EOS> Bush nominated Jennifer M. Anderson
    for a 15-year term as associate judge of the Superior Court of the District of
    Columbia, replacing Steffen W. Graae.<EOS> ***<EOS> Bush also nominated A. Noel Anketell
    Kramer for a 15-year term as associate judge of the District of Columbia Court
    of Appeals, replacing John Montague Steadman.<EOS> 

    The sheikh in wheel-chair has been attacked with a F-16-launched bomb.<EOS> He could
    be killed years ago and the israelians have all the reasons, since he founded
    and he is the spiritual leader of Hamas, but they didn't.<EOS> Today's incident
    proves that Sharon has lost his patience and his hope in peace.<EOS> 
    """

    result = prepare_text(sample, tokenizer)

    tokens = tokenizer.convert_ids_to_tokens(result["input_ids"])
    print(tokens)
    print(result['labels'])

    dataset = EOSDataset([sample],tokenizer)
    print(dataset.__getitem__(0))