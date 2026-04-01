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

tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased') # Multilingual tokenizer

def prepare_text(full_text, tokenizer):
    
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
        1 if end > 0 and end in eos_pos else 0
        for (start, end) in offsets
    ]

    return {
        "input_ids":      encoding["input_ids"],
        "attention_mask": encoding["attention_mask"],
        "labels":         labels,
    }



if __name__ == "__main__":

    # Playing with function for debugging

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