import os
from typing import List
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

"""
    Define huggingface tokenize function
"""

def rev_wordpiece(str):
    """
        change tokens to general string text
    """

    if len(str) > 1:
        for i in range(len(str)-1, 0, -1):
            if str[i] == '[PAD]':
                str.remove(str[i])
            elif len(str[i]) > 1 and str[i][0] == '#' and str[i][1] == '#':
                str[i-1] += str[i][2:]
                str.remove(str[i])
    return "".join(str[1:-1])


def convert_ids_to_str(ids: List[int], tokenizer: PreTrainedTokenizerBase):
    """
        convert bert token ids to string text with BertTokenizer
    """
    tokens = [tokenizer._convert_id_to_token(idx) for idx in ids]
    output = rev_wordpiece(tokens)

    return output
