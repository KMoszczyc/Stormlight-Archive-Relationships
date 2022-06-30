import pandas as pd
import numpy as np
import os
import spacy
from spacy import displacy
import networkx as nx
import re
import matplotlib.pyplot as plt


def ner(file_name):
    """
    Function to process text from a text file (.txt) using Spacy.
    
    Params:
    file_name -- name of a txt file as string
    
    Returns:
    a processed doc file using Spacy English language model
    
    """
    # Load spacy English languague model
    nlp = spacy.load("en_core_web_sm")
    book_text = open(file_name).read()
    book_doc = nlp(book_text)
    
    return book_doc


def get_ne_list_per_sentence(spacy_doc):
    """
    Get a list of entites per sentence of a Spacy document and store in a dataframe.
    
    Params:
    spacy_doc -- a Spacy processed document
    
    Returns:
    a dataframe containing the sentences and corresponding list of recognised named entities       in the sentences
    """
    
    sent_entity_df = []

    # Loop through sentences, store named entity list for each sentence
    for sent in spacy_doc.sents:
        entity_list = [ent.text for ent in sent.ents]
        sent_entity_df.append({"sentence": sent, "entities": entity_list})

    sent_entity_df = pd.DataFrame(sent_entity_df)
    
    return sent_entity_df


def filter_entity(ent_list, character_df):
    """
    Function to filter out non-character entities.
    
    Params:
    ent_list -- list of entities to be filtered
    character_df -- a dataframe contain characters' names, nicknames
    
    Returns:
    a list of entities that are characters (matching by names, nicknames).
    
    """
    mask_df = character_df.isin(ent_list)
    mask_df = mask_df[mask_df.any(axis=1)]
    filtered_df  = character_df[character_df.index.isin(mask_df.index)]

    return list(filtered_df['character_main_name'])


def create_relationships(df, window_size):
    
    """
    Create a dataframe of relationships based on the df dataframe (containing lists of chracters per sentence) and the window size of n sentences.
    
    Params:
    df -- a dataframe containing a column called character_entities with the list of chracters for each sentence of a document.
    window_size -- size of the windows (number of sentences) for creating relationships between two adjacent characters in the text.
    
    Returns:
    a relationship dataframe containing 3 columns: source, target, value.
    
    """
    
    relationships = []

    for i in range(df.index[-1]):
        end_i = min(i+5, df.index[-1])
        char_list = sum((df.loc[i: end_i].character_entities), [])

        # Remove duplicated characters that are next to each other
        char_unique = [char_list[i] for i in range(len(char_list)) 
                       if (i==0) or char_list[i] != char_list[i-1]]

        if len(char_unique) > 1:
            for idx, a in enumerate(char_unique[:-1]):
                b = char_unique[idx + 1]
                relationships.append({"source": a, "target": b})
           
    relationship_df = pd.DataFrame(relationships)
    # Sort the cases with a->b and b->a
    relationship_df = pd.DataFrame(np.sort(relationship_df.values, axis = 1), 
                                   columns = relationship_df.columns)
    relationship_df["value"] = 1
    relationship_df = relationship_df.groupby(["source","target"], 
                                              sort=False, 
                                              as_index=False).sum()
                
    return relationship_df


def remove_empty_lines(src_path, dst_path):
    text = open(src_path).read()
    # text_wo_empty_lines = text.replace('\n\n', '\n')
    text_wo_empty_lines = re.sub(r'^$\n', '', text, flags=re.MULTILINE)

    with open(dst_path, 'w') as text_file:
        text_file.write(text_wo_empty_lines)



