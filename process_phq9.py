import matplotlib.pyplot as plt
import IPython.display as Ipd
import pandas as pd
import torch
from pathlib import Path
import numpy as np
from b2aiprep.process import SpeechToText
from b2aiprep.process import Audio, specgram
from b2aiprep.dataset import VBAIDataset
from b2aiprep.process import extract_opensmile



# Load the b2ai dataset
path = '/home/bridge2ai/Desktop/bridge2ai-data/bids_with_sensitive_recordings/'
dataset = VBAIDataset(path)

# every user has a sessionschema which we can get info for the users from
qs = dataset.load_questionnaires('qgenericpatienthealthquestionnaire9schema')
q_dfs = []
for i, questionnaire in enumerate(qs):
    # get the dataframe for this questionnaire
    df = dataset.questionnaire_to_dataframe(questionnaire)
    df['dataframe_number'] = i
    q_dfs.append(df)
    i += 1

# concatenate all the dataframes
phq9_df = pd.concat(q_dfs)
phq9_df = pd.pivot(phq9_df, index='dataframe_number', columns='linkId', values='valueString')

mapping = {
    "Not at all": 0,
    "Several days": 1,
    "More than half the days": 2,
    "Nearly every day": 3,
}
phq9_df.replace(mapping, inplace=True)

phq9_df = phq9_df[['record_id', 'feeling_bad_self', 'feeling_depressed',
       'move_speak_slow', 'no_appetite', 'no_energy', 'no_interest', 'thoughts_death',
       'trouble_concentrate', 'trouble_sleeping']]

phq9_df['phq9_score'] = phq9_df.drop(columns=['record_id']).sum(axis=1)



# Define custom bin edges
bin_edges = [0, 5, 10, 15, 28]

# Create bins for column 'A'
phq9_df['Severity'] = pd.cut(phq9_df['phq9_score'], bins=bin_edges, labels=['None-minimal', 'Mild', 
                                                         'Moderate', 'Moderately Severe/Severe'], right=False)


print(phq9_df.shape)


# Combine with Rainbow passage
rainbow_df = pd.read_csv('/home/bridge2ai/Desktop/bridge2ai-data/rainbow.csv')

# Combine with phq9 data
phq9_rainbow_df = pd.merge(
    rainbow_df,
    phq9_df[['record_id', 'phq9_score', 'Severity']],
    on='record_id',
    how='inner'
)
# Drop columns not used for machine learning
phq9_rainbow_df.drop(columns=['Unnamed: 0', 'Unnamed: 0.1'], inplace=True)

phq9_rainbow_clean_df = phq9_rainbow_df.drop_duplicates(subset=['record_id'], keep=False)


print(phq9_rainbow_clean_df.shape)
phq9_rainbow_clean_df.head(3)

phq9_rainbow_clean_df.to_csv('phq9_df.csv')

