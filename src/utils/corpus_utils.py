import re
import pandas as pd
from convokit import Corpus, Speaker, Utterance, Conversation
from IPython.display import display
from model.config import *




# intention: make dataframe object from a pre-processed dataframe with correct primary info and metadata info needed 
#ID, timestamp, text, speaker (a string ID), reply_to (a string ID), conversation_id (a string ID).

def setUtteranceHeaders(headers):
   global utterance_headers
   utterance_headers = headers

def setSpeakerHeaders(headers):
    global speaker_metadata
    speaker_metadata = headers

def setConversationHeaders(headers):
    global conversation_headers
    conversation_headers = headers

def setConversationMetadata(metadata):
    global conversation_metadata
    conversation_metadata = metadata   

def setSpeakerMetadata(metadata):
    global speaker_metadata 
    speaker_metadata = metadata

def addMetadataCols(data):
    df_convo =  data.getDataframe()
    df_utt = data.getUtterancesDF()
     # conversation side
    if conversation_metadata:
        for col in conversation_metadata:
            if col not in df_convo.columns:
                df_convo[col] = None
    print(utterance_metadata)
    # utterance side
    if utterance_metadata:
        for col in utterance_metadata:
            if col not in df_utt.columns:
                df_utt[col] = None
    data.setUtterancesDF(df_utt)
    data.setDataframe(df_convo)

def convertHeaders(df, corpus_type):
    if corpus_type.lower() == 'utterance':
        rename_map = {
        'speaker_id': 'speaker',
        'timestamp': 'timestamp',
        'message': 'text'
        }
        df = df.rename(columns=rename_map)
        if utterance_metadata is not None:
            all_cols = utterance_headers + utterance_metadata
            df = df[all_cols]
            df = prepend_meta(df, utterance_metadata)
        else:
            df = df[utterance_headers]
        df = df.where(pd.notnull(df), None)
        return df
    
    if corpus_type.lower() == 'conversation':
        if conversation_metadata is not None:
            all_cols =conversation_headers +conversation_metadata
            df = df[all_cols]
            df =prepend_meta(df, conversation_metadata)
        else:
            df = df[conversation_headers]
        df = df.where(pd.notnull(df), None)
        return df

    if corpus_type.lower() == 'speaker':
        df['id'] = df['speaker_id']
        if speaker_metadata is not None:
            all_cols =speaker_headers +speaker_metadata
            df = df[all_cols]
            df = prepend_meta(df,speaker_metadata)
        else:
            df = df[speaker_headers]
        df = df.where(pd.notnull(df), None)
        return df

def setReplyTo(df):
    shifted = df.groupby('row_idx')['id'].shift().astype(object)
    # mask the NaNs with None
    return shifted.mask(shifted.isna(), None)

def prepend_meta(df, meta_list):
    for col in meta_list:
        if col in df.columns:
            df = df.rename(columns={col: 'meta.' + col})
    return df
    
def buildUtteranceDF(df):
    df['row_idx'] = df['row_idx'].apply(lambda x: int(float(x)))
    df['uttidx'] = df['row_idx'].apply(lambda x: int(float(x)))
    df = df.copy()
    #df.drop('speaker', axis=1, inplace=True)
    df['row_idx'] = df['row_idx'].apply(lambda x: int(float(x)))
    df[['row_idx', 'uttidx']]= df[['row_idx', 'uttidx']].astype(str)
    df['id'] = df.apply(lambda row: f"utt{row['uttidx']}_con{row['row_idx']}", axis=1)
    df['reply_to'] = setReplyTo(df)
    # df.drop('uttidx', axis=1, inplace=True)
    df['conversation_id'] = df.groupby('row_idx')['id'].transform('first')
    df['row_idx'] = df['row_idx'].apply(lambda x: int(float(x)))
    convo_mapping = (df.drop_duplicates('row_idx')[['row_idx','conversation_id']].set_index('row_idx')['conversation_id'])
    df = convertHeaders(df,'utterance')
    return df,  convo_mapping

def buildSpeakerDF(df):
    df = df.copy()
    df =convertHeaders(df,'speaker')
    return df

def buildConvoDF(df, mapping):
    df = df.copy()
    df['id'] = df.index.map(mapping)
    df =convertHeaders(df,'conversation')
    df.set_index('id', inplace=True, drop=False)
    return df

def is_valid_timestamp(val):
    try:
        return val is not None and str(val).lower() != 'nan' and pd.notnull(val)
    except:
        return False

def clean_corpus_timestamps(utts, speakers, convos):
    utts['timestamp'] = pd.to_numeric(utts['timestamp'], errors='coerce')
    # 2) impute
    missing = utts['timestamp'].isna()
    for idx in utts[missing].index:
        conv_id = utts.at[idx, 'conversation_id']
        parent_id = utts.at[idx, 'reply_to']
        new_ts = None

        # try to grab parent timestamp
        if pd.notnull(parent_id):
            parent_series = utts.loc[utts['id'] == parent_id, 'timestamp']
            if not parent_series.empty and pd.notnull(parent_series.iloc[0]):
                new_ts = parent_series.iloc[0] - 1

        # fallback
        if new_ts is None:
            # get the integer position of this row in the DF
            loc = utts.index.get_loc(idx)
            for next_pos in range(loc+1, len(utts)):
                row = utts.iloc[next_pos]
                if (row['conversation_id'] == conv_id
                    and not pd.isna(row['timestamp'])):
                    new_ts = row['timestamp'] - 1
                    break

        utts.at[idx, 'timestamp'] = new_ts

    # 3) cast to integer POSIX seconds
    utts['timestamp'] = utts['timestamp'].astype(int)
    zero_ts = utts[utts['timestamp'] == 0]
    print(f"*** {len(zero_ts)} utterances with timestamp == 0 after imputation ***")
    # 4) prune speakers & convos
    used_speakers = set(utts['speaker'])
    used_convos   = set(utts['conversation_id'])
    speakers = speakers[speakers['id'].isin(used_speakers)].reset_index(drop=True)
    convos   = convos[convos['id'].isin(used_convos)].reset_index(drop=True)

    return utts, speakers, convos

def corpusBuilder(data):
    addMetadataCols(data)
    utts, convo_mapping = buildUtteranceDF(data.getUtterancesDF())
    speakers = buildSpeakerDF(data.getUtterancesDF())
    convos = buildConvoDF(data.getDataframe(), convo_mapping)
    utts, speakers, convos = clean_corpus_timestamps(utts, speakers, convos)
    display(utts)
    display(speakers)
    display(convos)
    print(f"conversation metadata: {utterance_metadata}")
    print(f"utterance metadata: {conversation_metadata}")
    convos = convos.set_index('id', drop=False)  # ensure 'id' is the index
    corpus_ob = Corpus.from_pandas(utterances_df=utts, speakers_df=speakers, conversations_df=convos)
    missing_in_convos = set(utts['conversation_id']) - set(convos['id'])
    print(f"Utterance conversation_ids missing from convos.id: -> {len(missing_in_convos)}")

    # 2) Are there any convo IDs that no utterance refers to?
    unused_convos = set(convos['id']) - set(utts['conversation_id'])
    print(f"Conversation ids in convos not used by any utterance: -> {len(unused_convos)}")
    # 3) Quick boolean checks
    all_match = utts['conversation_id'].isin(convos['id']).all()
    print(f"Every utterance.conversation_id exists in convos.id?: -> {all_match}")

    all_used = convos['id'].isin(utts['conversation_id']).all()
    print(f"Every convos.id is referred to by at least one utterance?: -> {all_used}")
    print(f"Unique conversation_id’s in utterance DF: -> {utts['conversation_id'].nunique()}")
    print(utts['conversation_id'].unique()[:2])  # sample of the first few
    return corpus_ob

'''return: Speaker DataFrame'''
def buildspeakerParams(self, df):
    speakers_dict = {}
    # We'll store all utterances and conversations by speaker base id
    grouped = df.groupby("speaker_id")
    for full_speaker_id, group in grouped:
        utts = {}
        convos = {}
        for _, row in group.iterrows():
            utt_id = f"{row['uttidx']}"  # unique across dataset
            conv_id = f"{row['row_idx']}"
            utterance = Utterance(
                id=utt_id,
                speaker=full_speaker_id,
                conversation_id=conv_id,
                text=row["message"]
            )
            utts[utt_id] = utterance
            # Only create conversation if not already added
            if conv_id not in convos:
                convos[conv_id] = Conversation(id=conv_id, utterances=[utterance])
            else:
                convos[conv_id].add_utterance(utterance)

        if full_speaker_id not in speakers_dict:
            speakers_dict[full_speaker_id] = {
                "id": full_speaker_id,
                "utts": utts,
                "convos": convos
            }
        else:
            speakers_dict[full_speaker_id]["utts"].update(utts)
            speakers_dict[full_speaker_id]["convos"].update(convos)

    return list(speakers_dict.values())

    speaker_list=[]
if __name__ == "__main__":
    print("This file")