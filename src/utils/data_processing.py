import pandas as pd
import re
from collections import defaultdict
import numpy as np
from scipy.stats import entropy
import os
import ast
import json
from IPython.display import display
import readline
from datetime import datetime
import time
from abc import ABC, abstractmethod

class DataProcesser:

    def __init__(self, filepath,  label = False,drop_AI= False):
        self.num_matches = {}
        self.metric_keys = ['timestamp', 'speaker', 'message', 'value', 'Case Match Type', 'matchidx', 'convolen', 'uttidx', 'matchfreq', 'speaker_id']
        self.match_stats = {'relative_pos'}

        '''{{phrase_to_match: [all_utt_mask, matched_convo_stats, matched_utt_stats]}, }'''
        '''Match has: row_id, timestamp, speaker, message, value, utt_idx, speaker_id, case_match_type, match_idx'''
        '''Matched Convo has: match_freq, convo_len'''
        self.text_matches_new = {} 
        self.utterancesDF = None
        self.corpus_utt = None
        self.corpus_convos = None
        self.corpus_speakers =None

        '''add check for type here'''
        if filepath:
            self.getFromCSV(filepath)
        # Ensure df exists even if loading fails
        if label:
            # label_df = pd.read_csv(label_path)
            # self.df = self.df.merge(
            # label_df[["formattedChat", "Outcome"]],
            # on="formattedChat",
            # how="inner"
            # )     
            self.dropChatNA()
            # Drop missing values
            n_missing = self.df['Outcome'].isna().sum()
            print(f"Dropping {n_missing} rows with missing Outcome")    
            self.df = self.df.dropna(subset=["Outcome"])
        if drop_AI:
            #get rid of buyer is AI dialopgues
            self.df = self.df[self.df["buyer_is_AI"] != True]
            #get rid of seller is AI dialopgues
            self.df = self.df[self.df["seller_is_AI"] != True]
        self.df["provided_outcome"] = self.df["Outcome"].apply(self.addDisputeOutcomesfromLabels)

    """Dispute Outcome functions"""   
    def addDisputeOutcomesbySpeaker(self, parsed_dialog):
        flag_speaker = {
            0: ['Buyer', 'I Walk Away.'], 
            1: ['Buyer', 'Accept Deal'],
            2: ['Seller', 'I Walk Away.'],
            3: ['Seller', 'Accept Deal']
        }
        
        if not isinstance(parsed_dialog, list) or len(parsed_dialog) == 0:
            return None
        # Get the last entry from the parsed_dialog list
        last_entry = parsed_dialog[-1]
        # Retrieve speaker and message; default to empty string if not present
        speaker = last_entry.get("speaker", "")
        message = last_entry.get("message", "")
        # Check for an exact match (including case) in flag_speaker
        for key, (expected_speaker, expected_message) in flag_speaker.items():
            if speaker == expected_speaker and message == expected_message:
                return key
        return None
    
    def addDisputeOutcomes(self, parsed_dialog):
        flag_general = {0: ['Accept Deal'],
                        1: ['I Walk Away.']}
        
        if not isinstance(parsed_dialog, list) or len(parsed_dialog) == 0:
            return None
        last_entry = parsed_dialog[-1]
        message = last_entry.get("message", "").strip()
        for key, expected_message in flag_general.items():
            if message == expected_message[0]:
                    return key
        return None

    def filterValidOutcomes(self, remove_AI = True):
        self.filterMatches("Accept Deal")
        final_success_df = self.getMatchedConvoDF("Accept Deal")
        if remove_AI:
            final_success_df = final_success_df[final_success_df["is_AI"] != True]
        display(final_success_df)
        print("Data type of parsed_dialog:", type(final_success_df["parsed_dialog"].iloc[0]))
        final_success_df  = final_success_df[final_success_df["parsed_dialog"].apply(self.is_accept_deal)]
        self.filterMatches("I Walk Away")
        final_reject_df = self.getMatchedConvoDF("I Walk Away")
        final_reject_df  = final_reject_df[final_reject_df["parsed_dialog"].apply(self.is_walk_away)]
        if remove_AI:
            final_reject_df = final_reject_df[final_reject_df["is_AI"] != True]
        self.df = pd.concat([final_success_df, final_reject_df]).sort_index()

    def addDisputeOutcomesfromLabels(self, row):
        return 0 if row == "Resolution" else 1
    
    """Dialog Parsing Utilities"""
    def parsedtoUtteranceDF(self):
        all_rows = []
        for row_idx, parsed_dialog in self.df["parsed_dialog"].items():
            for entry in parsed_dialog:
                if not isinstance(entry, dict):
                     print(f"Warning: Entry in row {row_idx} is not a dictionary: {entry}")
                     continue  # Skip if entry is not a dict
                #print(entry)
                entry["row_idx"] = row_idx
                entry["match_idx"] = False  # Boolean column
                entry["Case Match Type"] = None  # Empty string
                all_rows.append(entry)
        #print(all_rows)
        self.utterancesDF = pd.DataFrame(all_rows)
        self.utterancesDF.loc[13988, 'speaker']= 'Seller'
        self.utterancesDF.loc[13988, 'speaker_id'] = 'Seller_1049'
        self.utterancesDF.loc[13988, 'timestamp']= '1702723625'
        self.utterancesDF["Case Match Type"] = self.utterancesDF["Case Match Type"].astype(object)
        # Compute conversation length per row_idx
        convolen_utt = self.utterancesDF.groupby("row_idx").size().rename("convo_len")
        # Assign conversation length to each utterance in df
        self.utterancesDF = self.utterancesDF.merge(convolen_utt, on="row_idx", how="right")
        if "convo_len" not in self.df.columns:
            self.df= self.df.merge(convolen_utt,left_index=True, right_index=True, how="left")
        self.utterancesDF.drop('speaker', axis=1, inplace=True)
   
    def parseRowKodis(self, row_idx, row_entry, col_name):
        structured_dialog = []
        # Convert to string to avoid errors (NaN -> 'nan' -> we'll treat that like empty)
        if pd.isnull(row_entry):
            row_entry = ""  # or "No chat available"
        
        spk =self.getSpeakerFromCol(col_name, row_idx)
        if isinstance(row_entry, (int,float)):
            structured_dialog.append({
                        'timestamp': None,
                        'speaker': spk,
                        'message': None,
                        #'value': row_entry,
                        'uttidx': None,
                        'speaker_id': spk+ '_'+str(row_idx),
                        'is_AI':    None
                            })
            return
        pattern = re.compile(r'^\s*(\d+|nan)?\s*(Buyer|Seller):\s*(.*)$', re.IGNORECASE)
        lines = str(row_entry).split('\n')
        # Determine if the first line indicates AI involvement
        first_line = lines[0].strip() if lines else ""
        # print("first line is:", first_line , "\n")
        ai_speaker = self.checkAI(row_idx)  # "seller", "buyer", or None
        # print("first speaker AI is:", ai_speaker , "\n")
        line_count =0
        for line in str(row_entry).split('\n'):
            line = line.strip()
            # get rid of empty text
            if not line:
                continue
            match = pattern.match(line)
            if match:
                timestamp_str, speaker, message = match.groups()
                timestamp = int(timestamp_str) if timestamp_str.isdigit() else timestamp_str
                 # Determine if the current speaker is AI based on ai_speaker
                is_AI = (str(ai_speaker).lower() == "seller" and str(speaker).lower() == "seller") or \
                (str(ai_speaker).lower() == "buyer" and str(speaker).lower() == "buyer")
                # print("is AI is:", is_AI , "\n")
                structured_dialog.append({
                    'timestamp': timestamp,
                    'speaker': speaker,
                    'message': message.strip(),
                    #'value': None,
                    'uttidx': line_count,
                    'speaker_id': speaker + '_' + str(row_idx) if speaker is not None else None,
                    'is_AI': is_AI

                })
                line_count +=1
            else:
                # TODO: Handle AI Chats
                if structured_dialog and not line.startswith("Submitted agreement:"):
                    structured_dialog[-1]['message'] += " " + line
                else:
                # some other text response for self-report or survery by speaker 
                    structured_dialog.append({
                        'timestamp': None,
                        'speaker': spk,
                        'message': line,
                        #'value': None,
                        'uttidx': line_count,
                        'speaker_id': spk + '_' + str(row_idx) if spk is not None else None,
                        'is_AI': ai_speaker
                            })
                    line_count +=1
        return structured_dialog

    def addParsedDialogue(self, col_name):
        # Convert all NaN to empty strings right in the DataFrame
        self.dropChatNA()
        print(len(self.df))
        #self.df[col_name] = self.df[col_name].fillna("")
        """
        Adds a new column 'parsed_dialog' to the DataFrame containing structured dialog data.
        """
        # Ensure the 'formattedChat' column exists
        if col_name not in self.df.columns:
            raise ValueError("DataFrame must contain 'col_name' column.")
        # Apply the formatChat function to each row and create a new column
        parsed_rows = []
        for index, row in self.df.iterrows():
            row_value = row[col_name]
            parsed_row = self.parseRowKodis(index, row_value, col_name)
            parsed_rows.append(parsed_row)
        self.df['parsed_dialog'] = parsed_rows
        self.parsedtoUtteranceDF()
        self.df["flag_speaker"] = self.df["parsed_dialog"].apply(self.addDisputeOutcomesbySpeaker)
        self.df["dispute_outcome"] = self.df["parsed_dialog"].apply(self.addDisputeOutcomes)

    def checkAI(self, row_idx):
        df = self.df
        AI_seller_str = "Your sudden demand for a refund is unwarranted. Our product description is crystal clear, and we stand by our policy. Your behavior is disappointing, and your negative review is unfounded."
        AI_Buyer_str = "Your response is utterly unacceptable. I bought the jersey for my nephew, a Kobe Bryant fan, based on your explicit representation. Your deceptive behavior is disgraceful."
        # Check if AI_seller_str is in any row of formattedChat
        # print(df["formattedChat"][row_idx])
        AI_seller_match = AI_seller_str.lower() in df["formattedChat"][row_idx].lower()
        AI_buyer_match = AI_Buyer_str.lower() in df["formattedChat"][row_idx].lower()

        if AI_seller_match:
            return "Seller"
        elif AI_buyer_match:
            return "Buyer"
        else:
            return None
       
    """Phrase Matching Getters"""
    def getMatchedUtterancesDF(self, key_val, all = False):
        # Make sure both DataFrames have the same index for comparison
        match_idx_df = self.text_matches_new[key_val][0]
        if len(self.utterancesDF) != len(match_idx_df):
            raise ValueError("DataFrames must have the same number of rows for comparison.")
        if all:
             matched_utterances = self.utterancesDF.copy()
             matched_utterances["match_idx", "Case Match Type", 'convo_len'] = self.text_matches_new[key_val][0]["match_idx", "Case Match Type" ,"convo_len"]

        else:
            # Filter the rows where 'match_idx' is True in the match_idx column of the second DataFrame
            matched_indices = match_idx_df[match_idx_df['match_idx'] == True].index
            # Now, filter self.utterancesDF using these matched indices
            matched_utterances = self.utterancesDF.loc[matched_indices]
            matched_utterances[["Case Match Type"]] = self.text_matches_new[key_val][2][["Case Match Type"]].values
            matched_utterances["match_idx"] = True
        return matched_utterances
    
    def getMatchedConvoDF(self, key_val, all = False):
        match_idx_df = self.text_matches_new[key_val][1]  # This contains 'row_idx' and 'matchfreq'
        df_main = self.getDataframe().copy()  # Main convo dataframe
        # Find unique 'row_idx' values where 'matchfreq' is nonzero
        if all:
            df_main[["match_freq", "convo_len"]] = match_idx_df[["match_freq", "convo_len"]]
            matched_convos = df_main
        else:
            matched_row_idxs = match_idx_df.loc[match_idx_df['match_freq'] != 0, 'row_idx'].unique()
        # Filter self.getDataframe() where 'row_idx' is in matched_row_idxs
            matched_convos = df_main[df_main.index.isin(matched_row_idxs)]
        #matched_convos[[["convo_len"]] 
        return matched_convos

    """CSV Utilities"""
    def saveToCSV(self, final_filepath, drop_parsed = False):
        os.makedirs(os.path.dirname(final_filepath), exist_ok=True)

        if drop_parsed:
            self.df.drop(columns=['parsed_dialog'], inplace =True)
        self.getDataframe().to_csv(final_filepath, index= True, index_label="Row_Index")
        print(f"Data saved to {final_filepath}")

    def getFromCSV(self, filepath):
        self.df = pd.read_csv(filepath)
        if "Row_Index" in self.df.columns:
                print("Row Index in columns")
                self.df.set_index("Row_Index", inplace=True)  # Set it as the index
        else:
            self.df = pd.read_csv(filepath)
        if "parsed_dialog" not in self.df.columns:
            self.addParsedDialogue("formattedChat")
        else:
            self.df["parsed_dialog"] = self.df["parsed_dialog"].apply(ast.literal_eval)

            self.parsedtoUtteranceDF()


    """Last Utternace Checks""" 
    def is_accept_deal(self, lst):
        if isinstance(lst, list) and len(lst) > 0:
            last = lst[-1]
            if isinstance(last, dict) and 'message' in last:
                return last['message'] == "Accept Deal"
        return False

    def is_walk_away(self, lst):
        if isinstance(lst, list) and len(lst) > 0:
            last = lst[-1]
            if isinstance(last, dict) and 'message' in last:
                return last['message'] == "I Walk Away."
        return False
    
    def is_submit_agreement(self, lst):
        if isinstance(lst, list) and len(lst) > 0:
            last = lst[-1]
            if isinstance(last, dict) and 'message' in last:
                return "Submitted agreement" in last["message"]
        return False
    
    '''Functions for matched key words in text from utterances DF''' 
    def filterMatches(self, value_to_check, subset_to_exclude = None, case_in= None, case_ex= None):
        """
        Checks if the phrase 'string_to_check' appears in any value within the text for 'col_name' dictionary.
        Parameters:
        """
        df = self.utterancesDF
        # Ensure the column is string type, replacing NaN values with empty strings
        df["temp_col"] = df['message'].fillna("").astype(str)

        # Create Boolean masks for each match type
        exact_match = df["temp_col"].str.contains(value_to_check, na=False)
        lower_match = df["temp_col"].str.contains(r'\b' + re.escape(value_to_check.lower()) + r'\b', na=False) & ~exact_match
        case_insensitive_match = df["temp_col"].str.contains(r'\b' + re.escape(value_to_check) + r'\b', case=False, na=False) & ~exact_match & ~lower_match
        df.loc[exact_match, "Case Match Type"] = "Exact"
        df.loc[lower_match, "Case Match Type"] = "Lower"
        df.loc[case_insensitive_match, "Case Match Type"] = "Case Insensitive"
        df.loc[exact_match | lower_match | case_insensitive_match, "match_idx"] = True 
  
        if subset_to_exclude:
            included_rows = self.filterRows("temp_col", value_to_check, subset_to_exclude, case_in, case_ex)
            df["match_idx"] = df.index.isin(included_rows.index)  # This sets match_idx True for rows in df_include, False elsewhere.
            mask = ~df.index.isin(included_rows.index)
            df.loc[mask, "Case Match Type"] = None
    
        # Filter matched utterances (keep same length as self.df
        matched_df_utt_mask = df[["row_idx", "match_idx", "Case Match Type", "convo_len"]]
       
        #changes filtered df to exclude the subset for the matched row statistics 
        if subset_to_exclude:
            utt_stats = included_rows[included_rows["match_idx"] == True][["row_idx", "uttidx", "Case Match Type", "convo_len", "is_AI"]]
        else:
            utt_stats = df[df["match_idx"] == True][["row_idx", "uttidx", "Case Match Type", "convo_len", "is_AI"]]
        # Copute match frequency (count matches per row_idx group)
        match_freq = df.groupby("row_idx")['Case Match Type'].apply(lambda x: x.isin(["Exact", "Lower", "Case Insensitive"]).sum()).reset_index(name="match_freq")

        # Compute conversation length per row_idx
        convolen = df.groupby("row_idx").size().reset_index(name='convo_len')
        # Merge match_freq and convolen on row_idx to create the summary DataFrame
        convo_stats = pd.merge(match_freq, convolen, on="row_idx", how="left")   
        convo_stats.head() 
        self.text_matches_new[value_to_check] = [matched_df_utt_mask, convo_stats, utt_stats]
        self.resetUtDF()
        self.normalizedRelativePos(value_to_check)# Optionally drop it
        df.drop(columns=["temp_col"], inplace=True)

    def groupbyMatchUttStat(self, value_key, group_by, stat_cols, agg_list):
        # df_utt = self.getMatchedUtterancesDF(value_key).copy()
        # df_utt.loc[:,stat_col] = self.text_matches_new[value_key][2][stat_cols] # Assuming this exists
        # stat = df_utt.groupby(group_by)[stat_cols].agg(agg_list)
        # print(f"Key Value: {value_key}, Grouped by: {group_by}, Aggregated column: {stat_cosl}, Aggregations: {agg_list}")
        """
        This function groups by the given columns and applies aggregation functions to the provided statistic columns.
        """
        df_utt = self.getMatchedUtterancesDF(value_key)
        df_utt.loc[:,stat_cols] = self.text_matches_new[value_key][2][stat_cols]
         # Build the aggregation dictionary using the original column names
        agg_dict = {col: agg_list[i] for i, col in enumerate(stat_cols)}
        stat = df_utt.groupby(group_by).agg(agg_dict)
        rename_dict = {col: f"{agg_list[i]}_{col}" for i, col in enumerate(stat_cols)}
        stat.rename(columns=rename_dict, inplace=True)
        print(f"Key Value: {value_key}, Grouped by: {group_by}, Aggregated columns: {stat_cols}, Aggregations: {agg_dict}")
        return stat
    
    def groupbyMatchConvoStat(self, value_key, group_by, stat_cols, agg_list):
        df_convo = self.getMatchedConvoDF(value_key)
        df_convo.loc[:,stat_cols] = self.text_matches_new[value_key][1][stat_cols]  # Assuming this exists
        # Build the aggregation dictionary using the original column names
        agg_dict = {col: agg_list[i] for i, col in enumerate(stat_cols)}
        stat = df_convo.groupby(group_by).agg(agg_dict)
        rename_dict = {col: f"{agg_list[i]}_{col}" for i, col in enumerate(stat_cols)}
        stat.rename(columns=rename_dict, inplace=True)
        print(f"Key Value: {value_key}, Grouped by: {group_by}, Aggregated column: {stat_cols}, Aggregations: {agg_list}")
        return stat
    
    def groupbyConvoMatchesByCase(self, value_key):
        df_utt = self.utterancesDF
        df_utt['Case Match Type'] = self.text_matches_new[value_key][2]['Case Match Type']
        # df_utt.groupby("row_idx")['Case Match Type'].apply(lambda x: x.isin(["Exact", "Lower", "Case Insensitive"]).sum())
        group = df_utt.groupby("row_idx")["Case Match Type"].value_counts().unstack(fill_value=0)
        df_utt['Case Match Type'] = np.nan
        df_utt['Case Match Type'] = df_utt['Case Match Type'].astype('object')
        print(f"\n'{value_key}` Total Number of Case Match Types Across Utterances")
        return group.sum().to_frame(name="Total Count").reset_index()
   
    def filterRows(self, column, include_val=None, exclude_val=None, case_in=True, case_ex=True):
        """
        Filters the DataFrame to include rows where `column` matches any of include_val
        and excludes rows where `column` matches any of exclude_val.
        
        include_val / exclude_val can be:
        - a single int/float
        - a single string
        - a list/tuple of ints/floats
        - a list/tuple of strings
        
        case_in / case_ex control case-sensitivity for string matching.
        """
        if column not in self.utterancesDF.columns:
            filtered_df = self.df.copy()
        else:
            filtered_df = self.utterancesDF.copy()

        def _as_list(x):
            return x if isinstance(x, (list, tuple)) else [x]

        if include_val is not None:
            vals = _as_list(include_val)
    
            if all(isinstance(v, (int, float)) for v in vals):
                filtered_df = filtered_df[filtered_df[column].isin(vals)]
            else:
                # string case: build mask of rows containing any val
                mask = pd.Series(False, index=filtered_df.index)
                for v in vals:
                    mask |= filtered_df[column].astype(str).str.contains(str(v), case=case_in, na=False)
                filtered_df = filtered_df[mask]

        # Exclusion
        if exclude_val is not None:
            vals = _as_list(exclude_val)
            # numeric case
            if all(isinstance(v, (int, float)) for v in vals):
                filtered_df = filtered_df[~filtered_df[column].isin(vals)]
            else:
                mask = pd.Series(False, index=filtered_df.index)
                for v in vals:
                    mask |= filtered_df[column].astype(str).str.contains(str(v), case=case_ex, na=False)
                filtered_df = filtered_df[~mask]

        return filtered_df
  
    def filterParsedDialog(self, no_last_utterance = True, phrase_to_match = None):
        df = self.getUtterancesDF().copy()
        if no_last_utterance:
            
            def strip_last(dlg):
                if isinstance(dlg, list) and len(dlg) > 0:
                    last = dlg[-1]
                    last_text = last.get("message", "") if isinstance(last, dict) else str(last)
                    if "Accept Deal" in text or "I Walk Away" in last_text:
                          return dlg[:-1]
                    return dlg
            df["parsed_dialog"] = df["parsed_dialog"].apply(strip_last)
        
            display(df)
        df['parsed_dialog'] = (
            df['parsed_dialog']
            .apply(lambda dlg: [utt for utt in dlg
                                if phrase_to_match not in utt.get('text','')])
        )

        self.setUtterancesDF()
   
   
    """Dataframe Utilities"""
    def setUtterancesDF(self, utterancesDF):
        self.utterancesDF = utterancesDF

    def getUtterancesDF(self):
        return self.utterancesDF
    
    def getDataframe(self):
        return self.df
    
    def setDataframe(self, df):
        self.df = df
    
    def dropChatNA(self):
        n_missing_chat = self.df['formattedChat'].isna().sum()
        print(f"Dropping {n_missing_chat} rows with missing formattedChat")
        self.df = self.df.dropna(subset=["formattedChat"]).reset_index(drop=True)

    def getSpeakerFromCol(self, col_name, row_idx):
        if col_name.lower().startswith("b"):
            spk = "Buyer"
        elif col_name.lower().startswith("s"):
            spk = "Seller"
        else:
            spk = None
        return spk

    def resetUtDF(self):
        df = self.utterancesDF
        df['Case Match Type'] = np.nan
        df['Case Match Type'] = df['Case Match Type'].astype('object')
        df['match_idx'] = False
        self.utterancesDF = df

    ''' All statistics for words'''
    # def speakerPhrase(self):
    def getTTR(self, key_value, matched):
        df = self.text_matches_new[key_value][1]
        if matched:
            df = df[df[['match_freq'] != 0]]
        convos_with_phrase = df['match_freq'].sum()
      
        return pd.to_numeric(convos_with_phrase/ (df.shape[0]))
        
    def getDispersion(self, key_val, matched):
        #f_i  = frequency of phrase in conversation 
        #f_bar mean frequency across all conversations.
        df = self.text_matches_new[key_val][1]
        if matched:
            df = df[df['match_freq'] != 0]

        f_mean  = df['match_freq'].mean()
        squared_dev = ((df['match_freq'] - f_mean) ** 2).sum()
        print("f_mean is:", f_mean)
        print(" squared dev is:", squared_dev)
        dispersion_index = ((df['match_freq'] - f_mean) ** 2).sum() / f_mean
        return dispersion_index

    def getEntropy(self, key_val, matched):
        df = self.text_matches_new[key_val][1]
   
        if matched:
            df = df[df[['match_freq'] != 0]]
        phrase_counts = df['match_freq']
        phrase_probs = phrase_counts / phrase_counts.sum()
        phrase_entropy = entropy(phrase_probs, base=2)
        return phrase_entropy


    # measures whether a phrase or feature is uniformly distributed
    # across conversations or if it is concentrated in a few conversations
    def getStandardizedDispersion(self, key_val, matched):
        df = self.text_matches_new[key_val][1]     
        if matched:
            df = df[df['match_freq'] != 0]
        f_mean  = df['match_freq'].mean()
        std_dev = df['match_freq'].std()
        num_conversations = len(df['match_freq'])
        # print(num_conversations)
        juilland_d = 1 - (std_dev / (f_mean * np.sqrt(num_conversations)))
        juilland_d = float(juilland_d)
        print(f"The juilland's Dispersion for {key_val} across all conversations is:", juilland_d)
        return juilland_d
    
    def computeAvgInterval(self, keyword, matched = False):
        self.text_matches_new[keyword][1][f"avg_interval_{keyword}"] = self.getMatchedConvoDF(keyword)['formattedChat'].apply(
            lambda row: avg_interval_between(row['formattedChat'], keyword, splitter="\n", case_sensitive=False), axis=1
        )   

        def avg_interval_between(text, keyword, splitter="\n",case_sensitive=False):
            if not isinstance(text, str):
                return np.nan
            if not case_sensitive:
                text    = text.lower()
                keyword = keyword.lower()
            lines = text.split(splitter)

            idxs = [i for i, line in enumerate(lines) if keyword in line]
            # Need at least two to compute an interval
            if len(idxs) < 2:
                return np.nan

            gaps = np.diff(idxs) 
            gaps = gaps - 1
            return float(gaps.mean())
        
    # always matched
    #measures the relative position of a phrase across all utterances
    def normalizedRelativePos(self, key_value):
        #save this column to getMatchedUtterancesDF
        df = self.getMatchedUtterancesDF(key_value)
        df_2 = self.text_matches_new[key_value][2] #stats_df for matched cases
        df_2['relative_pos'] = df_2["uttidx"] / df_2["convo_len"].replace(1, float('nan'))
        df_2['relative_pos'] = pd.to_numeric(df_2["uttidx"] / (df_2["convo_len"] - 1))
        # self.text_matches_new[key_value][2] = df_2
    
    def getUttStat(self, key_value, col_name, func=None):
        df_stats = self.text_matches_new[key_value][2]
        if func:
            result = df_stats[col_name].agg(func)
            func_name = getattr(func, '__name__', str(func))
            print(f"\nThe '{func_name}' '{col_name}' for the phrase `{key_value}` is: {result}")
            return result
        else:
            return df_stats[col_name]
    
    def getConvoStat(self, key_value, col_name, func =None):
        df_stats = self.text_matches_new[key_value][1]
        if func:
            result = df_stats[col_name].agg(func)
            func_name = getattr(func, '__name__', str(func))
            print(f"\nThe '{func_name}' '{col_name}' for the phrase `{key_value}` is: {result}")
            return result
        else:
            return df_stats[col_name].to_frame()
        
    def addUttStat(self,key_value, col_name, col):
        self.text_matches_new[key_value][2][col_name] = col

    def addConvoStat(self,key_value, col_name, col):
        self.text_matches_new[key_value][1][col_name] = col
    
    def getMatchedConvoStatsDF(self, key_value):
        return self.text_matches_new[key_value][1]
    
    def getMatchedUttStatsDF(self, key_value):
        return self.text_matches_new[key_value][2]




class TextProcesser:

    def __init__(self, filepaths):
        self.filepaths = filepaths
        self.records = self._parse_file()
        print(f"Parsed {len(self.records)} unique dialogues from {filepaths}")
        # build dataframes
        self.df_utts = self._create_utterances_df()
        self.df_convos = self._create_conversations_df()

    def _parse_file(self):
        seen = set()
        seen_idx = []
        records = []
        for file in self.filepaths:
            with open(file, 'r', encoding='utf-8') as f:
                text = f.read()

            pattern = re.compile(
                r'<input>\s*(?P<input>[\d\s]+?)\s*</input>\s*'
                r'<dialogue>\s*(?P<dialogue>.*?)\s*</dialogue>\s*'
                r'<output>\s*(?P<output>.*?)\s*</output>\s*'
                r'<partner_input>\s*(?P<partner_input>[\d\s]+?)\s*</partner_input>',
                re.DOTALL
            )
            match_idx =1 
            for match in pattern.finditer(text):
                rec = match.groupdict()
                # normalize whitespace
                rec = {k: v.strip() for k, v in rec.items()}
                # build a core key: utterance texts ignoring speaker prefix
                utterances = []
                for part in rec['dialogue'].split('<eos>'):
                    p = part.strip()
                    if not p:
                        continue
                    # remove 'SPEAKER:' prefix
                    if ':' in p:
                        _, msg = p.split(':', 1)
                        utterances.append(msg.strip())
                    else:
                        utterances.append(p)
                key = tuple(utterances)
               #print(key)
                if key in seen:
                    seen_idx.append(match_idx)
                    match_idx +=1
                    continue
                seen.add(key)
                records.append(rec)
                match_idx +=1
        return records

    def _create_utterances_df(self):
        rows = []
        base_ts = int(time.time())
        for conv_id, rec in enumerate(self.records):
            raw_utts = rec['dialogue'].split('<eos>')
            prev_uid = None
            ts = 0
            uid = 0
            for raw in raw_utts:
                raw = raw.strip()
                if not raw:
                    continue
                if '<selection>' in raw:
                    continue
                if ':' in raw:
                    speaker, text = raw.split(':', 1)
                    speaker = speaker.strip()
                    text = text.strip()
                else:
                    speaker = None
                    text = raw
                timestamp = base_ts + uid
                rows.append({
                    'uttidx': f"deal{uid}",
                    'speaker_id': f"{speaker}_{conv_id}",
                    'row_idx': conv_id,
                    'reply_to': int(prev_uid) if prev_uid is not None else None,
                    'timestamp': timestamp,
                    'text': text
                })
                prev_uid = uid
                uid += 1
                ts += 1
        df_utts = pd.DataFrame(rows)
        print(f"Created utterances df with {len(df_utts)} rows")
        return df_utts

    def _create_conversations_df(self):
        """
        Builds a DataFrame with one row per unique dialogue,
        deduplicating mirror records by dialogue key. Columns:
          - conversation_id: integer ID per unique dialogue
          - input: original input string
          - partner_input: partner's input string
          - selection_you: dict of allocations YOU chose (or flag)
          - selection_them: dict of allocations THEM chose (or flag)
        """
        conv_rows = []
        for conv_id, rec in enumerate(self.records):
            out = rec['output']
            if 'item0' in out:
                parts = out.split()
                sel_you = {p.split('=')[0]: int(p.split('=')[1]) for p in parts[:3]}
                sel_them = {p.split('=')[0]: int(p.split('=')[1]) for p in parts[3:6]}
            else:
                sel_you = {'agreement': False}
                sel_them = {'agreement': False}
            conv_rows.append({
                'input': rec['input'],
                'partner_input': rec['partner_input'],
                'selection_you': sel_you,
                'selection_them': sel_them
            })
        df_convos = pd.DataFrame(conv_rows)
        lengths = self.df_utts.groupby('row_idx').size()
        df_convos['convo_len'] = df_convos.index.map(lengths).fillna(0).astype(int)
        print(f"Created conversations df with {len(df_convos)} rows")
        return df_convos
        

    def getUtterancesDF(self):
        return self.df_utts

    def getDataframe(self):
        return self.df_convos


if __name__ == "__main__":
        filepath = "/Users/mishkin/Desktop/Research/Convo_Kit/ConvoKit_Disputes/data/alldyads.csv"
        data_preprocessor = DataProcesser(filepath)
        #data_preprocessor.addParsedDialogColumn()
        data_preprocessor.show()

