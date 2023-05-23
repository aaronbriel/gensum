# coding=utf-8
# Copyright 2023 Aaron Briel
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import os
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import openai


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Augmentor(object):
    """
    Uses Generative Summarization for Data Augmentation to address multi-label 
    class imbalance.
    
    Parameters:
        df (:class:`pandas.Dataframe`, `optional`, defaults to None): Dataframe 
            containing text and text-based classifier.
        text_column (:obj:`string`, `optional`, defaults to "text"): Column in 
            df containing text.
        classifier (:obj:`list`, `optional`, defaults to None): Classifier to 
            augment data for.
        classifier_values (:obj:`list`, `optional`, defaults to None): Specific
            classifier values to augment data for, otherwise use all.
        min_length (:obj:`int`, `optional`, defaults to None): The min length of 
            the sequence to be generated. Between 0 and infinity.
        max_length (:obj:`int`, `optional`, defaults to None): The max length of 
            the sequence to be generated. Between min_length and infinity. 
        num_samples (:obj:`int`, `optional`, defaults to 20): Number of 
            samples to pull from dataframe with specific feature to use in 
            generating new sample with Generative Summarization.
        threshold (:obj:`int`, `optional`, defaults to mean count for all 
            classifier values): Maximum ceiling for each classifier value, 
            normally the under-sample max.
        prompt (:obj:`string`, `optional`, defaults to "Create SUMMARY_COUNT 
            unique, informally written sentences similar to the ones listed 
            here:") The prompt to use for the generative summarization. If you 
            change the prompt, please be sure to keep the SUMMARY_COUNT string 
            in it somewhere as this is expected and replaced based on the 
            append count calculated for said classifier value.
        llm (:obj:`string`, `optional`, defaults to 'chatgpt'): The 
            generative LLM to use for summarization.
        model (:obj:`string`, `optional`, defaults to 'gpt-3.5-turbo'): The 
            specific model to use.
        temperature (:obj:`int`, `optional`, defaults to 0): Determines the 
            randomness of the generated sequences. Between 0 and 1, where a
            higher value means the generated sequences will be more random.
        debug (:obj:`bool`, `optional`, defaults to True): If set, prints 
            generated summarizations.
    """
    def __init__(
            self,
            df=pd.DataFrame(),
            text_column='text',
            classifier=None,
            classifier_values=None,
            min_length=None,
            max_length=None,
            num_samples=20,
            threshold=None,
            multiproc=True,
            prompt="Create SUMMARY_COUNT unique, informally written sentences \
                similar to the ones listed here:",
            llm = 'chatgpt',
            model = 'gpt-3.5-turbo',
            temperature = 0,
            debug=True
    ):
        self.df = df
        self.text_column = text_column
        self.classifier = classifier
        self.classifier_values = self.get_classifier_values(
            df, classifier_values)
        self.min_length = min_length
        self.max_length = max_length
        self.num_samples = num_samples
        self.threshold = threshold
        self.multiproc = multiproc
        self.prompt = prompt
        self.llm = llm
        self.model = model
        self.temperature = temperature
        self.debug = debug
        # self.append_index = 0
        # self.df_append = None
        self.generator = Generator(llm=llm, model=model)
        
        # If threshold not specified, set to median count for all classifiers
        # to prevent outliers from skewing results
        if self.threshold is None:
            self.threshold = int(self.df['intent'].value_counts().median())
        
        # Set min and max length for summarization if specified.Expects 
        # min_length to be set if max_length specified, however this doesn't 
        # seem to be the most useful feature anyway if we want to summarize 
        # based on sampled text 
        if min_length is not None:
            self.prompt = self.prompt.replace(
                ":", f" with a minimum length of {min_length} words:")
        if max_length is not None:
            self.prompt = self.prompt.replace(
                ":", f" and a maximum length of {max_length} words:")
        
    def get_classifier_values(
            self, 
            df: pd.DataFrame, 
            classifier_values: List[str]) -> List[str]:
        """
        Checks passed in classifier values against those in dataframe to
        ensure that they are valid and returns validated list.
        
        :param df: Dataframe containing text and text-based classifier.
        :param classifier_values: Specific classifier values to augment data 
            for.
        :return: List of verified classifier values.
        """
        filtered_values = []
        unique_classifier_values = df[self.classifier].unique()
        if classifier_values is None:
            filtered_values = unique_classifier_values.tolist()
        else:
            for value in classifier_values:
                if value in unique_classifier_values:
                    filtered_values.append(value)
                else:
                    logger.warning(
                        "Classifier value not found in dataframe: ", value)
                    
        return filtered_values

    def get_generative_summarization(self, texts: List[str]) -> str:
        """
        Computes generative summarization of specified text
        
        :param texts: List of texts to create summarization for
        :param debug: Whether to log output
        :return: generative summarization text
        """
        logger.info("Generating summarization...")
        prompt = self.prompt + "\n" + "\n".join(texts)
        
        # Set min and max word counts for summarization based on sampled 
        # text if not specified in constructor
        if self.min_length is None and self.max_length is None:
            min_length, max_length = get_min_max_word_counts(texts)
            self.prompt = self.prompt.replace(":", f" with a minimum length of \
                {min_length} words and a maximum length of {max_length} words:")
        
        output = self.generator.generate_summary(prompt)
        if self.debug:
            logger.info(f"\nSummarized text: \n{output}")

        return output

    def gen_sum_augment(self) -> pd.DataFrame:
        """
        Gets append counts for each classifier value, generates new samples,
        and appends them to a DataFrame.
        
        :return: Dataframe appended with augmented samples
        """        
        df_augment = pd.DataFrame()
        append_counts = self.get_append_counts()

        for classifier_value in self.classifier_values:
            num_to_append = append_counts[classifier_value]
            if num_to_append > 0:
                logger.info(f"Appending {num_to_append} rows for " + 
                            f"'{classifier_value}' classifier value")
                append_data = self.process_generative_summarization(
                    classifier_value, num_to_append)
                df_append = pd.DataFrame(append_data)
                df_augment = pd.concat(
                    [df_augment, df_append], ignore_index=True)
            else:
                logger.info(f"No rows to append for {classifier_value}")

        if len(df_augment) == 0:
            logger.warning("No rows to append, returning empty DataFrame.")
            
        return df_augment

    def process_generative_summarization(
            self, 
            classifier_value: str,
            num_to_append: int, 
            multiplier: int = 4) -> List[Dict[str, str]]:
        """
        Samples a subset of rows (with replacement if necessary) from main 
        dataframe where classifier is the specified value. The subset is then 
        passed as a list to a generative summarizer to generate a new data 
        entry for the append count, augmenting dict to oversample 
        underrepresented data.
        
        :param classifier_value: Classifier value to filter on
        :param num_to_append: Number of rows to append for given classifier value
        :multiplier: Multiplier used to decide if replacement is necessary
        """
        append_data = []
        
        # Replacing SUMMARY_COUNT with number of texts to summarize
        self.prompt = self.prompt.replace("SUMMARY_COUNT", str(num_to_append))
        df_value = self.df[self.df[self.classifier] == classifier_value]
        # Only use replacement if there is a substantial sample count
        replace = True if len(df_value) < self.num_samples * multiplier else False
        
        df_sample = df_value.sample(self.num_samples, replace=replace)
        text_to_summarize = df_sample[:self.num_samples][self.text_column].tolist()
        new_texts = self.get_generative_summarization(text_to_summarize)
        
        for text in new_texts.split("\n"):
            append_data.append(
                {self.text_column: text, 
                 self.classifier: classifier_value})
            
        return append_data

    def get_value_counts(self) -> Dict[str, int]:
        """
        Gets dictionary of classifier values and their respective counts

        :return: Dictionary containing count of each unique classifier value
        """
        shape_array = {}
        for value in self.classifier_values:
            shape_array[value] = len(
                self.df[self.df[self.classifier] == value])
            
        return shape_array

    def get_append_counts(self) -> Dict[str, int]:
        """
        Gets number of rows that need to be augmented for each classifier value 
        up to threshold
 
        :return: Dictionary containing number to append for each category
        """
        append_counts = {}
        value_counts = self.get_value_counts()
        for value in self.classifier_values:
            if value_counts[value] >= self.threshold:
                count = 0
            else:
                count = self.threshold - value_counts[value]

            append_counts[value] = count

        return append_counts


class Generator(object):
    """
    Calls specified LMM and underlying model to generate text based on prompt.
    
    Parameters:
        llm (:obj:`string`, `optional`, defaults to 'chatgpt'): The 
            generative LLM to use for summarization.
        model (:obj:`string`, `optional`, defaults to 'gpt-3.5-turbo'): The 
            specific model to use.
        temperature (:obj:`int`, `optional`, defaults to 0): Determines the 
            randomness of the generated sequences. Between 0 and 1, where a
            higher value means the generated sequences will be more random.
        debug (:obj:`bool`, `optional`, defaults to True): If set, prints 
            debug to console.
    """
    def __init__(
        self,
        llm = 'chatgpt',
        model = 'gpt-3.5-turbo',
        temperature = 0,
        debug = True
    ):
        self.llm = llm
        self.model = model
        self.temperature = temperature
        self.debug = debug
        
        if llm == 'chatgpt':
            openai.api_key = os.getenv('OPENAI_API_KEY')
    
    def generate_summary(self, prompt: str) -> str:
        """
        Calls specified LLM and underlying model to generate text based on 
        prompt.
        
        :param prompt (:obj:`string`): Prompt to generate text from.
        :return: response (:obj:`string`): Generated text.
        """
        if self.llm == 'chatgpt':
            response = self._chatgpt(prompt)
        else:
            raise ValueError(f"LLM {self.llm} not yet supported.")

        return response
    
    def _chatgpt(self, prompt: str, retry_attempts: int = 3) -> str:
        """
        Calls OpenAI's ChatGPT API to generate text based on prompt.
        
        :param prompt (:obj:`string`): Prompt to generate text from.
        :param retry_attempts (:obj:`int`, `optional`, defaults to 3): Number
            of retry attempts to make if OpenAI fails.
        :return: response (:obj:`string`): Generated text.
        """
        messages = [{"role": "user", "content": prompt}]
        # OpenAI seems to intermittently fail, so we'll retry a few times
        attempts = 0
        wait_time = 1
        while attempts < retry_attempts:
            try:
                response = openai.ChatCompletion.create(
                    model=self.model,
                    temperature=self.temperature,
                    messages=messages
                )["choices"][0]["message"]["content"]
                break
            except (openai.error.RateLimitError, 
                    openai.error.APIConnectionError) as err:
                attempts += 1
                time.sleep(wait_time * attempts)
                response = ''
                if self.debug:
                    logger.warning(f"ERROR: {err}")
                
        
        return response


def get_min_max_word_counts(sents: List[str]) -> Tuple[int, int]:
    """
    Gets min and max word counts from list of sentences
    
    :param sents: List of sentences to get min/max word counts from
    :return: Tuple of min and max word counts
    """
    min_word_count = len(min(sents).split(' '))
    max_word_count = len(max(sents).split(' '))
    
    return min_word_count, max_word_count


def main():
    # Sample usage
    start = time.time()
    csv = 'path_to_csv'
    df = pd.read_csv(csv)
    augmentor = Augmentor(df, text_column='text', classifier='intent')
    df_augmented = augmentor.gen_sum_augment()
    df_augmented.to_csv(csv.replace(
        '.csv', '-augmented.csv'), encoding='utf-8', index=False)
    logger.info(f"Runtime: {time.time() - start} seconds")
    

if __name__ == "__main__":
    main()