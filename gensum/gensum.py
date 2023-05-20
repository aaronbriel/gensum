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
from multiprocessing import Process
from typing import Callable, Dict, List

import numpy as np
import pandas as pd

from . import generator

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
        num_samples (:obj:`int`, `optional`, defaults to 100): Number of 
            samples to pull from dataframe with specific feature to use in 
            generating new sample with Abstract Summarization.
        threshold (:obj:`int`, `optional`, defaults to mean count for all 
            classifier values): Maximum ceiling for each classifier value, 
            normally the under-sample max.
        multiproc (:obj:`bool`, `optional`, defaults to True): If set, stores 
            calls for Generative Summarization in array which is then passed to 
            run_cpu_tasks_in_parallel to allow for increasing performance 
            through multiprocessing.
        prompt (:obj:`string`, `optional`, defaults to "Create 5 unique, 
            informally written sentences similar to the ones listed here:")
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
            num_samples=100,
            threshold=None,
            multiproc=True,
            prompt="Create 5 unique, informally written sentences similar \
                to the ones listed here:",
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
        self.num_samples = num_samples
        self.threshold = threshold
        self.multiproc = multiproc
        self.prompt = prompt
        self.llm = llm
        self.model = model
        self.temperature = temperature
        self.debug = debug
        self.append_index = 0
        self.df_append = None
        self.generator = generator.Generator(llm=llm, model=model)
        
        # Set min and max length for summarization if specified.Expects 
        # min_length to be set if max_length specified, however this doesn't 
        # seem to be the most useful feature anyway if we want to summarize 
        # based on sampled text 
        if min_length is not None:
            self.prompt = self.prompt.replace(
                ":", f" with a min length of {min_length} words:")
        if max_length is not None:
            self.prompt = self.prompt.replace(
                ":", f" and a max length of {max_length} words:")
        
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
        prompt = self.prompt + "\n" + "\n".join(texts)
        output = self.generator.generate_summary(prompt)

        if self.debug:
            logger.info("\nSummarized text: \n", output)

        return output

    def gen_sum_augment(self) -> pd.DataFrame:
        """
        Gets append counts (number of rows to append) for each feature and 
        initializes main classes' dataframe to be appended to that number
        of rows. Initializes all feature values of said array to 0 to 
        accommodate future one-hot encoding of features. Loops over each 
        feature then executes loop to number of rows needed to be appended for
        oversampling to reach needed amount for given feature. If multiproc is 
        set, calls to process_generative_summarization are stored in a tasks 
        array, which is then passed to a function that allows multiprocessing 
        of said summarizations to vastly reduce runtime.
        
        :return: Dataframe appended with augmented samples to make 
            underrepresented features match the count of the majority features.
        """
        append_counts = self.get_append_counts(self.df)
        # Create append dataframe with length of all rows to be appended
        self.df_append = pd.DataFrame(
            index=np.arange(sum(append_counts.values())), 
            columns=self.df.columns)

        # Creating array of tasks for multiprocessing
        tasks = []

        for classifier_value in self.classifier_values:
            num_to_append = append_counts[classifier_value]
            for num in range(
                self.append_index, 
                self.append_index + num_to_append):
                if self.multiproc:
                    tasks.append(
                        self.process_generative_summarization(
                            classifier_value, num)
                    )
                else:
                    self.process_generative_summarization(
                        classifier_value, num)

            # Updating index for insertion into shared appended dataframe to 
            # preserve indexing in multiprocessing situation
            self.append_index += num_to_append

        if self.multiproc:
            run_cpu_tasks_in_parallel(tasks)

        return self.df_append

    def process_generative_summarization(
            self, 
            classifier_value: str, 
            num: int):
        """
        Samples a subset of rows (with replacement) from main dataframe where 
        classifier is the specified value. The subset is then passed as a list 
        to a generative summarizer to generate a new data entry for the append 
        count, augmenting said dataframe with rows to essentially oversample 
        underrepresented data. df_append is set as a class variable to 
        accommodate that said dataframe may need to be shared among multiple 
        processes.
        
        :param classifier_value: Classifier value to filter on
        :param num: Count of place in gen_sum_augment loop
        """
        # Pulling rows for specified feature
        df_value = self.df[self.df[self.classifier] == classifier_value]
        df_sample = df_value.sample(self.num_samples, replace=True)
        text_to_summarize = df_sample[:self.num_samples][self.text_column].tolist()
        new_text = self.get_generative_summarization(text_to_summarize)
        
        # Only add new text samples that aren't empty strings (ie error)
        if len(new_text) > 0:
            self.df_append.at[num, self.text_column] = new_text
            self.df_append.at[num, self.classifier] = classifier_value
        else:
            pass

    def get_value_counts(self, df: pd.DataFrame) -> Dict[str, int]:
        """
        Gets dictionary of classifier values and their respective counts
        
        :param df: Dataframe with classifier column to pull values from
        :return: Dictionary containing count of each unique classifier value
        """
        shape_array = {}
        for value in self.classifier_values:
            shape_array[value] = len(df[df[self.classifier] == value])
            
        return shape_array

    def get_append_counts(self, df: pd.DataFrame) -> Dict[str, int]:
        """
        Gets number of rows that need to be augmented for each classifier value 
        up to threshold
        
        :param df: Dataframe with one hot encoded features to pull 
            categories/features from
        :return: Dictionary containing number to append for each category
        """
        append_counts = {}
        value_counts = self.get_value_counts(df)

        for value in self.classifier_values:
            if value_counts[value] >= self.threshold:
                count = 0
            else:
                count = self.threshold - value_counts[value]

            append_counts[value] = count

        return append_counts


def run_cpu_tasks_in_parallel(tasks: List[Callable]):
    """
    Takes array of tasks, loops over them to start each process, then loops 
    over each to join them
    
    :param tasks: Array of tasks or function calls to start and join
    """
    running_tasks = [Process(target=task) for task in tasks]
    for running_task in running_tasks:
        running_task.start()
    for running_task in running_tasks:
        running_task.join()


def main():
    # Sample usage
    csv = 'path_to_csv'
    df = pd.read_csv(csv)
    augmentor = Augmentor(df, text_column='text')
    df_augmented = augmentor.gen_sum_augment()
    df_augmented.to_csv(csv.replace(
        '.csv', '-augmented.csv'), encoding='utf-8', index=False)


if __name__ == "__main__":
    main()