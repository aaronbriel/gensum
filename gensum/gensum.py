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
            containing text and one-hot encoded features.
        text_column (:obj:`string`, `optional`, defaults to "text"): Column in 
            df containing text.
        features (:obj:`list`, `optional`, defaults to None): Features to 
            possibly augment data for.
        min_length (:obj:`int`, `optional`, defaults to 10): The min length of 
            the sequence to be generated. Between 0 and infinity.
        max_length (:obj:`int`, `optional`, defaults to 50): The max length of 
            the sequence to be generated. Between min_length and infinity. 
        num_samples (:obj:`int`, `optional`, defaults to 100): Number of 
            samples to pull from dataframe with specific feature to use in 
            generating new sample with Abstract Summarization.
        threshold (:obj:`int`, `optional`, defaults to 3500): Maximum ceiling 
            for each feature, normally the under-sample max.
        multiproc (:obj:`bool`, `optional`, defaults to True): If set, stores 
            calls for Abstract Summarization in array which is then passed to 
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
            features=None,
            min_length=10,
            max_length=50,
            num_samples=100,
            threshold=3500,
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
        self.features = features
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
        self.append_index = 0
        self.df_append = None
        self.generator = generator.Generator(llm=llm, model=model)

    def get_generative_summarization(self, text):
        """
        Computes generative summarization of specified text
        
        :param text: Text to summarize
        :param debug: Whether to print
        :return: generative summarization text
        """
        output = self.generator.generate_summary(text)

        if self.debug:
            logger.info("\nSummarized text: \n", output)

        return output

    def abs_sum_augment(self):
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
        counts = self.get_append_counts(self.df)
        # Create append dataframe with length of all rows to be appended
        self.df_append = pd.DataFrame(
            index=np.arange(sum(counts.values())), columns=self.df.columns)

        # Creating array of tasks for multiprocessing
        tasks = []

        # set all feature values to 0
        for feature in self.features:
            self.df_append[feature] = 0

        for feature in self.features:
            num_to_append = counts[feature]
            for num in range(
                self.append_index, 
                self.append_index + num_to_append):
                if self.multiproc:
                    tasks.append(
                        self.process_generative_summarization(feature, num)
                    )
                else:
                    self.process_generative_summarization(feature, num)

            # Updating index for insertion into shared appended dataframe to 
            # preserve indexing in multiprocessing situation
            self.append_index += num_to_append

        if self.multiproc:
            run_cpu_tasks_in_parallel(tasks)

        return self.df_append

    def process_generative_summarization(self, feature, num):
        """
        Samples a subset of rows from main dataframe where given feature is 
        exclusive. The subset is then concatenated to form a single string and 
        passed to a generative summarizer to generate a new data entry for the 
        append count, augmenting said dataframe with rows to essentially 
        oversample underrepresented data. df_append is set as a class variable 
        to accommodate that said dataframe may need to be shared among multiple 
        processes.
        
        :param feature: Feature to filter on
        :param num: Count of place in abs_sum_augment loop
        """
        # Pulling rows where only specified feature is set to 1
        df_feature = self.df[
            (self.df[feature] == 1) & 
            (self.df[self.features].sum(axis=1) == 1)
        ]
        df_sample = df_feature.sample(self.num_samples, replace=True)
        text_to_summarize = ' '.join(
            df_sample[:self.num_samples][self.text_column])
        new_text = self.get_generative_summarization(text_to_summarize)
        
        # Only add new text samples that aren't empty strings (ie error)
        if len(new_text) > 0:
            self.df_append.at[num, self.text_column] = new_text
            self.df_append.at[num, feature] = 1
        else:
            pass

    def get_feature_counts(self, df):
        """
        Gets dictionary of features and their respective counts
        
        :param df: Dataframe with one hot encoded features to pull 
            categories/features from
        :return: Dictionary containing count of each feature
        """
        shape_array = {}
        for feature in self.features:
            shape_array[feature] = df[feature].sum()
        return shape_array

    def get_append_counts(self, df):
        """
        Gets number of rows that need to be augmented for each feature up to 
        threshold
        
        :param df: Dataframe with one hot encoded features to pull 
            categories/features from
        :return: Dictionary containing number to append for each category
        """
        append_counts = {}
        feature_counts = self.get_feature_counts(df)

        for feature in self.features:
            if feature_counts[feature] >= self.threshold:
                count = 0
            else:
                count = self.threshold - feature_counts[feature]

            append_counts[feature] = count

        return append_counts


def run_cpu_tasks_in_parallel(tasks):
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
    df_augmented = augmentor.abs_sum_augment()
    df_augmented.to_csv(csv.replace(
        '.csv', '-augmented.csv'), encoding='utf-8', index=False)


if __name__ == "__main__":
    main()