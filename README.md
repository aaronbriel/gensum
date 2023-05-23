# gensum - Generative Summarization for Data Augmentation

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![PyPI version](https://badge.fury.io/py/gensum.svg)](https://badge.fury.io/py/gensum)
![Python 3.10](https://img.shields.io/badge/python-3.6%20%7C%203.7-green.svg)

## Introduction
Imbalanced class distribution remains a classic common problem in ML. Undersampling combined with oversampling are two methods of attempting to address this issue. 
Techniques such as SMOTE and MLSMOTE have been proposed, but the high dimensional nature of numerical vectors created from text makes other data augmentation approaches preferable.

gensum is an NLP library based on [absum](https://github.com/aaronbriel/absum) that uses generative summarization to perform data augmentation by oversampling under-represented classes in text classification datasets. Recent advancements in generative models such as ChatGPT make this approach optimal in achieving realistic *but unique* data for the augmentation process.

It uses [ChatGPT](https://openai.com/blog/chatgpt) by default, but is designed in a modular way to allow you to use any large language models capable of generative summarization. `gensum` is format agnostic, expecting only a DataFrame containing a text and classifier column. 

## Installation
### Via pip

```bash
pip install gensum
```

### From source

```bash
git clone https://github.com/aaronbriel/gensum.git
pip install [--editable] .
```

or

```bash
pip install git+https://github.com/aaronbriel/gensum.git
```

## Usage

gensum expects a DataFrame containing a text column which defaults to 'text', and another classifier column which defaults to 'classifier'. All available parameters are detailed in the Parameters section below. Be sure to set the `OPENAI_API_KEY` environmental parameter prior to running the code.

```bash
import pandas as pd
from gensum import Augmentor

csv = 'path_to_csv'
df = pd.read_csv(csv)
augmentor = Augmentor(df, text_column='text', classifier='intent')
df_augmented = augmentor.gen_sum_augment()
# Store resulting dataframe as a csv
df_augmented.to_csv(csv.replace('.csv', '-augmented.csv'), encoding='utf-8', index=False)
```

NOTE: The output dataframe contains only the augmented rows.

## Parameters

| Name | Type | Description |
| ---- | ---- | ----------- |
| df | (:class:`pandas.Dataframe`, `required`, defaults to None) | Dataframe containing text and one-hot encoded features.
| text_column | (:obj:`string`, `optional`, defaults to "text") | Column in df containing text.
| classifier | (:obj:`string`, `optional`, defaults to "classifier") | Classifier to augment data for.
| classifier_values | (:obj:`string`, `optional`, defaults to None) | Specific classifier values to augment data for.
| min_length | (:obj:`int`, `optional`, defaults to 10) | The min length of the sequence to be generated. Between 0 and infinity. Default to 10.
| max_length | (:obj:`int`, `optional`, defaults to 50) | The max length of the sequence to be generated. Between min_length and infinity. Default to 50.
| num_samples | (:obj:`int`, `optional`, defaults to 20) | Number of samples to pull from dataframe with specific feature to use in generating new sample with Generative Summarization.
| threshold | (:obj:`int`, `optional`, defaults to mean count for all classifier values) | Maximum ceiling for each feature, normally the under-sample max.
| prompt  | (:obj:`string`, `optional`, defaults to "Create SUMMARY_COUNT unique, informally written sentences similar to the ones listed here:") | The prompt to use for the generative summarization. If you change the prompt, please be sure to keep the SUMMARY_COUNT string in it somewhere as this is expected and replaced based on the append count calculated for said classifier value.
| llm | (:obj:`string`, `optional`, defaults to 'chatgpt') | The generative LLM to use for summarization.
| model | (:obj:`string`, `optional`, defaults to 'gpt-3.5-turbo') | The specific model to use.
| temperature | (:obj:`int`, `optional`, defaults to 0) | Determines the randomness of the generated sequences. Between 0 and 1, where a higher value means the generated sequences will be more random.
| debug | (:obj:`bool`, `optional`, defaults to True) | If set, prints generated summarizations.

## Citation

Please reference [this library](https://github.com/aaronbriel/gensum) if you use this work in a published or open-source project.
