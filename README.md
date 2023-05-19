# gensum - Generative Summarization for Data Augmentation

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![PyPI version](https://badge.fury.io/py/gensum.svg)](https://badge.fury.io/py/gensum)
![Python 3.10](https://img.shields.io/badge/python-3.6%20%7C%203.7-green.svg)

## Introduction
Imbalanced class distribution is a common problem in ML. Undersampling combined with oversampling are two methods of addressing this issue. 
A technique such as SMOTE can be effective for oversampling, although the problem becomes a bit more difficult with multilabel datasets. 
[MLSMOTE](https://www.sciencedirect.com/science/article/abs/pii/S0950705115002737) has been proposed, but the high dimensional nature of numerical vectors created from text can sometimes make other forms of data augmentation more appealing.

gensum is an NLP library based on [absum](https://github.com/aaronbriel/absum) that uses generative summarization to perform data augmentation in order to oversample under-represented classes in datasets. Recent advancements in generative models such as ChatGPT make this approach optimal in achieving realistic *but unique* data for the augmentation process.

It uses [ChatGPT](https://openai.com/blog/chatgpt) by default, but is designed in a modular way to allow you to use any pre-trained or out-of-the-box Transformers models capable of abstractive summarization. 
absum is format agnostic, expecting only a dataframe containing text and all features. It also uses multiprocessing to achieve optimal performance.

Singular summarization calls are also possible. 

## Algorithm
1. Append counts or the number of rows to add for each feature are first calculated with a ceiling threshold. Namely, if a given feature has 1000 rows and the ceiling is 100, its append count will be 0.

2. For each feature it then completes a loop from an append index range to the append count specified for that given feature. The append index is stored
to allow for multi processing.

3. Generative summarization is completed for a specified size subset of all rows that uniquely have the given feature. 
If multiprocessing is set, the call to generative summarization is stored in a task array later passed to a sub-routine that runs the calls in parallel using the [multiprocessing](https://docs.python.org/2/library/multiprocessing.html) library, vastly reducing runtime.

4. Each summarization is appended to a new dataframe with the respective features one-hot encoded. 

## Installation
### Via pip

```bash
pip install absum
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

gensum expects a DataFrame containing a text column which defaults to 'text', and another feature column which defaults to 'classifier'. All available parameters are detailed in the Parameters section below.

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

Running singular summarization on any chunk of text is simple:
```
text = chunk_of_text_to_summarize
augmentor = Augmentor(min_length=100, max_length=200)
output = augmentor.get_generative_summarization(text)
```

## Parameters

| Name | Type | Description |
| ---- | ---- | ----------- |
| df | (:class:`pandas.Dataframe`, `required`, defaults to None) | Dataframe containing text and one-hot encoded features.
| text_column | (:obj:`string`, `optional`, defaults to "text") | Column in df containing text.
| classifier | (:obj:`string`, `optional`, defaults to "classifier") | Classifier to augment data for.
| min_length | (:obj:`int`, `optional`, defaults to 10) | The min length of the sequence to be generated. Between 0 and infinity. Default to 10.
| max_length | (:obj:`int`, `optional`, defaults to 50) | The max length of the sequence to be generated. Between min_length and infinity. Default to 50.
| num_samples | (:obj:`int`, `optional`, defaults to 100) | Number of samples to pull from dataframe with specific feature to use in generating new sample with Abstractive Summarization.
| threshold | (:obj:`int`, `optional`, defaults to 3500) | Maximum ceiling for each feature, normally the under-sample max.
| multiproc | (:obj:`bool`, `optional`, defaults to True) | If set, stores calls to generative summarization in array which is then passed to run_cpu_tasks_in_parallel to allow for increasing performance through multiprocessing.
| debug | (:obj:`bool`, `optional`, defaults to True) | If set, prints generated summarizations.

## Citation

Please reference [this library](https://github.com/aaronbriel/gensum) if you use this work in a published or open-source project.
