
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

import openai

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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