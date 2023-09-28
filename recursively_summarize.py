import openai
import os
from dotenv import load_dotenv
from time import time,sleep
import textwrap
import re
import tiktoken


def open_file(filepath):
    with open(filepath, 'r', encoding='ANSI') as infile:
        return infile.read()


# Call OpenAi API key from the settings environment file
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


def save_file(content, filepath):
    with open(filepath, 'w', encoding='utf-8') as outfile:
        outfile.write(content)

max_tokens = 8000
tokens = 4000
model  = 'gpt-4'
def gpt_completion(prompt, model=model, temp=0.6, top_p=1.0, tokens=tokens, freq_pen=0.25, pres_pen=0.0, stop=['<<END>>']):
    max_retry = 5
    retry = 0
    sleep(10)
    while True:
        try:
            response = openai.ChatCompletion.create(
                model=model,
                temperature=temp,
                max_tokens=tokens,
                top_p=top_p,
                frequency_penalty=freq_pen,
                presence_penalty=pres_pen,
                stop=stop,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant for text summarization. IF YOU CANNOT ANSEWAR A QUESTION JUST RETURN AN EMPTY STRING",
                    },
                    {
                        "role": "user",
                        "content": f"{prompt}",
                    },
                ],
            )
            text = response['choices'][0]['message']["content"].strip()
            text = re.sub('\s+', ' ', text)
            filename = '%s_gpt.txt' % time()
            with open('gpt_logs/%s' % filename, 'w') as outfile:
                outfile.write('PROMPT:\n\n' + prompt + '\n\n==========\n\nRESPONSE:\n\n' + text)
            return text
        except Exception as oops:
            retry += 1
            if retry >= max_retry:
                return "GPT error: %s" % oops
            print('Error communicating with OpenAI:', oops)
            


if __name__ == '__main__':
    alltext = open_file('Malz et al_2022_The value of airborne wind energy to the electricity system.txt')
    chunks = textwrap.wrap(alltext,  max_tokens-tokens)
    result = list()
    count = 0
    result_tokens = tokens + 1
    while result_tokens > tokens:
        if count > 0:
            chunks = textwrap.wrap(alltext, (max_tokens-tokens)*0.75)

        for chunk in chunks:
            count = count + 1
            prompt = open_file('prompt.txt').replace('<<SUMMARY>>', chunk)
            prompt = prompt.encode(encoding='ASCII',errors='ignore').decode()
            summary = gpt_completion(prompt)
            print('\n\n\n', count, 'of', len(chunks), ' - ', summary)
            result.append(summary)

        # Count the number of output text tokens
        encoding = tiktoken.get_encoding("cl100k_base")
        result_tokens = len(encoding.encode('\n\n'.join(result)))
    
    save_file('\n\n'.join(result), 'output_%s.txt' % time())