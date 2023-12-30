"""Prompt tracking with PromptWatch.io."""
import sys
sys.path.append("..")
sys.path.append("c://users//mccar//miniconda3//lib//site-packages")
from config import set_environment
set_environment()

from langchain import LLMChain, OpenAI, PromptTemplate
from promptwatch import PromptWatch

prompt_template = PromptTemplate.from_template("Finish this sentence {input}")
my_chain = LLMChain(llm=OpenAI(), prompt=prompt_template)

with PromptWatch() as pw:
    my_chain("The quick brown fox jumped over")

if __name__ == "__main__":
    pass
