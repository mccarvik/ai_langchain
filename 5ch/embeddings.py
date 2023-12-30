
import sys
sys.path.append("..")
from config import set_environment
set_environment()

from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import ArxivLoader
from langchain.text_splitter import CharacterTextSplitter
from scipy.spatial.distance import pdist, squareform
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain.memory import ConversationSummaryMemory, CombinedMemory
from langchain.chains import ConversationChain, OpenAIModerationChain
from langchain.prompts.prompt import PromptTemplate
from langchain.memory import ConversationKGMemory
from langchain.memory import ZepMemory
from langchain.schema import StrOutputParser

import uuid
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dataframe_image as dfi




def sample_embeddings():
    embeddings = OpenAIEmbeddings()
    text = "This is a sample query."
    query_result = embeddings.embed_query(text)
    print(query_result)
    print(len(query_result))


def sample_documents():
    words = ["cat", "dog", "computer", "animal"]
    embeddings = OpenAIEmbeddings()
    doc_vectors = embeddings.embed_documents(words)
    print(doc_vectors)
    X = np.array(doc_vectors)
    dists = squareform(pdist(X))
    df = pd.DataFrame(
        data=dists,
        index=words,
        columns=words
    )
    plt.figure(figsize=(10, 6))
    styled_df = df.style.background_gradient(cmap='coolwarm')
    dfi.export(styled_df, 'embed_heatmaps.png')
    # # Save the styled DataFrame as an image
    # fig, ax = plt.subplots(figsize=(6, 4))  # Set the figure size
    # ax.axis('off')  # Turn off axis for better visualization
    # styled_df.set_table(ax)
    # plt.savefig('embed_heatmaps.png', bbox_inches='tight')
    # plt.close()


def chroma_func():
    loader = ArxivLoader(query="2310.06825")
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000,
    chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    vectorstore = Chroma.from_documents(documents=docs, embedding=OpenAIEmbeddings())
    # similar_vectors = vectorstore.query(query_vector, k)


def conv_chain():
    # Creating a conversation chain with memory
    memory = ConversationBufferMemory()
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo", temperature=0, streaming=True
    )
    chain = ConversationChain(llm=llm, memory=memory)
    # User inputs a message
    user_input = "Hi, how are you?"
    # Processing the user input in the conversation chain
    response = chain.predict(input=user_input)
    # Printing the response
    print(response)
    # User inputs another message
    user_input = "What's the weather like today?"
    # Processing the user input in the conversation chain
    response = chain.predict(input=user_input)
    # Printing the response
    print(response)
    # Printing the conversation history stored in memory
    print(memory.chat_memory.messages)

    conversation = ConversationChain(
        llm=llm,
        verbose=True,
        memory=ConversationBufferMemory()
    )
    
    memory = ConversationBufferWindowMemory(k=1)
    memory.save_context({"input": "hi"}, {"output": "whats up"})
    memory.save_context({"input": "not much you"}, {"output": "not much"})


def upd_prompt_temp():
    llm = OpenAI(temperature=0)
    template = """The following is a friendly conversation between a
    human and an AI. The AI is talkative and provides lots of specific
    details from its context. If the AI does not know the answer to a
    question, it truthfully says it does not know.
    Current conversation:
    {history}
    Human: {input}
    AI Assistant:"""
    PROMPT = PromptTemplate(input_variables=["history", "input"],
    template=template)
    conversation = ConversationChain(
        prompt=PROMPT,
        llm=llm,
        verbose=True,
        memory=ConversationBufferMemory(ai_prefix="AI Assistant"),
    )
    print(conversation)


def summary_chain():
    # Initialize the summary memory and the language model
    memory = ConversationSummaryMemory(llm=OpenAI(temperature=0))
    # Save the context of an interaction
    memory.save_context({"input": "hi"}, {"output": "whats up"})
    # Load the summarized memory
    memory.load_memory_variables({})


def knowledge_graph():
    llm = OpenAI(temperature=0)
    memory = ConversationKGMemory(llm=llm)


def combined_mem():
    # Initialize language model (with desired temperature parameter)
    llm = OpenAI(temperature=0)
    # Define Conversation Buffer Memory (for retaining all past messages)
    conv_memory = ConversationBufferMemory(memory_key="chat_history_lines",
    input_key="input")
    # Define Conversation Summary Memory (for summarizing conversation)
    summary_memory = ConversationSummaryMemory(llm=llm, input_key="input")
    # Combine both memory types
    memory = CombinedMemory(memories=[conv_memory, summary_memory])
    # Define Prompt Template
    _DEFAULT_TEMPLATE = """The following is a friendly conversation between a
                        human and an AI. The AI is talkative and provides lots of specific details
                        from its context. If the AI does not know the answer to a question, it
                        truthfully says it does not know.
                        Summary of conversation:
                        {history}

                        Current conversation:
                        {chat_history_lines}
                        Human: {input}
                        AI:"""
    PROMPT = PromptTemplate(input_variables=["history", "input", "chat_history_lines"], template=_DEFAULT_TEMPLATE)
    # Initialize the Conversation Chain
    conversation = ConversationChain(llm=llm, verbose=True, memory=memory,
    prompt=PROMPT)
    # Start the conversation
    conversation.run("Hi!")


def zep_mem():
    ZEP_API_URL = "http://localhost:8000"
    ZEP_API_KEY = "<your JWT token>"
    session_id = str(uuid.uuid4())
    memory = ZepMemory(
        session_id=session_id,
        url=ZEP_API_URL,
        api_key=ZEP_API_KEY,
        memory_key="chat_history",
    )


def moderation_func():
    moderation_chain = OpenAIModerationChain()
    cot_prompt = PromptTemplate.from_template(
        "{question} \nLet's think step by step!"
    )
    llm_chain = cot_prompt | ChatOpenAI() | StrOutputParser()
    chain = llm_chain | moderation_chain
    response = chain.invoke({"question": "What is the future of programming?"})
    print(response)


# sample_embeddings()
# sample_documents()
# chroma_func()
# conv_chain()
# upd_prompt_temp()
# summary_chain()
# knowledge_graph()
# combined_mem()
# zep_mem()
moderation_func()