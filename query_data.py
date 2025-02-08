


## Works 1


# from langchain_community.vectorstores import Chroma
# from langchain_huggingface import HuggingFaceEmbeddings
# from transformers import pipeline
# from langchain.prompts import ChatPromptTemplate
# import argparse

# CHROMA_PATH = "chroma"

# PROMPT_TEMPLATE = """
# Answer the question based only on the following context:

# {context}

# ---

# Answer the question based on the above context: {question}
# """

# def main():
#     # Create CLI.
#     parser = argparse.ArgumentParser()
#     parser.add_argument("query_text", type=str, help="The query text.")
#     args = parser.parse_args()
#     query_text = args.query_text

#     # Prepare the DB.
#     embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
#     db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

#     # Search the DB.
#     results = db.similarity_search_with_relevance_scores(query_text, k=3)
#     print(f"Search results: {results}")
#     if len(results) == 0 or results[0][1] < 0.5:
#         print(f"Unable to find matching results.")
#         return

#     context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
#     prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
#     prompt = prompt_template.format(context=context_text, question=query_text)
#     print(prompt)

#     # Use transformers pipeline for text generation
#     generator = pipeline('text-generation', model='gpt2')
#     response = generator(prompt, max_new_tokens=100, num_return_sequences=1)
#     response_text = response[0]['generated_text']

#     sources = [doc.metadata.get("source", None) for doc, _score in results]
#     formatted_response = f"Response: {response_text}\nSources: {sources}"
#     print(formatted_response)

# if __name__ == "__main__":
#     main()





## WORKS 2


# from langchain_community.vectorstores import Chroma
# from langchain_huggingface import HuggingFaceEmbeddings
# from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
# from langchain.prompts import ChatPromptTemplate
# import argparse
# import os
# from dotenv import load_dotenv
# import torch

# # Load environment variables
# load_dotenv()

# CHROMA_PATH = "chroma"

# PROMPT_TEMPLATE = """
# You are a helpful assistant that provides clear and accurate answers based on the given context.
# Please answer the question using only the information provided in the context below.
# If the context doesn't contain enough information to fully answer the question, say so.

# Context:
# {context}

# Question: {question}

# Provide a clear, well-structured answer:
# """

# def main():
#     # Create CLI.
#     parser = argparse.ArgumentParser()
#     parser.add_argument("query_text", type=str, help="The query text.")
#     args = parser.parse_args()
#     query_text = args.query_text

#     try:
#         # Prepare the DB.
#         embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
#         db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

#         # Search the DB.
#         results = db.similarity_search_with_relevance_scores(query_text, k=20)
#         if len(results) == 0 or results[0][1] < 0.5:
#             print(f"Unable to find matching results.")
#             return

#         context_text = "\n\n".join([doc.page_content for doc, _score in results])
#         prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
#         prompt = prompt_template.format(context=context_text, question=query_text)

#         # Load model and tokenizer with appropriate memory settings
#         model_name = "google/flan-t5-base"
#         tokenizer = AutoTokenizer.from_pretrained(model_name)
#         model = AutoModelForSeq2SeqLM.from_pretrained(
#             model_name,
#             low_cpu_mem_usage=False,  # Disable low CPU memory usage
#             torch_dtype=torch.float32  # Use standard precision
#         )

#         # Create pipeline with loaded model and tokenizer
#         generator = pipeline(
#             'text2text-generation',
#             model=model,
#             tokenizer=tokenizer,
#             device='cpu'  # Explicitly use CPU
#         )
        
#         # Generate response
#         response = generator(
#             prompt,
#             max_length=300,  # Increase max_length to encourage longer responses
#             num_return_sequences=1,
#             temperature=0.7,
#             do_sample=True
#         )
        
#         response_text = response[0]['generated_text'].strip()

#         sources = [doc.metadata.get("source", None) for doc, _score in results]
#         formatted_response = f"Answer: {response_text}\n\nSources: {sources}"
#         print(formatted_response)

#     except Exception as e:
#         print(f"An error occurred: {e}")
#         print("Please make sure you have all required dependencies installed:")
#         print("pip install -U transformers sentencetransformers torch accelerate")

# if __name__ == "__main__":
#     main()






from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
from langchain.prompts import ChatPromptTemplate
import argparse
import os
from dotenv import load_dotenv
import torch

# Load environment variables
load_dotenv()

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
You are a helpful assistant that provides clear and accurate answers based on the given context.
Please answer the question using only the information provided in the context below.
If the context doesn't contain enough information to fully answer the question, say so.

Context:
{context}

Question: {question}

Provide a clear, well-structured answer:
"""

def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text

    try:
        # Prepare the DB.
        embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

        # Search the DB.
        results = db.similarity_search_with_relevance_scores(query_text, k=5)
        print(f"Search results: {results}")
        if len(results) == 0 or results[0][1] < 0.5:
            print(f"Unable to find matching results.")
            return

        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=query_text)
        print(prompt)

        # Load model and tokenizer with appropriate memory settings
        model_name = "google/flan-t5-base"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            low_cpu_mem_usage=False,  # Disable low CPU memory usage
            torch_dtype=torch.float32  # Use standard precision
        )

        # Create pipeline with loaded model and tokenizer
        generator = pipeline(
            'text2text-generation',
            model=model,
            tokenizer=tokenizer,
            device='cpu'  # Explicitly use CPU
        )
        
        # Generate response
        response = generator(
            prompt,
            max_length=300,  # Increase max_length to encourage longer responses
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True
        )
        
        response_text = response[0]['generated_text'].strip()

        sources = [doc.metadata.get("source", None) for doc, _score in results]
        formatted_response = f"Answer: {response_text}\n\nSources: {sources}"
        print(formatted_response)

    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please make sure you have all required dependencies installed:")
        print("pip install -U transformers sentencetransformers torch accelerate")

if __name__ == "__main__":
    main()