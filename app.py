# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from langchain_chroma import Chroma
# from langchain_huggingface import HuggingFaceEmbeddings
# from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
# import torch
# from langchain.prompts import ChatPromptTemplate
# from dotenv import load_dotenv
# import os

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

# class QueryRequest(BaseModel):
#     query_text: str

# app = FastAPI()

# @app.post("/query")
# async def query(request: QueryRequest):
#     query_text = request.query_text

#     try:
#         # Prepare the DB
#         embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#         db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

#         # Search the DB with more results and debug logging
#         results = db.similarity_search_with_relevance_scores(query_text, k=10)  # Increased from 5 to 10
#         print(f"\nSearch Results ({len(results)} found):")
#         for idx, (doc, score) in enumerate(results):
#             print(f"\nResult {idx + 1} (Score: {score:.4f}):")
#             print(f"Content: {doc.page_content}")
#             print("-" * 80)

#         # Filter results with positive scores and sort by relevance
#         valid_results = [r for r in results if r[1] > 0]
#         if not valid_results:
#             raise HTTPException(
#                 status_code=404,
#                 detail={
#                     "message": "No relevant information found",
#                     "scores": [f"{score:.4f}" for _, score in results[:3]]
#                 }
#             )

#         # Use only the most relevant results for context
#         top_results = valid_results[:5]
#         context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in top_results])
        
#         # Prepare context and prompt
#         prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
#         prompt = prompt_template.format(context=context_text, question=query_text)

#         # Generate response
#         model_name = "google/flan-t5-base"
#         tokenizer = AutoTokenizer.from_pretrained(model_name)
#         model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
#         inputs = tokenizer(prompt, return_tensors="pt", max_length=1500, truncation=True)
#         outputs = model.generate(**inputs, max_length=500, num_return_sequences=1, temperature=0.7, top_p=0.9, repetition_penalty=1.2,  num_beams=5 )
#         response_text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

#         # Enhanced response formatting
#         sources = [os.path.basename(doc.metadata.get("source", "")) for doc, _score in top_results]
#         return {
#             "answer": response_text,
#             "sources": sources,
#             "confidence_scores": [f"{score:.4f}" for _, score in top_results],
#             "num_results_found": len(valid_results)
#         }

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)












# works 2


# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from langchain_chroma import Chroma
# from langchain_huggingface import HuggingFaceEmbeddings
# from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
# from langchain.prompts import ChatPromptTemplate
# from dotenv import load_dotenv
# import torch
# import os

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

# class QueryRequest(BaseModel):
#     query_text: str

# app = FastAPI()

# def generate_queries(question: str, model_name: str = "google/flan-t5-base") -> list:
#     """Generate alternative queries for the input question."""
#     prompt = f"""
#     Generate three different ways to ask this question: {question}
#     Provide variations that might help in finding relevant information.
#     Return only the questions, one per line.
#     """
    
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
#     inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
#     outputs = model.generate(**inputs, max_length=150, num_return_sequences=1)
#     variations = tokenizer.decode(outputs[0], skip_special_tokens=True).strip().split('\n')
    
#     # Always include the original question
#     queries = [question] + [v.strip() for v in variations if v.strip()]
#     return queries[:4]  # Limit to original + 3 variations

# @app.post("/query")
# async def query(request: QueryRequest):
#     query_text = request.query_text

#     try:
#         # Prepare the DB
#         embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#         db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

#         # Generate multiple queries
#         queries = generate_queries(query_text)
#         print(f"\nGenerated queries: {queries}")

#         # Search with multiple queries
#         all_results = []
#         for q in queries:
#             results = db.similarity_search_with_relevance_scores(q, k=5)
#             all_results.extend(results)

#         # Deduplicate results based on document content
#         seen_content = set()
#         unique_results = []
#         for doc, score in all_results:
#             if doc.page_content not in seen_content:
#                 seen_content.add(doc.page_content)
#                 unique_results.append((doc, score))

#         # Sort by relevance score
#         valid_results = sorted(unique_results, key=lambda x: x[1], reverse=True)[:10]

#         if not valid_results:
#             raise HTTPException(
#                 status_code=404,
#                 detail={
#                     "message": "No relevant information found",
#                     "queries_tried": queries
#                 }
#             )

#         # Debug logging
#         print(f"\nSearch Results ({len(valid_results)} found):")
#         for idx, (doc, score) in enumerate(valid_results):
#             print(f"\nResult {idx + 1} (Score: {score:.4f}):")
#             print(f"Content: {doc.page_content}")
#             print("-" * 80)

#         # Use top results for context
#         top_results = valid_results[:5]
#         context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in top_results])
        
#         # Generate response
#         prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
#         prompt = prompt_template.format(context=context_text, question=query_text)

#         model_name = "google/flan-t5-base"
#         tokenizer = AutoTokenizer.from_pretrained(model_name)
#         model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
#         inputs = tokenizer(prompt, return_tensors="pt", max_length=1500, truncation=True)
#         outputs = model.generate(
#             **inputs,
#             max_length=500,
#             num_return_sequences=1,
#             temperature=0.7,
#             top_p=0.9,
#             repetition_penalty=1.2,
#             num_beams=5
#         )
#         response_text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

#         # Enhanced response formatting
#         sources = [os.path.basename(doc.metadata.get("source", "")) for doc, _score in top_results]
#         return {
#             "answer": response_text,
#             "sources": sources,
#             "confidence_scores": [f"{score:.4f}" for _, score in top_results],
#             "queries_used": queries,
#             "num_results_found": len(valid_results)
#         }

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)












from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
# from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
import torch
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma  # Changed from langchain_chroma
import os

load_dotenv()

# Use environment variables with defaults
CHROMA_PATH = os.getenv("CHROMA_PATH", "chroma")
MODEL_PATH = os.getenv("MODEL_PATH", "google/flan-t5-base")
PORT = int(os.getenv("PORT", 8000))

# Rest of your code remains the same...

# Load environment variables
load_dotenv()

# Update CHROMA_PATH to use absolute path
CHROMA_PATH = os.getenv("CHROMA_PATH", "/opt/render/project/src/chroma")

PROMPT_TEMPLATE = """
You are a helpful assistant that provides clear and accurate answers based on the given context.
Please answer the question using only the information provided in the context below.
If the context doesn't contain enough information to fully answer the question, say so.

Context:
{context}

Question: {question}

Provide a clear, well-structured answer:
"""

class QueryRequest(BaseModel):
    query_text: str

app = FastAPI()

def generate_queries(question: str, model_name: str = "google/flan-t5-base") -> list:
    """Generate alternative queries for the input question."""
    prompt = f"""
    Generate three different ways to ask this question: {question}
    Provide variations that might help in finding relevant information.
    Return only the questions, one per line.
    """
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(**inputs, max_length=150, num_return_sequences=1)
    variations = tokenizer.decode(outputs[0], skip_special_tokens=True).strip().split('\n')
    
    # Always include the original question
    queries = [question] + [v.strip() for v in variations if v.strip()]
    return queries[:4]  # Limit to original + 3 variations

@app.post("/query")
async def query(request: QueryRequest):
    query_text = request.query_text

    try:
        # Prepare the DB
        embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

        # Generate multiple queries
        queries = generate_queries(query_text)
        print(f"\nGenerated queries: {queries}")

        # Search with multiple queries
        all_results = []
        for q in queries:
            results = db.similarity_search_with_relevance_scores(q, k=5)
            all_results.extend(results)

        # Deduplicate results based on document content
        seen_content = set()
        unique_results = []
        for doc, score in all_results:
            if doc.page_content not in seen_content:
                seen_content.add(doc.page_content)
                unique_results.append((doc, score))

        # Sort by relevance score
        valid_results = sorted(unique_results, key=lambda x: x[1], reverse=True)[:10]

        if not valid_results:
            raise HTTPException(
                status_code=404,
                detail={
                    "message": "No relevant information found",
                    "queries_tried": queries
                }
            )

        # Debug logging
        print(f"\nSearch Results ({len(valid_results)} found):")
        for idx, (doc, score) in enumerate(valid_results):
            print(f"\nResult {idx + 1} (Score: {score:.4f}):")
            print(f"Content: {doc.page_content}")
            print("-" * 80)

        # Use top results for context
        top_results = valid_results[:5]
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in top_results])
        
        # Generate response
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=query_text)

        model_name = "google/flan-t5-base"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        inputs = tokenizer(prompt, return_tensors="pt", max_length=1500, truncation=True)
        outputs = model.generate(
            **inputs,
            max_length=500,
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2,
            num_beams=5
        )
        response_text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        # Enhanced response formatting
        sources = [os.path.basename(doc.metadata.get("source", "")) for doc, _score in top_results]
        return {
            "answer": response_text,
            "sources": sources,
            "confidence_scores": [f"{score:.4f}" for _, score in top_results],
            "queries_used": queries,
            "num_results_found": len(valid_results)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)



