# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain.evaluation import load_evaluator
# from dotenv import load_dotenv
# import os

# # Load environment variables. Assumes that project contains .env file with API keys
# load_dotenv()

# def main():
#     # Get embedding for a word.
#     embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
#     vector = embedding_function.embed_query("apple")
#     print(f"Vector for 'apple': {vector}")
#     print(f"Vector length: {len(vector)}")

#     # Compare vector of two words
#     evaluator = load_evaluator("pairwise_embedding_distance")
#     words = ("apple", "iphone")
#     x = evaluator.evaluate_string_pairs(prediction=words[0], prediction_b=words[1])
#     print(f"Comparing ({words[0]}, {words[1]}): {x}")

# if __name__ == "__main__":
#     main()





from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load environment variables. Assumes that project contains .env file with API keys
load_dotenv()

def cosine_similarity_score(vector1, vector2):
    vector1 = np.array(vector1).reshape(1, -1)
    vector2 = np.array(vector2).reshape(1, -1)
    return cosine_similarity(vector1, vector2)[0][0]

def main():
    # Get embedding for a word.
    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_apple = embedding_function.embed_query("apple")
    #print(f"Vector for 'apple': {vector_apple}")
    print(f"Vector length: {len(vector_apple)}")

    vector_iphone = embedding_function.embed_query("iphone")
    #print(f"Vector for 'iphone': {vector_iphone}")
    print(f"Vector length: {len(vector_iphone)}")

    # Compare vector of two words using cosine similarity
    similarity_score = cosine_similarity_score(vector_apple, vector_iphone)
    print(f"Cosine similarity between 'apple' and 'iphone': {similarity_score}")

if __name__ == "__main__":
    main()