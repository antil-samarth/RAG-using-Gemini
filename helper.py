import re
from sentence_transformers import SentenceTransformer

from pymilvus import connections, utility
from pymilvus import Collection, DataType, FieldSchema, CollectionSchema
import time
from os import getenv
from dotenv import load_dotenv, find_dotenv

def insert_embeddings(collection, text_chunks, embeddings):
    if len(text_chunks) != len(embeddings):
        print("Error: The number of chunks and embeddings do not match")
        raise ValueError
    
    print(f"Inserting embeddings into collection: {collection.name}")
    
    if len(embeddings) < 200:
        ins_res = collection.insert([text_chunks, embeddings])
        print(ins_res)
        print("Success!")
    else:
        i = 0
        while i < len(text_chunks):    

            ins_res = collection.insert([text_chunks[i:i+200], embeddings[i:i+200]])

            print(ins_res)

            i += 200
            
            time.sleep(1)
        
    print(f"Insert completed!")

def connect_milvus():
    
    load_dotenv(find_dotenv())
    
    host = getenv("MILVUS_HOST")
    key = getenv("MILVUS_API_KEY")
    
    if not host or not key:
        raise Exception("MILVUS_HOST and MILVUS_API_KEY must be set in .env file")
    
    print(f"Connecting to Milvus...")
    
    connections.connect("default", uri=host, token=key)
    
    print(f"Connected to Milvus!")

def create_collection(collection_name, dimension):
    print(f"Creating collection: {collection_name}")
    
    text_field = FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=8192, description="text", auto_id=False, is_primary=True)
    embedding_field = FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dimension, description="embedding", is_primary=False, auto_id=False)

    schema = CollectionSchema(fields=[text_field, embedding_field], description="collection description")
    
    collection = Collection(name=collection_name, schema=schema)

    print(f"Collection {collection_name} created")
    print(f"Schemas: {schema}")
    
    return collection

def semantic_search(collection, query_embedding, top_k=5, metric_type="L2"):
    print("Searching for similar embeddings...")

    search_params = {
        "metric_type": metric_type,
        "params": {"nprobe": 16}
    }
    
    results = collection.search(query_embedding, "embedding", limit=top_k, param=search_params)
    print("Search completed!")
    return results



def generate_embeddings(text_chunks, model_name="all-MiniLM-L12-v2"):
    model = SentenceTransformer(model_name)

    return model.encode(text_chunks)


def text_splitter(file_name, chunk_size=500, step=50):
    
    with open(file_name, "r", encoding="utf-8") as f:
        data = f.read()
    
    data.replace(" ", " ").replace("\n", " ").replace("\t", " ")
    
    data = re.sub(r'\s+', ' ', data)

    chunks = []

    words = data.split(" ")
    
    del data
    
    chunks = []
    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
        
    print(f"Total chunks: {len(chunks)}")
        
    return chunks