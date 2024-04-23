import re
from sentence_transformers import SentenceTransformer

from pymilvus import connections, utility
from pymilvus import Collection, DataType, FieldSchema, CollectionSchema
import time
from os import getenv
from dotenv import load_dotenv, find_dotenv

def connect_milvus():
    
    load_dotenv(find_dotenv())
    
    host = getenv("MILVUS_HOST")
    key = getenv("MILVUS_API_KEY")
    
    connections.connect("default", uri=host, token=key)
    
    return connections

def create_collection(collection_name, dimension=768):
        
        collection = Collection(collection_name)
        
        if collection.exists():
            print(f"Collection {collection_name} already exists")
            return collection
        
        text_field = FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=8192, description="text", auto_id=False, is_primary=True)
        embedding_field = FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384, description="embedding", is_primary=False, auto_id=False)

        
        field = FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dimension)
        schema = CollectionSchema(fields=[field], description="Embedding collection")
        
        collection.create(schema)
        
        print(f"Collection {collection_name} created")
        
        return collection



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