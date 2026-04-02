from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import os


client = QdrantClient(url="http://localhost:6333", api_key= os.environ.get('QDRANT_API_KEY'))

collection = QdrantClient.create_collection(collection_name='reports')

