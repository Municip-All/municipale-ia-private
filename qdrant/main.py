from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import os


client = QdrantClient(url="http://localhost:6333", api_key= os.environ.get('QDRANT_API_KEY'))

savage_trash_clt = QdrantClient.create_collection(collection_name='st_reports_RAG')

road_way_clt = QdrantClient.create_collection(collection_name='rw_collection')

savage_animal_clt = QdrantClient.create_collection(collection_name='sa_collection')

urban_item_clt = QdrantClient.create_collection(collection_name='ut_collection')

forgotten_item_clt = QdrantClient.create_collection(collection_name='fi_collection')

incivility_clt = QdrantClient.create_collection(collection_name='iv_collection')

green_space_clt = QdrantClient.create_collection(collection_name='gs_collection')
