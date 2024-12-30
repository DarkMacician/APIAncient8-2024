from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi


uri = "mongodb+srv://prompt:123@cluster0.admu7.mongodb.net"

client = MongoClient(uri, server_api=ServerApi('1'))

db = client['NeoFeed']
gen = db['General']
ex = db['Extra']