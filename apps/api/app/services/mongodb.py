from pymongo import MongoClient

from app.core.config import settings


def get_database():
    client = MongoClient(settings.mongo_url)
    return client[settings.mongo_db]
