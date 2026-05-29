import shelve
import hashlib

class PersistentCache:

def __init__(self, db_path="cache/cache.db"):
    self.db_path = db_path

def _hash(self, text: str):

    normalized = text.lower().strip()

    return hashlib.md5(
        normalized.encode()
    ).hexdigest()

def get(self, query: str):

    key = self._hash(query)

    with shelve.open(self.db_path) as db:
        return db.get(key)

def set(self, query: str, response: str):

    key = self._hash(query)

    with shelve.open(self.db_path) as db:
        db[key] = response
