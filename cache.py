# cache.py
import os, shelve, hashlib

class SimpleCache:
    def __init__(self, filename="cache/db"):
        # Ensure the cache directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        self.filename = filename

    def _key(self, k):
        return hashlib.sha256(k.encode()).hexdigest()

    def get(self, key):
        with shelve.open(self.filename) as db:
            value = db.get(self._key(key))
            if value:
                print("🟢 Cache HIT")
            else:
                print("🔵 Cache MISS")
            return value


    def set(self, key, value):
        with shelve.open(self.filename) as db:
            db[self._key(key)] = value
