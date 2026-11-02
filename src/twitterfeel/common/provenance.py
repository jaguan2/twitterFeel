"""Stable hashing helpers shared by application and research artifacts."""
import hashlib


def text_digest(texts):
    """Order-sensitive hash, independent of CSV formatting and extra columns."""
    digest = hashlib.sha256()
    for text in texts:
        raw = str(text).encode("utf-8")
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()

def file_digest(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
