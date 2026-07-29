from pyspark import StorageLevel

class DatasetConfig:
    DISPLAY_ROWS: int = 5
    DISPLAY_COLS: int = 10

    SPARK_PANDAS_CONVERSION_LIMIT: int = 100_000
    SPARK_MAX_ROWS_FOR_DOT: int = 1000
    SPARK_INDEX_COL: str = "index"

class MatchingConfig:
    FAISS_PERSIST_POLITIC: StorageLevel = StorageLevel.MEMORY_AND_DISK
    FAISS_SAMPLE_TARGET: int = 5_000_000
    FAISS_DRIVER_INDEX_LIMIT: int = 5_000_000 
    FAISS_CHUNK_SIZE: int = 4096