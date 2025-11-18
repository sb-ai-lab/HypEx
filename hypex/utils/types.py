from pyspark.sql.types import StringType, LongType, DoubleType

class SparkTypeMapper:
    @staticmethod
    def types(value):
        if value is None:
            return StringType()
        elif isinstance(value, str):
            return StringType()
        elif isinstance(value, int):
            return LongType()
        elif isinstance(value, float):
            return DoubleType()
        else:
            return StringType()