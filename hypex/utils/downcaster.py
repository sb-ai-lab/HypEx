from pyspark.ml.feature import StringIndexer
import pyspark.sql.functions as F
import pyspark.sql  as spark

import pandas as pd

from typing import List

class Downcast:
    
    def __init__(self, data: spark.DataFrame):
        self.data = data
        self.labels = None

    def execute(self) -> pd.DataFrame:
        columns_dict = dict(self.data.dtypes)

        double_columns = [col for col, c_type in columns_dict.items() 
                               if c_type == 'double' or c_type.startswith('decimal')]
        categorical_columns = [col for col, c_type in columns_dict.items() 
                               if c_type in ['string', 'varchar']]
        
        indexer = StringIndexer(inputCols=categorical_columns,
                                outputCols=[f"{col}_indexed" for col in categorical_columns],
                                handleInvalid="keep")
        
        model = indexer.fit(self.data)
            
        self.labels = self._save_labels_table(model, categorical_columns)

        return (
                    model
                    .transform(self.data)
                    .select(*[
                            F.col(col).cast("float") if col in double_columns else
                            F.col(f"{col}_indexed").cast("int").alias(col) if col in categorical_columns else
                            F.col(col) for col in self.data.columns
                            ])
                    .toPandas()
                )
        
    @staticmethod
    def _save_labels_table(model, colums: List[str]):
        labels = model.labelsArray
        return {col : pd.DataFrame({"labels": label, 
                                    "encoded" : range(len(label))}) for label, col in zip(labels, colums)}
    
    @property
    def labels_dict(self):
        return self.labels