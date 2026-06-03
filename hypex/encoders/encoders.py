from __future__ import annotations

from ..dataset import Dataset
from ..extensions.encoders import DummyEncoderExtension
from .abstract import Encoder
from ..utils.registry import backend_factory

class DummyEncoder(Encoder):
    """
    Encoder that creates dummy (one-hot) encoded variables for categorical features.

    This class serves as a high-level interface for dummy encoding within the 
    experiment pipeline. It does not perform the computation directly, but rather 
    delegates the task to a backend-specific extension (e.g., Pandas or Spark) 
    resolved dynamically via the `backend_factory`.

    Inherits from:
        Encoder: The base abstract class for all data encoders in the HypEx library.
    
    Example:
        >>> # Typically used internally by the preprocessing pipeline
        >>> encoder = DummyEncoder(target_roles="category_column")
        >>> encoded_data = encoder.execute(experiment_data)
    """

    @staticmethod
    def _inner_function(
        data: Dataset, target_cols: str | None = None, **kwargs
    ) -> Dataset:
        """
        Perform the core dummy encoding logic on the specified target columns.

        This method checks if target columns are provided. If no columns are 
        specified, it safely returns an empty Dataset. Otherwise, it resolves 
        the appropriate backend-specific encoder implementation (based on the 
        input data's backend) and executes the encoding calculation.

        Args:
            data (Dataset): The input dataset containing the features to be encoded.
            target_cols (str | None, optional): The name(s) of the column(s) to be 
                encoded. If None or empty, no encoding is performed. Defaults to None.
            **kwargs: Additional keyword arguments passed directly to the 
                backend-specific encoder implementation (e.g., `DummyEncoderExtension`).

        Returns:
            Dataset: A new Dataset containing the newly created dummy-encoded columns 
            with appropriately assigned roles. Returns an empty Dataset if no 
            `target_cols` are provided.
        """
        if not target_cols:
            return Dataset.create_empty()
        
        encoder_cls = backend_factory.resolve_backend(DummyEncoderExtension, data)
        return encoder_cls.calc(
            data=data, target_cols=target_cols, **kwargs
        )
