from functools import wraps
from typing import Union, Any, Callable, cast

from docs.mock_docs import MyClass
from hypex.utils import DecoratedType, DocstringInheritDecorator


def inherit_docstring_from(
    source: Union[Callable[..., Any], property]
) -> DocstringInheritDecorator:
    """
    A decorator to inherit the docstring from another function or property.
    This decorator can be applied to both callable objects and properties. It copies the docstring
    from the source object to the decorated object if the latter does not already have a docstring.
    Args:
        source (Callable[..., Any] or property): The object from which the docstring will be inherited.
                                                 This should be either a callable or a property that has
                                                 a well-defined __doc__ attribute.
    Returns:
        Callable[[DecoratedType], DecoratedType]: A decorator that when applied to a function or property,
                                                  sets its __doc__ attribute to that of the source.
    Raises:
        TypeError: If the object to be decorated is neither a callable nor a property.
    Usage:
        class SomeClass:
            @property
            @inherit_docstring_from(pd.DataFrame.iloc)
            def iloc(self):
                return self._data.iloc
            @inherit_docstring_from(pd.DataFrame.mean)
            def mean(self):
                return self._data.mean()
    Example:
        >>> my_instance = MyClass()
        >>> print(my_instance.attr1.__doc__)
        >>> print(my_instance.__init__().__doc__)
    """

    def decorator(obj: DecoratedType) -> DecoratedType:
        """
        Apply the inherited docstring to a given function or property.
        This function acts as a decorator within 'inherit_docstring_from', applying the docstring
        from the 'source' object to the 'obj'. If 'obj' is a property, it modifies the property to include
        the source's docstring. If 'obj' is a callable, it wraps the callable in a function that preserves
        the original callable's functionality and metadata but updates the docstring.
        Args:
            obj (DecoratedType): The function or property to which the docstring will be applied.
                                 It must be either a callable or a property.
        Returns:
            DecoratedType: The original object with the updated docstring. If the object is a property,
                           it returns a new property object with the inherited docstring. If it's a callable,
                           it returns the wrapped callable with the updated docstring.
        Raises:
            TypeError: If 'obj' is neither a callable nor a property.
        """
        if isinstance(obj, property):
            doc = getattr(source, "__doc__", "No documentation provided.")
            return property(obj.fget, obj.fset, obj.fdel, doc)
        elif callable(obj):

            @wraps(obj)
            def wrapper(*args, **kwargs) -> Any:
                return obj(*args, **kwargs)

            wrapper.__doc__ = getattr(source, "__doc__", "No documentation provided.")
            return cast(DecoratedType, wrapper)
        else:
            raise TypeError(
                "The decorator can only be applied to callables or properties."
            )

    return decorator
