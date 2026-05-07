import inspect
import unittest
from abc import ABC, ABCMeta, abstractmethod


class StrictABCMeta(ABCMeta):
    """A metaclass that enforces Liskov Substitution Principle (LSP) compliance
    by validating method signatures in concrete subclasses against their abstract
    declarations.

    This metaclass extends Python's built-in ``abc.ABCMeta`` by intercepting class
    creation. For every abstract method inherited from a parent class, it compares
    the subclass's concrete implementation signature with the original abstract
    signature. If a signature violates LSP constraints, a ``TypeError`` is raised
    immediately at definition time, preventing the creation of non-compliant classes.

    Configuration
    -------------
    Validation behavior is controlled via the ``__strict_options__`` class attribute,
    which should be a dictionary. Available keys:

    * ``check_names`` (``bool``, default ``False``):
      Enforce exact parameter name matching. When ``True``, the child method must
      use the exact same parameter names as the parent.

    * ``check_defaults`` (``bool``, default ``True``):
      Forbid removing default parameter values. LSP dictates that preconditions
      cannot be strengthened; therefore, making a previously optional argument
      mandatory is a violation. Adding new defaults or keeping existing ones is
      permitted.

    * ``check_types`` (``bool``, default ``False``):
      Enforce exact type annotation matching for parameters. Both parent and child
      must declare identical type hints for corresponding parameters.

    * ``check_return_type`` (``bool``, default ``False``):
      Enforce covariant return type compatibility. The child's return type must be
      a subclass of (or identical to) the parent's declared return type.

    Validation Rules & LSP Guarantees
    ----------------------------------
    Beyond the configurable options, the metaclass enforces several structural
    rules derived from the Liskov Substitution Principle:

    1. **Descriptor Consistency**: Methods decorated with ``@staticmethod``,
       ``@classmethod``, or left as instance methods must match the parent's
       descriptor type exactly.

    2. **Variadic Parameter Handling**:
       - If a parent uses ``*args`` or ``**kwargs``, the child must retain them.
         Replacing variadics with fixed parameters is considered signature narrowing
         and is forbidden.
       - Adding ``*args``/``**kwargs`` to a fixed parent signature is allowed
         (expands the contract safely).
       - When variadics are present on both sides, the child must preserve all
         variadic kinds declared in the parent.

    3. **Parameter Kind Transitions**: Transitions that relax calling syntax are
       permitted (e.g., ``POSITIONAL_ONLY`` → ``POSITIONAL_OR_KEYWORD`` or
       ``KEYWORD_ONLY`` → ``POSITIONAL_OR_KEYWORD``). Restrictive transitions
       that limit how arguments can be passed raise a ``TypeError``.

    4. **Parameter Count**: Fixed-parameter signatures must have an identical
       number of parameters.

    5. **Default Value Rules**: Child implementations may introduce new defaults,
       but cannot remove defaults declared in the parent (unless
       ``check_defaults=False``).

    6. **Type Covariance**: When ``check_return_type=True``, return types must
       follow covariance rules (child type ``<=`` parent type in inheritance).

    Example
    -------
    Basic usage with default settings (prevents removing defaults)::

        from abc import abstractmethod
        from strict_abc import StrictABCMeta

        class BaseService(metaclass=StrictABCMeta):
            @abstractmethod
            def process(self, data: dict, cache: bool = True) -> str: ...

        # Valid: keeps defaults, matches signature
        class ValidImpl(BaseService):
            def process(self, data: dict, cache: bool = True) -> str:
                ...

        # Invalid: removes default for 'cache' (strengthens precondition)
        class InvalidImpl(BaseService):
            def process(self, data: dict, cache: bool) -> str:  # Raises TypeError
                ...

    Custom configuration::

        class StrictTypedBase(metaclass=StrictABCMeta):
            __strict_options__ = {
                'check_names': True,
                'check_defaults': True,
                'check_types': True,
                'check_return_type': True
            }

            @abstractmethod
            def fetch(self, url: str) -> bytes: ...

    Notes
    -----
    * Validation occurs at **class definition time**, failing fast before
      instantiation.
    * Methods without inspectable signatures (e.g., C extensions or built-ins)
      are skipped gracefully to avoid ``ValueError``.
    * Designed to be fully compatible with Python's standard ``abc`` module.
    * For convenience, a pre-configured base class ``StrictABC`` is provided
      with sensible defaults.

    Raises
    ------
    TypeError
        If a subclass method violates any of the enforced LSP signature rules.
    """

    def __new__(mcls, name: str, bases: tuple, namespace: dict, **kwargs):
        cls = super().__new__(mcls, name, bases, namespace, **kwargs)

        options = getattr(cls, '__strict_options__', {})
        check_names = options.get('check_names', False)
        check_defaults = options.get('check_defaults', True)
        check_types = options.get('check_types', False)
        check_return_type = options.get('check_return_type', False)

        # Collect unresolved abstract methods from direct parents
        parent_abstracts = set()
        for base in bases:
            parent_abstracts.update(getattr(base, '__abstractmethods__', frozenset()))

        for meth_name in parent_abstracts:
            concrete = cls.__dict__.get(meth_name)
            if concrete is None or getattr(concrete, '__isabstractmethod__', False):
                continue

            # Find the abstract definition via MRO
            parent_attr = None
            for base in cls.__mro__[1:]:
                attr = base.__dict__.get(meth_name)
                if attr is not None and getattr(attr, '__isabstractmethod__', False):
                    parent_attr = attr
                    break
            if parent_attr is None:
                continue

            try:
                sig_parent = inspect.signature(parent_attr)
                sig_child = inspect.signature(concrete)
            except (ValueError, TypeError):
                continue

            mcls._validate_signature(
                meth_name, sig_parent, sig_child, parent_attr, concrete,
                check_names, check_defaults, check_types, check_return_type,
                cls.__name__
            )

        return cls

    @staticmethod
    def _validate_signature(
        meth_name: str, sig_parent: inspect.Signature, sig_child: inspect.Signature,
        parent_attr: object, concrete: object,
        check_names: bool, check_defaults: bool, check_types: bool,
        check_return_type: bool, class_name: str
    ) -> None:
        def _descriptor_type(obj):
            if isinstance(obj, staticmethod):
                return "staticmethod"
            if isinstance(obj, classmethod):
                return "classmethod"
            return "instancemethod"

        p_type = _descriptor_type(parent_attr)
        c_type = _descriptor_type(concrete)
        if p_type != c_type:
            raise TypeError(
                f"{class_name}.{meth_name}: descriptor type mismatch "
                f"(expected {p_type}, got {c_type})"
            )

        p_params = list(sig_parent.parameters.values())
        c_params = list(sig_child.parameters.values())

        # Strip implicit receiver by position (not by name)
        if p_type != "staticmethod" and p_params:
            p_params = p_params[1:]
        if c_type != "staticmethod" and c_params:
            c_params = c_params[1:]

        # ------------------------------------------------------------------
        # LSP-compliant variadic policy (handles *args / **kwargs correctly)
        # ------------------------------------------------------------------
        p_has_var_pos = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in p_params)
        p_has_var_kw  = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in p_params)
        c_has_var_pos = any(c.kind == inspect.Parameter.VAR_POSITIONAL for c in c_params)
        c_has_var_kw  = any(c.kind == inspect.Parameter.VAR_KEYWORD for c in c_params)

        parent_has_variadic = p_has_var_pos or p_has_var_kw
        child_has_variadic  = c_has_var_pos or c_has_var_kw

        # Case 1: Parent has variadic, child removed it entirely -> narrowing (violation)
        if parent_has_variadic and not child_has_variadic:
            raise TypeError(
                f"{class_name}.{meth_name}: narrowing variadic parent signature to fixed "
                f"parameters violates LSP"
            )

        # Case 2: Both have variadic – child must retain ALL variadic kinds of parent
        if parent_has_variadic and child_has_variadic:
            if p_has_var_pos and not c_has_var_pos:
                raise TypeError(
                    f"{class_name}.{meth_name}: removed *args from parent signature "
                    f"(violates LSP)"
                )
            if p_has_var_kw and not c_has_var_kw:
                raise TypeError(
                    f"{class_name}.{meth_name}: removed **kwargs from parent signature "
                    f"(violates LSP)"
                )
            # If all parent variadic kinds are preserved, signatures are compatible;
            # skip detailed positional matching (variadic absorbs everything)
            return

        # Case 3: Parent strict, child adds variadic -> expands contract (LSP-safe)
        if not parent_has_variadic and child_has_variadic:
            return

        # At this point neither parent nor child have variadic parameters.
        # Perform strict signature comparison for fixed parameters.
        if len(p_params) != len(c_params):
            raise TypeError(
                f"{class_name}.{meth_name}: expected {len(p_params)} parameters, "
                f"found {len(c_params)}"
            )

        # LSP-compliant defaults: child can add defaults, but must not remove them
        if check_defaults:
            for p, c in zip(p_params, c_params):
                p_def = p.default is not inspect.Parameter.empty
                c_def = c.default is not inspect.Parameter.empty
                if p_def and not c_def:
                    raise TypeError(
                        f"{class_name}.{meth_name}: removing default value for '{p.name}' "
                        f"strengthens precondition and violates LSP"
                    )

        # Allowed kind transitions (from more restrictive to less restrictive)
        ALLOWED_KIND_EXPANSIONS = {
            (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD),
            (inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD),
        }

        for p, c in zip(p_params, c_params):
            if p.kind != c.kind and (p.kind, c.kind) not in ALLOWED_KIND_EXPANSIONS:
                raise TypeError(
                    f"{class_name}.{meth_name}: parameter '{p.name}' kind transition "
                    f"{p.kind.name} -> {c.kind.name} restricts call syntax and violates LSP"
                )

            if check_names and p.name != c.name:
                raise TypeError(
                    f"{class_name}.{meth_name}: parameter name mismatch "
                    f"(expected '{p.name}', got '{c.name}')"
                )

            if check_types:
                p_ann = p.annotation is not inspect.Parameter.empty
                c_ann = c.annotation is not inspect.Parameter.empty
                if p_ann and not c_ann:
                    raise TypeError(
                        f"{class_name}.{meth_name}: missing type annotation for '{p.name}'"
                    )
                if p_ann and c_ann and p.annotation != c.annotation:
                    raise TypeError(
                        f"{class_name}.{meth_name}: type annotation mismatch for '{p.name}' "
                        f"(expected {p.annotation}, got {c.annotation})"   # <-- FIXED: c_annotation -> c.annotation
                    )

        # Return type validation (covariance)
        if check_return_type:
            p_ret = sig_parent.return_annotation
            c_ret = sig_child.return_annotation
            if p_ret is not inspect.Signature.empty and c_ret is inspect.Signature.empty:
                raise TypeError(
                    f"{class_name}.{meth_name}: missing return type annotation"
                )
            if p_ret is not inspect.Signature.empty and c_ret is not inspect.Signature.empty:
                is_covariant = False
                if isinstance(p_ret, type) and isinstance(c_ret, type):
                    try:
                        is_covariant = issubclass(c_ret, p_ret)
                    except TypeError:
                        pass
                if not is_covariant and c_ret != p_ret:
                    raise TypeError(
                        f"{class_name}.{meth_name}: return type not covariant "
                        f"(expected {p_ret}, got {c_ret})"
                    )
                    
class StrictABC(ABC, metaclass=StrictABCMeta):
    __strict_options__ = {
        'check_names': False,
        'check_defaults': True,
        'check_types': False,
        'check_return_type': False
    }
