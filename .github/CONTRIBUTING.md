# Contributing to HypEx

## Table of contents

- [Contributing to HypEx](#contributing-to-hypex)
- [Beginning](#beginning-)
- [Pull Requests](#pull-requests)
    - [Types of Pull Requests](#types-of-pull-requests)
- [Codebase structure](#codebase-structure)
- [Developing HypEx](#developing-hypex)
    - [Installation](#installation)
    - [Style Guide](#style-guide)
- [Documentation](#documentation)
    - [Building Documentation](#building-documentation)
    - [Writing Documentation](#writing-documentation)
    - [Adding new submodules](#adding-new-submodules)
    - [Adding Tutorials](#adding-tutorials)

## Beginning

We're thrilled you're considering contributing to HypEx! Your contributions help keep HypEx robust and valuable. Here's
how you can get involved:

- First, please look for discussions on this topic in [issues](https://github.com/sb-ai-lab/HypEx/issues)
  before implementing anything inside the project.
- Pick an issue and comment that you would like to work on it.
- If there is no discussion on this topic, create one.
  Please, include as much information as you can,
  any accompanying data (your tests, expected behavior, articles),
  and maybe your proposed solution.
- If you need more details, please ask we will provide them ASAP.

Once you implement and test your feature or bug-fix, please submit
a Pull Request to <https://github.com/sb-ai-lab/HypEx>.

When adding functionality, please add examples that will fully explain it.
Examples can be added in several ways:

- [Inside the documentation](#writing-documentation)
- [Jupyter notebooks](#adding-tutorials)
- [Your own tests](#testing-)

## Pull Requests

We welcome pull requests from the community! Below are the types of pull requests we accept and the templates you should
use for each.

### Types of Pull Requests

1. **Feature Additions or Enhancements**
    - Use the [Feature PR Template](#feature-pull-request-template).
2. **Bug Fixes**
    - Use the [Bug Fix PR Template](#bug-fix-pull-request-template).
3. **Documentation Updates**
    - Use the [Documentation Update PR Template](#documentation-update-pull-request-template).

### Feature Pull Request Template

#### Description

<!-- Briefly describe the problem or user story this PR addresses. -->

#### Changes Made

<!-- Detail the code changes made. Include code snippets or screenshots as needed. -->

#### Related Issues

<!-- Link to related issues or feature requests. -->

#### Additional Notes

<!-- Include any extra information or considerations for reviewers, such as impacted areas of the codebase or specific areas needing thorough review. -->

#### Testing and Validation

<!-- Describe how the changes have been tested and validated. -->

#### Performance Considerations

<!-- Discuss any performance implications of the changes. -->

#### Breaking Changes

<!-- Indicate if the changes introduce any breaking changes and how they have been handled. -->

#### Dependencies

<!-- List any new dependencies introduced by this PR and reasons for their inclusion. -->

#### Merge Request Checklist

- [ ] Code follows project coding guidelines.
- [ ] Documentation reflects the changes made.
- [ ] Unit tests cover new or changed functionality.
- [ ] Performance and breaking changes have been considered.

### Bug Fix Pull Request Template

<!-- Briefly describe the bug that this PR addresses. Include relevant issue numbers if applicable. -->

#### Steps to Reproduce

<!-- List the steps to reproduce the behavior. This helps reviewers to verify the bug and understand the context. -->

1.
2.
3.
4.

#### Expected Behavior

<!-- Describe what should happen ideally after your changes are applied. -->

#### Actual Behavior

<!-- Describe what is actually happening. Include screenshots or error messages if applicable. -->

#### Changes Made

<!-- Summarize the changes made to fix the bug. Provide code snippets or screenshots as needed. -->

#### Testing Performed

<!-- Describe the tests you ran to verify your changes. Include instructions so reviewers can reproduce. -->

#### Related Issues

<!-- Link any related issues here. This helps to track the history and context of the bug. -->

#### Additional Notes

<!-- Include any extra information or considerations for reviewers. -->

#### Checklist

- [ ] The code follows project coding guidelines.
- [ ] I have added tests to cover my changes.
- [ ] All new and existing tests passed.
- [ ] Documentation has been updated to reflect the changes made.
- [ ] I have verified that the changes fix the issue as described.

### Documentation Update Pull Request Template

<!-- Provide a brief description of what the documentation update entails and the reason for the changes. -->

#### Areas of Documentation Updated

<!-- List the sections or pages of the documentation that have been updated. -->

1.
2.
3.

#### Details of Changes

<!-- Describe the specific changes made to the documentation. Include reasons for changes, if not obvious. -->

#### Screenshots / Code Snippets

<!-- If applicable, add screenshots or code snippets to help explain the changes. -->

#### Related Issues or Pull Requests

<!-- Link any related issues or previous pull requests that are relevant to this documentation update. -->

#### Additional Notes

<!-- Include any additional information that might be helpful for reviewers. -->

#### Checklist

- [ ] The changes are clear and easy to understand.
- [ ] I have verified that the changes are accurate and necessary.
- [ ] The updated documentation has been tested for clarity and comprehensibility.
- [ ] All modified sections are properly formatted and adhere to project documentation standards.

## How to Submit a Pull Request

1. Fork the repository and create your branch from `master`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints.
6. Issue the pull request!

## Codebase structure

- [docs](../docs) - For documenting we use [Sphinx](https://www.sphinx-doc.org/).
  It provides easy to use auto-documenting via docstrings.
    - [Tutorials](../examples/tutorials) - Notebooks with tutorials.

- [hypex](../hypex) - The code of HypEx library.
    - [ab_test](../hypex/ab_test) - The main module for A/B testing
    - [algorithms](../hypex/algorithms) - Modules with different algorithms for calculation (Like Knn)
      blenders and ready-made presets.
    - [dataset](../hypex/dataset) - The internal interface for working with data. Also contains data generator
    - [selectors](../hypex/selectors) - Modules with different pre-processing data (Filters, feature selectors)
    - [utils](../hypex/utils) - Common util tools (Timer, Profiler, Logging).

## Developing HypEx

### Installation

If you are installing from the source, you will need Python 3.8 or later.

1. Install poetry using [the poetry installation guide](https://python-poetry.org/docs/#installation).

2. Clone the project to your own local machine:

    ```bash
    git clone git@github.com:sb-ai-lab/HypEx.git
    cd HypEx
    ```

3. **Optional**: specify python for poetry

    ```bash
    poetry env use PYTHON_PATH
    ```

4. Install HypEx :

   To install only necessary dependencies, without extras:
    ```bash
    poetry install
    ```

   To install all dependencies:
    ```bash
    poetry install --all-extras
    ```

   To install only specific dependency groups:
    ```
    poetry install -E cv -E report
    ```

After that, there is virtual environment, where you can test and implement your own code.
So, you don't need to rebuild the full project every time.
Each change in the code will be reflected in the library inside the environment.

### Style Guide

We follow [PEP8 standards](https://www.python.org/dev/peps/pep-0008/). Automated code quality checks are in progress.

#### Automated code checking (in progress)

In order to automate checking of the code quality, we use
[pre-commit](https://pre-commit.com/). For more details, see the documentation,
here we will give a quick-start guide:

1. Install and configure:

```console
poetry run pre-commit install
```

2. Now, when you run `$ git commit`, there will be a pre-commit check.
   This is going to search for issues in your code: spelling, formatting, etc.
   In some cases, it will automatically fix the code, in other cases, it will
   print a warning. If it automatically fixed the code, you'll need to add the
   changes to the index (`$ git add FILE.py`) and run `$ git commit` again. If
   it didn't automatically fix the code, but still failed, it will have printed
   a message as to why the commit failed. Read the message, fix the issues,
   then recommit.
3. The pre-commit checks are done to avoid pushing and then failing. But, you
   can skip them by running `$ git commit --no-verify`, but note that the C.I.
   still does the check so you won't be able to merge until the issues are
   resolved.
   If you experience any issues with pre-commit, please ask for support on the
   usual help channels.

### Testing
(in progress)

## Documentation

Before writing the documentation, you should collect it to make sure that the code
you wrote doesn't break the rest of the documentation. The library might work,
but the documentation might not be. It is built on the Read the Docs service,
which uses its own virtual environment, which contains only part
of the HypEx library dependencies. This is done to make
the documentation more lightweight.

By default, functions, that have no description will be mock from overall documentation.

### Building Documentation

To build the documentation:

1. Clone repository to your device.

```bash
git clone git@github.com:sb-ai-lab/HypEx.git
cd HypEx
```

2. Make environment and install requirements.

```bash
poetry install -E cv -E nlp
```

3. Remove existing html files:

```bash
cd docs
poetry run make clean html
```

4. Generate HTML documentation files. The generated files will be in `docs/_build/html`.

```bash
poetry build
```

### Writing Documentation

There are some rules, that docstrings should fit.

1. HypEx
   uses [Google-style docstring formatting](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html).
   The length of the line inside docstring should be limited
   to 80 characters to fit into Jupyter documentation popups.

2. Every non-one-line docstring should have a paragraph at its end, regardless of where it will be used:
   in the documentation for a class, module, function, class
   method, etc. One-liners or descriptions,
   that have no special directives (Args, Warning, Note, etc.) may have no paragraph at its end.

3. Once you added some module to HypEx,
   you should add some info about it at the beginning of the module.
   Example of this you can find in `docs/mock_docs.py`.
   Also, if you use submodules, please add description to `__init__.py`
   (it is usefully for Sphinx's auto summary).

4. Please use references to other submodules. You can do it by Sphinx directives.
   For more information: <https://www.sphinx-doc.org/en/master/usage/restructuredtext/domains.html>
5. There is an example for documenting standalone functions.

```python3
from typing import List, Union

import numpy as np
import torch


def typical_function(a: int, b: Union['np.ndarray', None] = None) -> List[int]:
    """Short function description, terminated by dot.

    Some details. The parameter after arrow is return value type.

    Use 2 newlines to make a new paragraph,
    like in `LaTeX by Knuth <https://en.wikipedia.org/wiki/LaTeX>`_.

    Args:
        a: Parameter description, starting with a capital
          latter and terminated by a period.
        b: Textual parameter description.

    .. note::
        Some additional notes, with special block.

        If you want to itemize something (it is inside note):

            - First option.
            - Second option.
              Just link to function :func:`torch.cuda.current_device`.
            - Third option.
              Also third option.
            - It will be good if you don't use it in args.

    Warning:
        Some warning. Every block should be separated
        with other block with paragraph.

    Warning:
        One more warning. Also notes and warnings
        can be upper in the long description of function.

    Example:

        >>> print('MEME'.lower())
        meme
        >>> b = typical_function(1, np.ndarray([1, 2, 3]))

    Returns:
        Info about return value.

    Raises:
        Exception: Exception description.

    """

    return [a, 2, 3]
```

6. Docstring for generator function.

```python3
def generator_func(n: int):
    """Generator have a ``Yields`` section instead of ``Returns``.

    Args:
        n: Number of interations.

    Yields:
        The next number in the range of ``0`` to ``n-1``.

    Example:
        Example description.

        >>> print([i for i in generator_func(4)])
        [0, 1, 2, 3]

    """
    x = 0
    while x < n:
        yield x
        x += 1
```

7. Documenting classes.

```python3
from typing import List, Union
import numpy as np
import torch


class ExampleClass:
    """The summary line for a class that fits only one line.

    Long description.

    If the class has public attributes, they may be documented here
    in an ``Attributes`` section, like in ``Args`` section of function.

    Properties created with the ``@property`` decorator should be documented
    in the property's getter method. Use arrow to set the return type.

    On the stage before __init__ we don't know anything about `Attributes`,
    so please, add description about it's types.

    Attributes:
        attr1 (str): Description of `attr1`.
        attr2 (:obj:`int`, optional): Description of `attr2`.

    """

    def __init__(self, param1: int, param2: 'np.ndarray', *args, **kwargs):
        """Example of docstring of the __init__ method.

        Note:
            You can also add notes as ``Note`` section.
            Do not include the `self` parameter in the ``Args`` section.

        Args:
            param1: Description of `param1`.
            param2: Description of `param2`.
            *args: Description of positional arguments.
            **kwargs: Description of key-word arguments.

        """
        self.attr1 = param1
        self.attr2 = param2
        if len(args) > 0:
            self.attr2 = args[0]
        self.attr3 = kwargs  # will not be documented.
        self.figure = 4 * self.attr1

    @property
    def readonly_property(self) -> str:
        """Properties should be documented in
        their getter method.

        """
        return 'lol'

    @property
    def readwrite_property(self) -> List[str]:
        """Properties with both a getter and setter
        should only be documented in their getter method.

        If the setter method contains notable behavior, it should be
        mentioned here.
        """
        return [str(self.figure)]

    @readwrite_property.setter
    def readwrite_property(self, value: int):
        self.figure = value

    def some_method(self, param1: int, param2: float = np.pi) -> List[int]:
        """Just like a functions.

        Long description.

        Warning:
            This method do something. May be undefined-behaviour.

        Args:
            param1: Some description of param1.
            param2: Some description of param2. Default value
               will be contained in signature of function.

        Returns:
            Array with `1`, `2`, `3`.

        """
        self.attr1 = param1
        self.attr2 += param2

        return [1, 2, 3]

    def __special__(self):
        """By default we aren`t include dundered members.

        Also there may be no docstring.
        """
        pass

    def _private(self):
        """By default we aren't include private members.

        Also there may be no docstring.
        """
        pass

    @staticmethod
    def static_method(param1: int):
        """Description of static method.

        Note:
            As like common method of class don`t use `self`.

        Args:
            param1: Description of `param1`.

        """
        print(param1)
```

8. If you have a parameter that can take a finite number of values,
   if possible, describe each of them in the Note section.

```python3
import random


class A:
    """
    Some description.

    Some long description.

    Attributes:
        attr1 (:obj:`int`): Description of `attr1`.
        attr2 (:obj:`int`): Description of `attr2`.

    """

    def __init__(self, weight_initialization: str = 'none'):
        """

        Args:
            weight_initialization: Initialization type.

        Note:
            There are several initialization types:

                - '`zeros`': fill ``attr1``
                  and ``attr2`` with zeros.
                - '`ones`': fill ``attr1``
                  and ``attr2`` with ones.
                - '`none`': fill ``attr1``
                  and ``attr2`` with random int in `\[0, 100\]`.

        Raises:
            ValueError: If the entered initialization type is not supported.

        """
        if weight_initialization not in ['zeros', 'ones', 'none']:
            raise ValueError(
                f'{weight_initialization} - Unsupported weight initialization.')

        if weight_initialization == 'zeros':
            attr1 = 0
            attr2 = 0
        elif weight_initialization == 'ones':
            attr1 = 1
            attr2 = 1
        else:
            attr1 = random.randint(0, 100)
            attr2 = random.randint(0, 100)
```

### Adding new submodules

If you add your own directory to HypEx, you should add a corresponding module as new `.rst`
file to the `docs/`. And also mention it in `docs/index.rst`.

If you add your own module, class or function, then you will need
to add it description to the corresponding `.rst` in `docs`.

### Adding Tutorials

We use [nbsphinx](https://nbsphinx.readthedocs.io/) extension for tutorials.
Examples, you can find in `docs/notebooks`.
Please, put your tutorial in this folder
and after add it in `docs/Tutorials.rst`.
