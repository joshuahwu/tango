# Docstrings

## Overview

A docstring, short for documentation string, is a form of inline documentation written as a comment in the source code. It offers a comprehensive description of a module, class, method, or function, intended primarily for developers and advanced users. The typical syntax for a docstring involves enclosing the documentation within triple double quotes as shown below:

```python
"""This is a docstring.

It can span across multiple lines.
"""
```

## Docstring Best Practices

### Docstring Style
We recommend adhering to a recognized docstring style for consistency and readability. We recommend the [numpydoc](https://numpydoc.readthedocs.io/en/latest/format.html) style (over others like Google or pep257). Following a standard style has several benefits:

1. **Improved Code Readability:** A standard format is familiar to developers, including yourself in the future, allowing for quick access to specific information.
2. **Integration with Documentation Tools:** Tools like `MkDocs` and `sphinx` can parse standardized docstring formats to automatically generate comprehensive and visually appealing code documentation.
3. **Consistency:** Adhering to a single docstring style assures consistency across your code base.
4. **Facilitated Onboarding of New Members:** When a new member joins your team, your docstrings standards will be clear and familiar from the get-go.
5. **Reduced Risk of Misinterpretation:** With a set standard there will be less ambiguity in your docstrings specifications, reducing the risk of misinterpretation.

### Docstring Linting
Failing to conform to convention standards may result in improper documentation rendering. As your package grows, these violations may become harder to spot. We therefore strongly suggest using [`pydocstyle`](http://www.pydocstyle.org/en/latest/), a linter that checks the docstrings for you. You can find additional information about linters and code style in the [Linters and Tests](#05-linters-and-tests/) section.

### Type Annotations 

We also require the use of type annotations, which involves denoting the data types for variables, function parameters, and return types. Below is an example of a function with type annotations:

```python
def greeting(name: str) -> str:
    return 'Hello ' + name
```

Type annotations offer several advantages:

1. **Enhanced Code Readability:** By explicitly indicating data types, type hints make the code more readable and self-explanatory.
2. **Improved Maintainability:** Type annotations clarify the expected data types, facilitating easier maintenance and reducing errors due to misinterpretations.
3. **Static Type Checking:** Tools such as [`mypy`](https://mypy-lang.org) can inspect the code to detect type-related bugs at compile time, allowing developers to rectify type mismatches and errors before running the code.
4. **Code Editor Support:** Modern code editors and Integrated Development Environments (IDEs) can utilize type annotations to offer improved auto-completion, suggestions, and code analysis, thus enhancing the development process.
5. **Enhanced Collaboration:** By providing clear documentation of the expected data types in the codebase, type annotations improve communication among team members.

For the standard Python annotation syntax, you can refer to [PEP 484](https://peps.python.org/pep-0484/).
