# Contributing to LiveChess2FEN

First of all, thank you for reading these lines and taking the time to
 contribute! We are open to new issues and contributions of any kind.
 
## How to Contribute

#### **Did you find a bug or intend to add a new feature or change an existing one?**

- Please check the 
[Issues](https://github.com/davidmallasen/LiveChess2FEN/issues) tab first
to avoid duplicates.

- If there is no open issue addressing the situation, feel free to
[open a new one](https://github.com/davidmallasen/LiveChess2FEN/issues/new)!
Be sure to include a **title and clear description** and as much relevant 
information as possible. 

#### **Do you want to write some piece of code or any other contribution?**

- If you haven't done it already, fork the repository and create a new
 branch with a descriptive name.

- Write and commit the changes to the repo in the new branch. Please be sure
 to check the [styleguides](#style-guides) below.

- Open a new GitHub pull request with the patch!

- Ensure the pull request description clearly describes the thought behind
 the contribution. Include the relevant issue number if there is one open.

## Style Guides

### Git Commit Messages

- Use the present tense ("Add feature" not "Added feature").
- Use the imperative mood ("Move cursor to..." not "Moves cursor to...").
- Limit the first line to 72 characters or less.
- Reference issues and pull requests when needed ("Fix #123").

### Python Style Guide

- Please follow the [PEP 8](https://www.python.org/dev/peps/pep-0008/) and
[PEP 257](https://peps.python.org/pep-0257/) guides in order to keep the code
stylistically consistent.
  
    - To do so, we recommend using [VS Code](https://code.visualstudio.com/) as
    your code editor and taking the following steps:
      
      - Install the
      [Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python),
      [Pylance](https://marketplace.visualstudio.com/items?itemName=ms-python.vscode-pylance),
      and [Black Formatter](https://marketplace.visualstudio.com/items?itemName=ms-python.black-formatter) extensions.

        - The Black Formatter helps format code in the
        [Black](https://github.com/psf/black/tree/main) code style.

      - Open "settings.json" ([here](https://stackoverflow.com/a/70629074) is
      how) and add the following into its content:

        ```
        "[python]": {
            "editor.defaultFormatter": "ms-python.black-formatter",
            "editor.formatOnSave": true
        },
        "black-formatter.args": [
            "--line-length",
            "79"
        ],
        "editor.tokenColorCustomizations": {
            "textMateRules": [
                {
                    "scope": [
                        "string.quoted.docstring", // This includes Python docstrings
                    ],
                    "settings": { // Set your desired color for Python docstrings
                        "foreground": "#0d9419",
                    }
                }
            ]
        },
        "editor.rulers": [
            72,
            79
        ],
        "editor.formatOnSave": true,
        ```

      - When writing your Python code, use the editor rulers to make sure that
      all lines have a maximum length of 79 characters and docstrings and
      multiline comments have a maximum length of 72 characters.

      - When in doubt, check nearby code.

      - Use the [pydocstyle](https://pypi.org/project/pydocstyle/) package to
      check whether your docstrings are written consistently with PEP-257
      conventions.
      