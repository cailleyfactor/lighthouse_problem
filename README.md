# S2 Coursework
## Description
The aim of this project was to create . A PDF of a report describing the project
in detail is provided in the report folder in the root directory. Excluding the appendix, the word count for the report
is 2982 words.

## Usage
First clone the repository from git. Cloning the repository includes the

To run the code, we provided a dockerfile in the root directory with the environment needed to run the code, provided in the environment.yml file.
To run the code from the terminal navigate to the root directory and use, e.g.,
$docker build -t [image name of choice] .
$docker run [image name of choice]

With the appropriate environment, the code can also be run from the terminal
by navigating into the root directory of the cloned git repository and running the code with the following command

$ python main.py input.txt

## Documentation
Detailed documentation is available by running the Doxyfile using doxygen in the docs file in the root directory.
This can be run by navigating in the docs file and running doxygen with:
$doxygen

## Auto-generation tool citations
ChatGPT version 4.0 was used for:
- Prototyping the import of the input.txt file into the main.py module.
  - The following prompt was used: "methods of importing files efficiently from another directory in python".
  - Several options were provided and in the end, we decided on simple relative imports.

GitHub Copilot was used to help write documentation for docker and comments within the code.

## License
Released 2024 by Cailley Factor.
The License is included in the LICENSE.txt file.
