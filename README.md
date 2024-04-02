## S2 Coursework
## Description
The aim of this project was to solve the "Lighthouse Problem", including using Markov chain Monte Carlo methods to sample from the relevant posterior distributions to provide parameter estimates for the lighthouse position based on the data in the "lighthouse_flash_data.txt" file. A PDF of a report describing the project in detail is provided in the report folder in the root directory. Excluding the appendix, the word count for the report is  words.

## Usage
First clone the repository from git. Cloning the repository includes cloning the data stored in the "lighthouse_flash_data.txt" file.

To run the code, a dockerfile is povided in the root directory with the environment needed to run the code, provided in the environment.yml file.
To run the code from the terminal navigate to the root directory and use, e.g.,
$docker build -t [image name of choice] .
$docker run -t [image name of choice]

With the appropriate environment, the code can also be run from the terminal
by navigating into the root directory of the cloned git repository and running the code with the following command

$ python main.py lighthouse_flash_data.txt

## Documentation
Detailed documentation is available by running the Doxyfile using doxygen in the docs file in the root directory.
This can be run by navigating in the docs file and running doxygen with:
$doxygen

## Auto-generation tool citations
ChatGPT version 4.0 was used for:
- Setting the labels on the sides of the corner plots from sampling from both the two-dimensional and three-dimensional posterior
    -  The following prompt was used alongside the code for the corner and histogram plot: "How to set the labels on the edges of the corner plot"
- Removing the tick marks on the histograms added to the corner plot for sampling
    - The following prompts were used alongside the code for the corner and histogram plot: "How to hide x-axis labels and ticks for all but the bottom row" and "How to hide y-axis labels and ticks for all but the first column"
- My docker container did not flush in real time. I debugged this with ChatGPT and modified my docker code, accordingly.
    - I used the following prompts, alongside my dockerfile script: "Why isn't the output of my dockerfile flushing in real time on my Mac?" "What can I change such that my dockerfile flushes the output - I know it has something to do with conda".

GitHub Copilot was used to help write documentation for docker and comments within the code.

## License
Released 2024 by Cailley Factor.
The License is included in the LICENSE.txt file.
