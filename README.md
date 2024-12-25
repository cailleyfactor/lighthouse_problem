## S2 Coursework
## Description
The aim of this project was to solve the "Lighthouse Problem", including using Markov chain Monte Carlo methods to sample from the relevant posterior distributions to provide parameter estimates for the lighthouse position based on the data in the "lighthouse_flash_data.txt" file.
## Usage
First clone the repository from git. Cloning the repository includes cloning the data stored in the "lighthouse_flash_data.txt" file.

To run the code, a dockerfile is provided in the root directory with the environment needed to run the code, provided in the S2_environment.yml file.
To run the code from the terminal navigate to the root directory and use, e.g.,
$docker build -t [image name of choice] .
$docker run -v .:/cf593_doxy -t [image name of choice]
(Make sure to include the periods!)

With the environment in S2_environment.yml, the code can also be run from the terminal
by navigating into the root directory of the cloned git repository and running the code with the following command

$ python main.py lighthouse_flash_data.txt

## Documentation
Detailed documentation is available by running the Doxyfile using doxygen in the docs file in the root directory.
This can be run by navigating in the docs file and running doxygen with:
$doxygen

## License
Released 2024 by Cailley Factor.
The License is included in the LICENSE.txt file.
