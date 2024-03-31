# This is a good minimimal installer for Conda
FROM continuumio/miniconda3

# Set the working directory in the container
WORKDIR /cf593_doxy

# Copy the current directory contents into the container
COPY . /cf593_doxy

# new conda environment created called S2_environment.yml
RUN conda env create -f S2_environment.yml

SHELL ["conda", "run", "-n", "S2_environment", "/bin/bash", "-c"]

# Ensure Python outputs are flushed immediately
ENV PYTHONUNBUFFERED=1

# Run the command with the following command
CMD ["conda", "run", "-n", "S2_environment", "python", "-u", "main.py", "lighthouse_flash_data.txt"]
