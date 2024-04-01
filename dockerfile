# Use the official Miniconda3 image as the base image
FROM continuumio/miniconda3

# Set the working directory in the container
WORKDIR /cf593_doxy

# Copy the current directory contents into the container
COPY . /cf593_doxy

# Create the Conda environment
RUN conda env create -f S2_environment.yml

# Activate the Conda environment
SHELL ["conda", "run", "-n", "S2_environment", "/bin/bash", "-c"]

# Ensure Python outputs are flushed immediately
ENV PYTHONUNBUFFERED=1

# Run the Python script
CMD ["conda", "run", "--no-capture-output", "-n", "S2_environment", "python", "main.py"]
