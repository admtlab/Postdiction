# data-postdiction-pipeline

A repository containing the code for the Data Postdiction project using machine learning to replace values within a column.

### Dependencies

The Python components of this project were implemented using Python 3.12. Note that the specific versions for most dependencies are provided in requirements.txt which can be installed by running `pip install -r requirements.txt`. To run the pipeline run `python pipeline.py <config-file.yaml>` where the config file matches the configs under `configuration_files/` and the dataset is in the specified location using the `database` parameter.

For the AFD detection using the Pyro algorithm, the `Pyro-distro-1.0-SNAPSHOT-distro` and `metanome-cli-1.1.0` jars are needed to build the setup and open jdk build 11.0.28+6 was used to build and test.

### Environment

All of the experiments were conducted within a WSL2 environment (Ubuntu Version 2404.1.68.1 published by Canonical Group Limited) from Windows 11 and the WSL2 environment was configured to have a memory limit of 16 GB (50 GB for batch tuning) by adding the below parameters to the WSL2 config:

```
# Settings apply across all Linux distros running on WSL 2
[wsl2]

# Limits VM memory to use no more than 16 GB, this can be set as whole numbers using GB or MB
memory=16GB
```

## Citations

Anna Baskin, Scott Heyman, Brian T. Nixon, Constantinos Costa, and Panos K. Chrysanthis, "Remembering the Forgotten: Clustering, Outlier Detection, and Accuracy Tuning in a Postdiction Pipeline," in European Conference on Advances in Databases and Information Systems (ADBIS), 2023, pp. 46-55.
