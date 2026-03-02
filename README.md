# data-postdiction-pipeline

A repository containing the code for the Data Postdiction project using machine learning to replace values within a column.

### Dependencies

The Python components of this project were implemented using Python 3.12. Note that the specific versions for most dependencies are provided in requirements.txt which can be installed by running `pip install -r requirements.txt`. To run the pipeline run `python pipeline.py <config-file.yaml>` where the config file matches the configs under `configuration_files/` and the dataset is in the specified location using the `database` parameter.

For the AFD detection using the Pyro algorithm, the `Pyro-distro-1.0-SNAPSHOT-distro` and `metanome-cli-1.1.0` jars are needed to build the setup and open jdk build 11.0.28+6 was used to build and test.

### Environment

All of the experiments were conducted on physical hardware using a Thinkpad P14S Gen 5 equipped with an Intel Ultra 7 155H CPU with 22 logical processors with 64 GB of RAM. Scalability experiments were also conducted on a Thinkpad T470 equipped with an Intel Core i5-7300U CPU with 4 logical processors and 16 GB of RAM.

## Citations

Anna Baskin, Scott Heyman, Brian T. Nixon, Constantinos Costa, and Panos K. Chrysanthis, "Remembering the Forgotten: Clustering, Outlier Detection, and Accuracy Tuning in a Postdiction Pipeline," in European Conference on Advances in Databases and Information Systems (ADBIS), 2023, pp. 46-55.
