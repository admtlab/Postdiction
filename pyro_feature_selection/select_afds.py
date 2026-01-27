import json
from pathlib import Path
import subprocess
import sys
import os
import heapq
import dask.dataframe as dd

from functools import reduce
from config import get_column_info_table

class FunctionalDependency:
    def __init__(self, json_str: str, metadata_dict: dict, determinants=None, dependant=None):
        if determinants is None and dependant is None:
            data = json.loads(json_str)
            determinant_dicts = data['determinant']['columnIdentifiers']
            self.determinants = list(map(lambda x: x['columnIdentifier'], determinant_dicts))        
            self.dependant = data['dependant']['columnIdentifier']
            
            # Edge Case: Sometimes the library will return AFDs with an empty list of determinants (potentially for columns that are always the same value), in which case, we throw a ValueError
            if not self.determinants:
                raise ValueError(f"Detected an AFD for {self.dependant} with an empty list of determinants. Skipping...")
            
            determinant_sizes = list(map(lambda x: metadata_dict[x][1], self.determinants))
            self.determinant_size = reduce(lambda x, y: x + y, determinant_sizes)
            self.dependant_size = metadata_dict[self.dependant][1]
            
            # In addition to the total size for ordering, maintain the size of each determinant to support subtracting size when traversing AFDs
            self.determinant_size_dict = {}
            for determinant in self.determinants:
                self.determinant_size_dict[determinant] = metadata_dict[determinant][1]
        else:
            self.determinants = determinants
            self.dependant = dependant
            
            # Edge Case: Sometimes the library will return AFDs with an empty list of determinants (potentially for columns that are always the same value), in which case, we throw a ValueError
            if not self.determinants:
                raise ValueError(f"Detected an AFD for {self.dependant} with an empty list of determinants.")
            
            determinant_sizes = list(map(lambda x: metadata_dict[x][1], self.determinants))
            self.determinant_size = reduce(lambda x, y: x + y, determinant_sizes)
            self.dependant_size = metadata_dict[self.dependant][1]
            
            # In addition to the total size for ordering, maintain the size of each determinant to support subtracting size when traversing AFDs
            self.determinant_size_dict = {}
            for determinant in self.determinants:
                self.determinant_size_dict[determinant] = metadata_dict[determinant][1]
    
    def reset_determinant_sizes(self):
        """ A helper function for restoring the size of the AFD when returning selected AFDs after performing a search/selection
        """
        determinant_sizes = list(map(lambda x: self.determinant_size_dict[x], self.determinants))
        self.determinant_size = reduce(lambda x, y: x + y, determinant_sizes)
    
    
    def has_attrs_as_determinants(self, lhs: list[str]) -> bool:
        """ A function for checking if a list of attributes are all part of the functional dependency's determinants.

        Args:
            lhs (list[str]): The list of attributes to check for.

        Returns:
            bool: True if all attributes in lhs are part of the functional dependency's determinants. False, otherwise.
        """
        return reduce(lambda x, y: x and self.has_attr_as_determinant(y), lhs, True)
    
    
    def has_attr_as_determinant(self, lhs: str) -> bool:
        """ A function for checking if an attribute is part of the functional dependency's determinants.

        Args:
            lhs (str): The attribute to check for.

        Returns:
            bool: True if lhs is part of the functional dependency's determinants. False, otherwise.
        """
        return lhs in self.determinants
    
    
    def determinant_subset_cost(self, determinant_subset: list[str]) -> int:
        """ A function for computing the storage cost (in bits) of a subset of the determinant list.
        Useful when traversing AFDs, particularly when the determinants are a superset of determinant_subset.

        Args:
            determinant_subset (list[str]): The subset list of attributes to compute the storage cost of.

        Returns:
            int: The total cost of the attribute subset list.
        """
        subset_costs = list(map(lambda x: self.determinant_size_dict[x], determinant_subset))
        return reduce(lambda x, y: x + y, subset_costs)
    
    
    def __repr__(self):
        # Handle edge case if there are no determinants, somehow from other algorithms' results
        if not self.determinants:
            return ''
        
        result = '('
        # Used to keep track of last element for printing comma
        last_item = self.determinants[-1]
        for column_determinant in self.determinants:
            result += column_determinant
            
            # If multiple determinants, add comma and a space
            if len(self.determinants) > 1 and column_determinant != last_item:
                result += ', '
        result += f') --> ({self.dependant})'
        result += f'\nDeterminant: {self.determinant_size} bit(s), Dependant: {self.dependant_size} bit(s)\n'
        return result
    
    def __eq__(self, other):
        if other.__class__ is not self.__class__:
            return False
        if self.dependant != other.dependant:
            return False
        if len(self.determinants) != len(other.determinants):
            return False
        if set(self.determinants) != set(other.determinants):
            return False
        
        return True
    
    
    def __lt__(self, other):
        if other.__class__ is not self.__class__:
            return NotImplemented
        # Note that the negated values are intentional to support a min sort as specified by heapq
        return (-self.dependant_size, self.determinant_size, -len(self.determinants)) < (-other.dependant_size, other.determinant_size, -len(other.determinants))
    
    
def num_determinants(fd: FunctionalDependency):
        # Make it easy to access number of determinants which will be helpful later when sorting
        return len(fd.determinants)
    

def same_determinants(lhs: list[str], dependency_lst: list[FunctionalDependency]) -> list[FunctionalDependency]:
    return list(filter(lambda x: set(x.determinants) == set(lhs), dependency_lst))


def attr_as_determinant(lhs: str, dependency_lst: list[FunctionalDependency]) -> list[FunctionalDependency]:
    """ Retrieves any functional dependencies where the specified attribute, lhs, appears in the determinant.

    Args:
        lhs (str): The attribute to search for in the lhs of functional dependencies.
        dependency_lst (list[FunctionalDependency]): The list of functional dependencies to search.

    Returns:
        list[FunctionalDependency]: The list of functional dependencies that were retrieved with the lhs in the determinant attributes.
    """
    return list(filter(lambda x: x.has_attr_as_determinant(lhs), dependency_lst))


def remove_attr_as_determinant(lhs: str, dependency_lst: list[FunctionalDependency]) -> list[FunctionalDependency]:
    """ Removes any functional dependencies where the specified attribute, lhs, appears in the determinant.

    Args:
        lhs (str): The attribute to filter from the lhs of functional dependencies.
        dependency_lst (list[FunctionalDependency]): The list of functional dependencies to filter.

    Returns:
        list[FunctionalDependency]: The list of functional dependencies that were filtered to remove the lhs as a determinant.
    """
    return list(filter(lambda x: not x.has_attr_as_determinant(lhs), dependency_lst))


def attr_as_dependant(rhs: str, dependency_lst: list[FunctionalDependency]) -> list[FunctionalDependency]:
    """ Retrieves any functional dependencies where the specified attribute, rhs, appears as the dependant
    attribute.

    Args:
        rhs (str): The attribute to search for as the rhs of functional dependencies.
        dependency_lst (list[FunctionalDependency]): The list of functional dependencies to search.

    Returns:
        list[FunctionalDependency]: The list of functional dependencies that were retrieved with the rhs as the dependant attribute.
    """
    return list(filter(lambda x: x.dependant == rhs, dependency_lst))


def remove_attr_as_dependant(rhs: str, dependency_lst: list[dict]) -> list[dict]:
    """ Removes any functional dependencies where the specified attribute, rhs, appears as the dependant
    attribute. 

    Args:
        rhs (str): The attribute to filter from the rhs of functional dependencies.
        dependency_lst (list[dict]): The list of functional dependencies to filter.

    Returns:
        list[dict]: The list of functional dependencies that were filtered to remove the rhs as the dependant attribute.
    """
    return list(filter(lambda x: x.dependant != rhs, dependency_lst))
    

def load_fds(file_path, metadata_file_path: str) -> list[FunctionalDependency]:
    """ Given a file path from the Metronome CLI / Pyro algorithm, read the
    file and convert the dependencies to Python Functional Dependency instances.

    Args:
        file_path (Path): The path of the file from Metronome CLI to read. The path
        should use Python's pathlib

    Returns:
        list[FunctionalDependency]: The list of functional dependencies parsed from the file into
            Python Functional Dependency instances.
    """
    metadata_dict = get_column_info_table(metadata_file_path)
    fd_lst = []
    
    with open(file_path) as fd_file:
        for line in fd_file:
            try:
                data = FunctionalDependency(line, metadata_dict)
                fd_lst.append(data)
            except ValueError as err:
                # If an AFD is not approved, e.g., due to an empty list of determinants. Display that it was skipped and do not add the AFD
                print(str(err))
            
    return (fd_lst, metadata_dict)
    
def display_fds(dependency_lst: list[FunctionalDependency]):
    """ Given a list of functional dependencies, displays the dependencies to console.

    Args:
        dependency_lst (list[FunctionalDependency]): The list of Functional Dependencies to display
    """
    for fd in dependency_lst:
        print(f'{fd}')
               

def select_attr_mappings(afd_lst: list[FunctionalDependency], metadata_dict: dict, columns_to_exclude: list[str], use_transitives=True):
    """ Given a list of AFDs and a metadata dictionary of attributes and sizes, this function searches through the AFDs in a 
    greedy manner for AFDs to save the most storage costs (originally for the purpose of data postdiction). AFDs are placed into a min
    priority queue heap based on (dependant_size DESC, determinants_size ASC, number_of_determinants ASC), which effectively prioritizes
    the replacement of attributes that require more storage costs and AFDs that can replace the attributes using the least storage overhead.
    This is a greedy approach since the search will always prioritize AFDs with the most costly dependants first. The search also takes into
    consideration previously selected AFDs when computing additional storage overheads. The general idea of this policy is to "replace as many
    attributes as possible from the attributes that were already chosen to be preserved."

    Args:
        afd_lst (list[FunctionalDependency]): The list of approximate functional dependencies to search through
        metadata_dict (dict): The metadata dictionary of attributes and sizes
        columns_to_exclude (list[str]): A list of columns to exclude from the search (e.g., Index)

    Returns:
        tuple: A tuple (selected_afds, outlier_columns) where the first element is a list of AFDs that were selected during the search and the second
        element is a list of outlier columns that are not replaced nor attributes that are preserved for replacement.
        
    Raises:
        ValueError: If afd_lst is empty such as when the external algorithm for finding AFDs does not find any mappings with the provided error threshold.
    """
    # Edge Case: If afd_lst is empty as a result of the third-party algorithm (e.g., Pyro) not finding any AFDs with the given error threshold
    if not afd_lst:
        raise ValueError("No AFDs were provided to the search and were likely not detected by the external algorithm. Consider tuning the error threshold.")
    
    # Initialize attributes and AFDs to be considered
    attr_candidates: list[str] = [x for x in metadata_dict.keys() if x not in columns_to_exclude]
    selected_afds: list[FunctionalDependency] = []
    
    # Edge Case: If every column is provided as an exclusion, there are no candidate attributes to consider and the search should immediately return
    if not attr_candidates:
        return ([], columns_to_exclude)
    
    afd_pq = afd_lst.copy()
    for excluded_col in columns_to_exclude:
        afd_pq = remove_attr_as_determinant(excluded_col, afd_pq)
        afd_pq = remove_attr_as_dependant(excluded_col, afd_pq)
        
    # Convert list of AFDs to heap and perform min sort based on FunctionalDependency's __lt__ and __eq__ functions
    heapq.heapify(afd_pq)
    
    # While heap is not empty
    while afd_pq:
        # Retrieve current min afd and add to selected afds
        current_min_afd = heapq.heappop(afd_pq)
        selected_afds.append(current_min_afd)
        current_determinants = current_min_afd.determinants
        current_dependant = current_min_afd.dependant
        
        print(f'Current min AFD: {current_min_afd}')
        
        # Remove any other AFDs with dependent on either side of AFD. Also remove any AFDs with determinant as the dependant, since it has to be preserved. Then re-heapify
        if current_dependant in attr_candidates:
            attr_candidates.remove(current_dependant) # Remove since it is being replaced by the determinants
        
        if not use_transitives:
            afd_pq = remove_attr_as_determinant(current_dependant, afd_pq)
            
        afd_pq = remove_attr_as_dependant(current_dependant, afd_pq)
        for current_determinant in current_determinants:
            if current_determinant in attr_candidates:
                attr_candidates.remove(current_determinant) # Remove since it must be preserved for recovery
            afd_pq = remove_attr_as_dependant(current_determinant, afd_pq)
            
        heapq.heapify(afd_pq)
        
        # Check other AFDs with exact same determinant to boost storage savings from same overhead/preserved attributes. For each one, predict dependant using same determinants currently chosen
        matching_afds = same_determinants(current_determinants, afd_pq)
        rhs_to_explore = []
        rhs_to_explore.append(current_dependant)
        for matched_afd in matching_afds:
            # Add AFD as selected, remove it from the candidate attributes, and remove other AFDs with the dependant attribute. 
            selected_afds.append(matched_afd)
            if matched_afd.dependant in attr_candidates:
                attr_candidates.remove(matched_afd.dependant)
            rhs_to_explore.append(matched_afd.dependant)
            afd_pq = remove_attr_as_dependant(matched_afd.dependant, afd_pq)
            
            print(f'Marking the following AFD with same determinant: {matched_afd}')
            
            if not use_transitives:
                afd_pq = remove_attr_as_determinant(matched_afd.dependant, afd_pq)
        
        # Handle transitive AFDs by exploring discovered attributes
        if use_transitives:
            print(f'RHS attributes to explore for transitive AFDs {rhs_to_explore}')
            while rhs_to_explore:
                cur_explored_attr = rhs_to_explore.pop(0)
                print(f'Transitive attribute being looked at: {cur_explored_attr}')
                
                # Find AFDs with currently explored attribute on left-hand side, with only attributes that are new to discover
                cur_explored_afds = same_determinants([cur_explored_attr], afd_pq)
                
                cur_explored_afds = list(filter(lambda x: x.dependant in attr_candidates, cur_explored_afds)) # Filter the AFDs to only AFDs that recover new attributes (that were not preserved or already discovered)
                
                print(f'AFDs left to explore for transitive AFDs: {cur_explored_afds}')
                
                # For each of the remaining AFDs, create a new AFD from our original source determinants to the new dependant, add it to the selected list, remove other AFDs that predict the same dependant
                # Add the dependant to the rhs_to_explore, remove the dependant from the attr_candidates
                for new_explored_afd in cur_explored_afds:
                    print(f'New explored AFD: {new_explored_afd}')
                    current_determinant_str = list(map(lambda x: {'columnIdentifier': x}, current_determinants))
                    trans_dict = {'type': 'FunctionalDependency', 'determinant':{'columnIdentifiers': current_determinant_str}, 'dependant': {'columnIdentifier': new_explored_afd.dependant}}
                    trans_afd = FunctionalDependency(json.dumps(trans_dict), metadata_dict)
                    
                    selected_afds.append(trans_afd)
                    afd_pq = remove_attr_as_dependant(new_explored_afd.dependant, afd_pq)
                    rhs_to_explore.append(new_explored_afd.dependant)
                    if new_explored_afd.dependant in attr_candidates:
                        attr_candidates.remove(new_explored_afd.dependant)
                    print(f'Added transitive FD: {trans_afd}')
                
                # After considering all of the AFDs, remove any AFDs with the currently explored attribute as the lhs/determinant
                afd_pq = remove_attr_as_determinant(cur_explored_attr, afd_pq)
            
        # Reorganize the heap after matched and transitive exploration
        heapq.heapify(afd_pq)
        
        # Since we are preserving the determinants already, remove their cost from the remaining AFDs to focus on only the newly added overhead
        for remaining_afd in afd_pq:
            # Subtract the cost of each determinant contained within the afd to account for the attribute already being preserved
            for remaining_determinant in current_determinants:
                if remaining_afd.has_attr_as_determinant(remaining_determinant):
                    remaining_afd.determinant_size -= remaining_afd.determinant_size_dict[remaining_determinant]
            
        heapq.heapify(afd_pq)
        
    # Restore all of the AFD determinant sizes before returning the result
    for afd in selected_afds:
        afd.reset_determinant_sizes()
    return (selected_afds, attr_candidates)
        
        
def find_afds(dataset_path: Path, metadata_path: Path, cols_to_remove: list[str], fd_algorithm: str, max_fd_error: float, max_rows=10000000, find_uccs="false", utilize_transitive_afds=True):
    """ A wrapper function for finding afds that handles calling the external Java library for finding AFDs and calls the function to select/search for
    which AFDs to select. 

    Args:
        dataset_path (Path): The path to the input dataset similar to the example YAML config files
        metadataset_path (Path): The path to the metadata YAML file that provides the table's schema and the types/sizes of each attribute
        cols_to_remove (list[str]): A list of columns that should to exclude from the search (e.g., Index)
        fd_algorithm (str): The Metanome algorithm for discovering AFDs such as Pyro, TaneX, etc.
        max_fd_error (float): The error threshold utilized when discovering AFDs for the ratio of tuples that can violate a FD
        find_uccs (str, optional): Whether to discover and generate UCCs for the input dataset using the selected Metanome algorithm. Defaults to "false".
        utilize_transitive_afds (bool, optional): Whether to utilize transitive AFDs when selecting/searching AFDs. Defaults to True.

    Returns:
        tuple: A tuple (selected_afds, outlier_columns) where the first element is a list of AFDs that were selected during the search and the second
        element is a list of outlier columns that are not replaced nor attributes that are preserved for replacement.
        
    Raises:
        ValueError: If afd_lst is empty such as when the external algorithm for finding AFDs does not find any mappings with the provided error threshold.
    """
       
    data_name = f"{dataset_path.stem}-{fd_algorithm}-{str(max_fd_error)}"
    java_result_path = Path('.') / 'results' / f'{data_name}_fds'
    
    # Note/Beware in the below lookup that ADuccDfd only works for AFDs if AUCCs are also enabled. Otherwise, nothing is provided as output. For this reason, the setting to
    # disable UCCs is always overriden for ADuccDfd
    if fd_algorithm == 'ADuccDfd':
        find_uccs = "true"
    
    fd_algorithm_lookup = {
        'Pyro': {
            'mainclass': 'de.hpi.isg.pyro.algorithms.Pyro',
            'fdErrorParam': f'maxFdError:{max_fd_error}',
            'maxRows': f'maxRows:{max_rows}'
        },
        'TaneX': {
            'mainclass': 'de.hpi.isg.pyro.algorithms.TaneX',
            'fdErrorParam': f'maxFdError:{max_fd_error}',
            'maxRows': f'maxRows:{max_rows}'
        },
        'ADuccDfd': {
            'mainclass': 'de.hpi.isg.pyro.algorithms.ADuccDfd',
            'fdErrorParam': f'maxFdError:{max_fd_error}',
            'maxRows': f'maxRows:{max_rows}'
        },
        'FdepX': {
            'mainclass': 'de.hpi.isg.pyro.algorithms.FdepX',
            'fdErrorParam': f'maxFdError:{max_fd_error}',
            'maxRows': f'maxRows:{max_rows}'
        }
    }
    
    # Edge Case: If the provided algorithm is not currently implemented/supported in the algorithm lookup
    if fd_algorithm not in fd_algorithm_lookup:
        raise NotImplementedError(f"The {fd_algorithm} algorithm is currently not implemented as part of the Metanome library loaded into this project. Please use a supported algorithm as listed below:\n{list(fd_algorithm_lookup.keys())}")
    
    # Only run the jar if the functional dependency file does not already exist as this is an overhead
    if not java_result_path.exists():
        
        cmd = f"java -cp metanome-cli-1.1.0.jar:pyro-distro-1.0-SNAPSHOT-distro.jar de.metanome.cli.App --algorithm {fd_algorithm_lookup[fd_algorithm]['mainclass']} --files {dataset_path} --file-key inputFile --separator comma --header --algorithm-config {fd_algorithm_lookup[fd_algorithm]['fdErrorParam']} isFindKeys:{find_uccs} --output file:{data_name} > java-log.txt"
        try:
            print(f'Attempting to run the Java library using the following command:\n{cmd}\n')
            jar_output = subprocess.check_output(cmd, shell=True)
        except subprocess.CalledProcessError as shell_err:
            print('An error occurred while trying to run the jar libraries. Specifically:')
            print(f'{shell_err}')
            sys.exit(0)
            
    (attr_mappings, metadata_dict) = load_fds(java_result_path, str(metadata_path))    
    
    return select_attr_mappings(attr_mappings, metadata_dict, cols_to_remove, use_transitives=utilize_transitive_afds)


def sample_afds(dataset_path: Path, metadataset_path: Path, sampling_rounds=3, desired_num_rows=1000000, sampling_seed=17) -> tuple[list[FunctionalDependency], dict]:
    """ A function for determining AFDs across samples of a specified dataset. The function will perform the specified number of sampling rounds and will approximately output
    the specified number of rows. The sampling is done by taking a percentage of rows from each partition of the dataset in order provide a sample with some representation from the
    entire dataset. The chosen AFDs are chosen as the most common determinants that appeared across the most sampling rounds and arbitrary if all determinants appeared with the
    same frequency. By default, the existing partition/batch size of the parquet file will be used.

    Args:
        dataset_path (Path): The path to the dataset that is being sampled and searched for AFDs
        metadataset_path (Path): The path to the metadata YAML file that provides the table's schema and the types/sizes of each attribute
        sampling_rounds (int, optional): The number of rounds to sample the dataset for the specified number of rows. Defaults to 3.
        desired_num_rows (int, optional): The approximate size of each sample (in terms of rows). Defaults to 1000000.
        sampling_seed (int, optional): A seed for setting the sampling behavior for reproducibility and testing. Defaults to 17.

    Returns:
        tuple: A tuple (elected_afds, afd_votes) where the first element is a list of AFDs that were selected as the most frequent across sampling rounds and the second
        element is a dictionary representing the history of each sampling round with the determinants and corresponding dependant attributes.
    """
    dataset_directories = dataset_path.parent
    data_name = dataset_path.stem
    
    afd_votes = {}
    afd_global_counts = {}
    
    # Perform the sampling multiple times to be able to compute votes for AFDs
    for sample_round in range(1, sampling_rounds+1):
        # Only run this if the output file(s) don't exist yet
        sampled_csv_output_path = dataset_directories / f'{data_name}' / f'{data_name}-sample{sample_round}.csv'
        if not sampled_csv_output_path.exists():
        
            if dataset_path.suffix == '.csv':
                data = dd.read_csv(dataset_path, assume_missing=True)
            else:
                data = dd.read_parquet(dataset_path, index=False)  
            total_rows_in_data = len(data)
            approx_sample_percentage_per_partition = desired_num_rows / total_rows_in_data
            if approx_sample_percentage_per_partition > 1.0:
                approx_sample_percentage_per_partition = 1.0
            
            # Added preprocessing stage for sampling a dataset down to fit within the memory constraints of the Java Library (since it loads the entire dataset into memory)
            print(f'[Sample Round {sample_round}]: Sampling from all {data.npartitions} partitions, this may take several minutes')
            sampled_data = data.sample(frac=approx_sample_percentage_per_partition, random_state=sampling_seed)
            
            # Drop rows with null values from sample, else Java Library will crash
            print('Dropping Nulls')
            sampled_data = sampled_data.compute()
            sampled_data = sampled_data.dropna(axis=0, how='any')
            sampled_data = sampled_data.reset_index(drop=True)

            if not os.path.exists(sampled_csv_output_path):
                os.makedirs(os.path.dirname(sampled_csv_output_path), exist_ok=True)
            sampled_data.to_csv(sampled_csv_output_path, index=False)
            
            print(f'[Sample Round {sample_round}]: Finished sampling and outputting to a CSV... Passing the sampled dataset to the Pyro Java Library for AFDs...')
        else:
            print(f'[Sample Round {sample_round}]: Detected that the output CSV file already exists. Skipping sampling and proceeding to the Pyro Java Library for AFDs...')
        
        (chosen_afds, outlier_columns) = find_afds(sampled_csv_output_path, metadataset_path, ['Index'], 'Pyro', 0.05)
        
        print(f'\n[Sample Round {sample_round}]: Selected AFDs are:\n')
        display_fds(chosen_afds)
        
        print(f'\n[Sample Round {sample_round}]: Outlier Attributes:\n')
        print(str(outlier_columns))
        
        
        print(f'\n[Sample Round {sample_round}]: Testing without transitive AFDS')
        
        (chosen_afds, outlier_columns) = find_afds(sampled_csv_output_path, metadataset_path, ['Index'], 'Pyro', 0.05, utilize_transitive_afds=False)
        
        print(f'\n[Sample Round {sample_round}]: Selected AFDs are:\n')
        display_fds(chosen_afds)
        
        print(f'\n[Sample Round {sample_round}]: Outlier Attributes:\n')
        print(str(outlier_columns))      
        
        # Add Determinants from current sampling round to aggregate for final voting
        for afd in chosen_afds:
            afd.determinants.sort()
            cur_determinants = f'{afd.determinants}'[2:-2]
            
            if f'round{sample_round}' not in afd_votes:
                afd_votes[f'round{sample_round}'] = {}
                
            if cur_determinants in afd_votes[f'round{sample_round}']:
                # Don't increment the count since it's tracking the number of rounds an AFD appears
                if afd.dependant not in afd_votes[f'round{sample_round}'][cur_determinants]['dependants']:
                    afd_votes[f'round{sample_round}'][cur_determinants]['dependants'].append(afd.dependant)
            else:
                afd_votes[f'round{sample_round}'][cur_determinants] = {'dependants': [afd.dependant]}
                if cur_determinants in afd_global_counts:
                    afd_global_counts[cur_determinants] += 1
                else:
                    afd_global_counts[cur_determinants] = 1
        
    # Using the AFD Votes, chose the determinants with the highest counts
    # 1. Sort the determinants/keys based on their round counts
    afd_counts = list(afd_global_counts.keys())
    afd_counts.sort(key=lambda x: afd_global_counts[x], reverse=True)
    
    # 2. Select determinants based on the most common vote. If there is no popular candidate, the oldest candidate is picked... just like US politics (ironic for a project about decaying)
    elected_determinants = []
    metadata_dict = get_column_info_table(str(metadataset_path))
    
    while len(afd_counts) > 0:
        most_popular_determinant = afd_counts[0]
        
        # 3. Remove the dependants predicted by the determinant across the rounds
        for prior_round in range(1, sampling_rounds + 1):
            if most_popular_determinant in afd_votes[f'round{sample_round}']:
                for round_dependant in afd_votes[f'round{sample_round}'][most_popular_determinant]['dependants']:
                    if round_dependant in afd_counts:
                        afd_counts.remove(round_dependant)
                    
                    cur_afd = FunctionalDependency(None, metadata_dict, determinants=[most_popular_determinant], dependant=round_dependant)
                    if cur_afd not in elected_determinants:
                        elected_determinants.append(cur_afd)
                
        # 4. Remove the current most popular dependant from the list to avoid repeating the same step
        if most_popular_determinant in afd_counts:
            afd_counts.remove(most_popular_determinant)
        
    return elected_determinants, afd_votes


if __name__ == '__main__':
    relation_name = 'rel0'
    dataset_path = Path('.') / '..' / 'data-decay' / 'python_postdiction' / 'archive' / f'measurements_airquality.csv'
    metadataset_path = Path('.') / '..' / 'data-decay' / 'python_postdiction' / 'archive' / f'measurements_airquality.csv'
    
    rounds = 3
    num_rows = 1000000
    sampling_seed = 17
    
    (elected_afds, fd_sample_history) = sample_afds(dataset_path, metadataset_path, sampling_rounds=rounds, desired_num_rows=num_rows, sampling_seed=sampling_seed)
    
    print(f'Your winners of the current election: {elected_afds}')
    print(f'Their previous platform: {fd_sample_history}')
