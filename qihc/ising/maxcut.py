import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def generate_random_graph(
    n,
    p,
    seed=None,
    node_color="lightblue",
    font_weight="bold",
    edge_color="gray",
    show=False,
):
    """
    Generate a random graph using the erdos_renyi_graph and display it.
    
    Parameters:
        n (int): Number of nodes in the graph.
        p (float): Probability of edge creation between any pair of nodes.
        seed (int, optional): Seed for random number generation for reproducibility.
        node_color (str or list, optional): Color for nodes (default is 'lightblue').
        font_weight (str, optional): Font weight for node labels (default is 'bold').
        edge_color (str, optional): Color for edges (default is 'gray').

    Returns:
        G (networkx.Graph): The generated random graph.
    """
    # Generate the random graph with the given number of nodes (n) and edge probability (p)
    G = nx.erdos_renyi_graph(n, p, seed=seed)
    
    # Visualize the graph
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G, seed=seed)  # Use spring layout for node positioning
    
    # Draw the nodes, edges, and labels
    nx.draw_networkx_nodes(G, pos, node_color=node_color, node_size=500)
    nx.draw_networkx_edges(G, pos, edge_color=edge_color, width=1.5)
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight=font_weight)
    
    # Display the plot with appropriate labels and title
    plt.title(f"Random Graph with {n} nodes and edge probability {p}", fontsize=14)
    plt.axis("off")  # Hide axes for better presentation
    if show:
        plt.show()
    else:
        plt.close()

    return G

def brute_force_max_cut(G):
    """
    Brute-force solution for the Max-Cut problem: Finds the cut that maximizes the number of edges 
    between two partitions of the graph. 

    Parameters:
        G (networkx.Graph): The graph on which to perform the Max-Cut.

    Returns:
        max_cut_value (int): The maximum number of edges between the two partitions.
        best_cut (tuple): A tuple of two sets representing the two partitions that maximize the cut.
    """
    nodes = list(G.nodes())
    n = len(nodes)
    max_cut_value = 0
    best_cut = None

    # Iterate over all 2^n possible subsets of nodes
    for i in range(1 << n):
        # Generate two sets based on the current subset
        set1 = {nodes[j] for j in range(n) if (i & (1 << j))}  # Nodes in the current subset
        set2 = set(nodes) - set1  # Remaining nodes form the second set
        
        # Calculate the cut size: number of edges between set1 and set2
        cut_size = sum(1 for u, v in G.edges() if (u in set1 and v in set2) or (u in set2 and v in set1))
        
        # Update if this cut is better than the previous maximum cut
        if cut_size > max_cut_value:
            max_cut_value = cut_size
            best_cut = (set1, set2)

    return max_cut_value, best_cut

def brute_force_max_cut_balanced(G):
    """
    Brute-force algorithm to solve the max-cut problem with a balanced partition constraint.
    The graph must have an even number of nodes to split into two equal parts.

    Parameters:
        G (networkx.Graph): The graph to solve the max-cut problem on.

    Returns:
        max_cut_value (int): The maximum number of edges crossing the partition.
        best_partition (tuple): A tuple of two sets representing the balanced partitions.
    """
    nodes = list(G.nodes())
    num_nodes = len(nodes)

    # Ensure the graph has an even number of nodes
    if num_nodes % 2 != 0:
        raise ValueError("Graph has an odd number of nodes, cannot split into two equal parts.")
    
    half_size = num_nodes // 2  # Each subset should have half of the nodes
    max_cut_value = 0
    best_partition = None
    
    # Generate all combinations of nodes that form exactly half of the graph
    for subset in itertools.combinations(nodes, half_size):
        # Convert the combination to a set for efficient lookup
        subset_set = set(subset)
        other_subset = set(nodes) - subset_set  # Remaining nodes form the second set
        
        # Calculate the cut value: sum of edge weights between the two subsets
        cut_value = 0
        for u, v in G.edges():
            if (u in subset_set and v in other_subset) or (v in subset_set and u in other_subset):
                # Add the edge weight, defaulting to 1 if not specified
                cut_value += G[u][v].get('weight', 1)
        
        # Update the max cut value and best partition if this cut is better
        if cut_value > max_cut_value:
            max_cut_value = cut_value
            best_partition = (subset_set, other_subset)
    
    return max_cut_value, best_partition

def graph_to_adjacency_matrix(G):
    """
    Converts a networkx graph to an adjacency matrix. For unweighted graphs, it assigns a weight of -1 to edges.
    
    Parameters:
        G (networkx.Graph): The graph to convert to an adjacency matrix.

    Returns:
        numpy.ndarray: The adjacency matrix.
    """
    # Get the list of nodes
    nodes = list(G.nodes())
    n = len(nodes)
    
    # Create a mapping from node labels to matrix indices
    node_index = {node: idx for idx, node in enumerate(nodes)}
    
    # Initialize the adjacency matrix with zeros
    A = np.zeros((n, n))
    
    # Fill the adjacency matrix with edge weights (using -1 for unweighted edges)
    for u, v in G.edges():
        # Use the index of the nodes to access the correct position in the matrix
        i, j = node_index[u], node_index[v]
        weight = G[u][v].get('weight', -1)
        A[i, j] = weight
        A[j, i] = weight 
    
    return A

def max_cut_to_ising(G):
    """
    Converts a max-cut problem to an Ising model interaction matrix (J).
    
    For the max-cut problem, edges are represented with negative interactions in the Ising model.
    This function converts the edges to the Ising model interaction matrix (J) where each edge
    between nodes corresponds to a negative interaction term.

    Parameters:
        G (networkx.Graph): The graph to convert to an Ising model.
    
    Returns:
        J (dict): The interaction matrix for the Ising model, where each key is a tuple (u, v) and value is the interaction term.
    """
    J = {}
    
    # Convert the edges into the Ising model's interaction matrix (J)
    for u, v in G.edges():
        # For max-cut, interaction strength is negative to minimize energy when the nodes are cut.
        # We also use the edge weight if available, defaulting to -1.
        weight = G[u][v].get('weight', 1)  # Default to weight of 1 if no weight is specified
        J[(u, v)] = -weight  # Negative interaction for max-cut
        
        # Since the Ising model is symmetric (J[i, j] = J[j, i]), ensure the symmetry
        J[(v, u)] = -weight  # Ensure symmetry for undirected graphs
    
    return J

def ising_simulated_annealing(J, steps=1000, T_start=10, T_end=0.1):
    """
    Performs simulated annealing to solve an Ising model given an interaction matrix J.
    The goal is to find the configuration of spins that minimizes the energy.
    
    Parameters:
        J (dict): The interaction matrix where the keys are tuples of nodes (i, j) and the values are interaction terms.
        steps (int): Number of simulated annealing steps.
        T_start (float): Starting temperature.
        T_end (float): Ending temperature.

    Returns:
        spins, current_energy_list, temperature_list
    """
    # Initialize nodes from the interaction matrix (edges of the graph)
    nodes = list(set([i for edge in J for i in edge]))
    
    # Randomly initialize the spins for each node
    spins = {node: 1 if np.random.rand() > 0.5 else -1 for node in nodes}
    
    current_energy = -sum(J[(i, j)] * spins[i] * spins[j] for (i, j) in J)  # Initial energy
    T = T_start  # Starting temperature

    # data collection for analysis (optional)
    current_energy_list, temperature_list = [current_energy], [T_start]
    
    # Simulated Annealing process
    for step in range(steps):
        # Update the temperature according to the annealing schedule
        T = T_start * (T_end / T_start) ** (step / steps)
        # another way to update temperature: T*= alpha  (alpha: damping factor)
        
        # Randomly select a node and flip its spin
        i = np.random.choice(nodes)
        spins[i] *= -1
        
        # Calculate new energy after flipping the spin
        new_energy = -sum(J[(i, j)] * spins[i] * spins[j] for (i, j) in J)
        # print(new_energy)
        
        # Metropolis acceptance criterion
        if new_energy < current_energy or np.random.rand() < np.exp((current_energy - new_energy) / T):
            current_energy = new_energy  # Accept the new configuration
            # print(current_energy)
        else:
            spins[i] *= -1  # Revert the spin flip if not accepted

        # data collection for analysis (optional)
        current_energy_list.append(current_energy)
        temperature_list.append(T)
    
    return spins, current_energy_list, temperature_list  # Return spins and the negative of the energy

def max_cut_to_ising_with_penalty(G, p):
    """
    Converts a max-cut problem to an Ising model with a size balance penalty.
    
    Parameters:
        G (networkx.Graph): The input graph.
        p (float): Penalty parameter for size imbalance. Higher values impose a stronger penalty.
    
    Returns:
        J (dict): The interaction matrix for the Ising model, where each key is a tuple (u, v) and value is the interaction strength.
        h (dict): The external field for the Ising model, where each key is a node and value is the external field strength.
    """
    J = {}
    h = {}
    
    # Set interaction coefficients for edges (use edge weight if available, default to -1)
    for u, v in G.edges():
        weight = G[u][v].get('weight', -1)  # Default to -1 if no weight is specified
        J[(u, v)] = weight  # Negative because we want to maximize the cut
        J[(v, u)] = weight  # Ensure symmetry for undirected graph
    
    # Set external field to penalize size imbalance
    # The external field is a bias to encourage balanced partition sizes
    for node in G.nodes():
        h[node] = 2 * p  # External field, penalizing size imbalance
    
    return J, h

def ising_simulated_annealing_with_penalty(J, h, nodes, steps=1000, T_start=10, T_end=0.1):
    """
    Performs simulated annealing to solve the Ising model with a size balance penalty.
    
    Parameters:
        J (dict): Interaction matrix where the keys are (i, j) pairs of nodes and the values are interaction strengths.
        h (dict): External field dict, where each key is a node and each value is the field strength.
        nodes (list): List of nodes in the graph.
        steps (int): Number of annealing steps.
        T_start (float): Initial temperature.
        T_end (float): Final temperature.
    
    Returns:
        tuple: Final spin configuration (spins) and the minimized energy.
    """
    # Initialize spins randomly
    spins = {node: 1 if np.random.rand() > 0.5 else -1 for node in nodes}
    
    cut_energy = -sum(J[(i, j)] * spins[i] * spins[j] for (i, j) in J)
    field_energy = -sum(h[i] * spins[i] for i in nodes)
    current_energy = cut_energy + field_energy
    T = T_start
    
    for step in range(steps):
        # Decrease temperature according to annealing schedule
        T = T_start * (T_end / T_start) ** (step / steps)
        
        # Randomly select a node and flip its spin
        i = np.random.choice(nodes)
        spins[i] *= -1
        
        # Calculate the change in energy due to the spin flip
        # Efficient way to calculate energy change
        cut_energy_diff = -sum(J.get((i, j), 0) * spins[i] * spins[j] for j in nodes if (i, j) in J)
        field_energy_diff = -h[i] * spins[i] * 2  # The field energy change for flip
        
        # Total energy change
        new_energy = cut_energy_diff + field_energy_diff
        # Metropolis acceptance criterion
        if new_energy < 0 or np.random.rand() < np.exp((current_energy - new_energy) / T):
            current_energy = new_energy  # Accept the new configuration
            spins[i] *= -1  # Revert the spin flip if not accepted
    
    return spins, current_energy  # Return final spins and minimized energy

def convert_spins_to_cut(spins):
    set1 = {node for node, spin in spins.items() if spin == 1}
    set2 = {node for node, spin in spins.items() if spin == -1}
    return set1, set2

def calculate_cut_value(G, cut):
    set1, set2 = cut
    return sum(1 for u, v in G.edges() if (u in set1 and v in set2) or (u in set2 and v in set1))

def plot_partition(graph, cut):
    ''' 
    Parameters:
        graph (networkx.Graph): The graph to plot.
        cut (tuple): A partition of the nodes of the graph.
    '''
    
    (set1, set2) = cut
    cut = {node: 1 if node in set1 else 0 for node in graph.nodes()}
    edges = [(u, v) for (u, v) in graph.edges() if cut[u] != cut[v]]
    # edges_in_cut = [(u, v) for (u, v) in graph.edges() if cut[u] == cut[v]]
    edges_in_cut_1 = [(u, v) for (u, v) in graph.edges() if u in set1 and v in set1]
    edges_in_cut_2 = [(u, v) for (u, v) in graph.edges() if u in set2 and v in set2]
    
    pos = nx.spring_layout(graph)
    nx.draw_networkx_edges(graph, pos, edgelist=edges, edge_color='red')
    # nx.draw_networkx_edges(graph, pos, edgelist=edges_in_cut, edge_color='blue')
    nx.draw_networkx_edges(graph, pos, edgelist=edges_in_cut_1, edge_color='orange')
    nx.draw_networkx_edges(graph, pos, edgelist=edges_in_cut_2, edge_color='green')
    nx.draw_networkx_nodes(graph, pos, nodelist=set1, node_color='orange')
    nx.draw_networkx_nodes(graph, pos, nodelist=set2, node_color='green')
    nx.draw_networkx_labels(graph, pos)
    plt.show()
