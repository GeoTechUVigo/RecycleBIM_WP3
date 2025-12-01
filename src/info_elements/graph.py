# -*- coding: utf-8 -*-
import networkx as nx
import matplotlib.pyplot as plt
from adjustText import adjust_text


def graph(class_pcd_df, connection_rooms, tit):
    """
    Visualizes the connectivity between rooms in a building using a graph representation.


    Parameters
    ----------
    class_pcd_df : pandas.DataFrame
        A DataFrame containing point cloud data for the building. T
    connection_rooms : list of tuples
        A list of tuples representing the connections (edges) between rooms. 
    tit : str
        The title to be displayed on the graph.

    Returns
    -------
    None.

    """
    # Remove duplicates
    edges = [tuple(point) for point in set(tuple(x) for x in connection_rooms)]
    # max_node = max(max(edge) for edge in edges)
    max_node = class_pcd_df['id_room'].unique().astype(int).max()
    # Create an undirected graph
    G = nx.Graph()
    
    # Add nodes to the graph
    G.add_nodes_from(range(1, max_node + 1))
    
    # Adding edges to the graph
    G.add_edges_from(edges)
    
    # Positioning of nodes, adjust k to avoid overlaps
    pos = nx.spring_layout(G, k=0.6, iterations=100)
    
    plt.figure(figsize=(14, 14))  
    
    # Draw nodes and labels first
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=1500)
    nx.draw_networkx_edges(G, pos, edge_color='gray')
    
    # Draw labels and adjust to avoid overlaps
    texts = [plt.text(pos[n][0], pos[n][1], s=n, fontsize=20, ha='center', va='center') for n in G.nodes()]
    
    # Add title
    plt.title(tit, fontsize=24)
    
    # Show the graph
    plt.show()

