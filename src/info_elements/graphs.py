# -*- coding: utf-8 -*-
from pyvis.network import Network
import networkx as nx

def inter_graph(main_connections, path_storey, title_graph, page_title):
    """
    Visualizes the connections between rooms in a building as a graph, 
    with interactive features.

    Parameters
    ----------
    main_connections : list of tuples
        A list of tuples where each tuple represents a connection between two
        rooms.
    title_graph : str
        The title for the graph visualization.
    page_title : str
        The title that will be displayed at the top of the HTML page above 
        the graph.

    Returns
    -------
    None.

    """
    l = []
    for u, v in main_connections:
        l.append(u)
        l.append(v)
    
    number_rooms = list(set(l))

    G = nx.Graph()
    
    for d_n in range(max(number_rooms)):
        room_name = f"Room {d_n + 1}"  
        G.add_node(room_name, main=True)  
        
    for room1, room2 in main_connections:
        node1 = f"Room {room1}"
        node2 = f"Room {room2}"
        G.add_edge(node1, node2)  
    
    net = Network(height="750px", width="100%", notebook=False, 
                  directed=False)
    net.from_nx(G)
    
    for node in net.nodes:
        if G.nodes[node['id']].get('main', False):  
            node['size'] = 50 
            node['color'] = "red"  
            node['font'] = {'size': 40, 'color': 'black', 'bold': True}  
            node['label'] = node['id']  
        
    for edge in net.edges:
        node1, node2 = edge['from'], edge['to']
        if G.nodes[node1].get('main', False) and G.nodes[node2].get('main', 
                                                                    False):
            edge['color'] = 'black'  
            edge['width'] = 2
        
    net.repulsion(
        node_distance=150,  
        central_gravity=0.2,  
        spring_length=100,  
        spring_strength=0.05  
    )
    
    net.set_options(""" 
    var options = {
      "physics": {
        "forceAtlas2Based": {
          "gravitationalConstant": -50,
          "centralGravity": 0.005,
          "springLength": 200,
          "springConstant": 0.08
        },
        "minVelocity": 0.75,
        "solver": "forceAtlas2Based"
      }
    }
    """)
    
    net.show(path_storey + '\\' + title_graph)
    
    with open(path_storey + '\\' + title_graph, "r", encoding="utf-8") as f:
        html_content = f.read()
    
    html_content = html_content.replace(
        "<body>",
        f'<body>\n'
        f'<h1 style="text-align: center; font-size: 36px; margin-top: 20px;">'
        f'{page_title}</h1>'
    )
    
    with open(path_storey + '\\' + title_graph, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    
def intra_graph(intra_adjacency, lines, path_storey, title_graph, page_title):
    """
    Visualizes the adjacency relationships between rooms and their walls as 
    an interactive graph.


    Parameters
    ----------
    intra_adjacency : list of lists
        A list of lists, where each sublist contains pairs of wallsthat are 
        adjacent to each other within the same room.
        These pairs represent connections between walls in the same room.
    lines : list of lists
        A list of lists, where each sublist corresponds to a room and contains 
        the indices of the walls within that room.
    title_graph : str
        The title for the graph visualization, which is also used as the 
        filename when saving the graph as an HTML file.
    page_title : str
        The title that will be displayed at the top of the HTML page above 
        the graph.

    Returns
    -------
    None.

    """
    l = []
    for e_i, i in enumerate(intra_adjacency):
        l_s = []
        for e_j, j in enumerate(i):
            l_s.append(j)       
        unique_numbers = set()
        for sublist in l_s:
            unique_numbers.update(sublist)
        unique_numbers = list(unique_numbers)
        l.append(unique_numbers)
    
    k = []
    for e_i, i in enumerate(lines):
        l_s = []
        for e_j, j in enumerate(i):
            l_s.append(e_j)
        k.append(l_s)
    
    net = Network(height="750px", width="100%", notebook=False, directed=False)
    
    for i in range(len(l)):
        net.add_node(i, label=f"Room {i + 1}", color="red", size=25, 
                     font=dict(size=30))
    
    for main_node, subnodes in enumerate(k):
        for subnode in subnodes:
            subnode_id = f'{main_node}-{subnode}' 
            net.add_node(subnode_id, label=f'R{main_node+1}-W{subnode}', 
                         color="lightblue", size=20, font=dict(size=20))
            net.add_edge(main_node, subnode_id)  
    
    for main_node, sub_adjacency in enumerate(intra_adjacency):
        for edge in sub_adjacency:
            subnode1 = f'{main_node}-{edge[0]}'
            subnode2 = f'{main_node}-{edge[1]}'
            net.add_edge(subnode1, subnode2, color="black")
    
    net.show(path_storey + '\\' + title_graph)
    
    with open(path_storey + '\\' + title_graph, "r", encoding="utf-8") as f:
        html_content = f.read()
    

    html_content = html_content.replace(
        "<body>",
        f'<body>\n'
        f'<h1 style="text-align: center; font-size: 36px; margin-top: 20px;">'
        f'{page_title}</h1>'
    )
    
    with open(path_storey + '\\' + title_graph, "w", encoding="utf-8") as f:
        f.write(html_content)


def graph_inter_adjacency(main_connections, sub_connections, lines, path_storey,
                          title_graph, page_title):
    """
    Visualizes the interconnections between rooms and their walls, using both 
    main connections and sub-connections.

    Parameters
    ----------
    main_connections : list of tuples
        A list of tuples representing the main connections between rooms.
    sub_connections : list of tuples
        A list of tuples representing the sub-connections between walls of 
        different rooms. 
        - room1 and room2 are the room numbers,
        - wall1 and wall2 are the respective wall identifiers in the respective
        rooms.
    lines : list of lists
        A list of lists, where each sublist represents a room and contains the 
        wall identifiers within that room.
    title_graph : str
        The title of the graph, which will also be used as the filename to save
        the graph as an HTML file.
    page_title : str
        The title that will be displayed at the top of the HTML page above the 
        graph.

    Returns
    -------
    None.

    """
    l = []
    for e_i, i in enumerate(lines):
        l_s = []
        for e_j, j in enumerate(i):
            l_s.append(e_j)
        l.append(l_s)
    
    G = nx.Graph()
    
    for d_n, subnodes in enumerate(l):
        room_name = f"R{d_n + 1}"  
        G.add_node(room_name, main=True)  
        
        for subnode in subnodes:
            subnode_name = f"{room_name} - W{subnode}"  
            G.add_node(subnode_name, main=False)    
            G.add_edge(room_name, subnode_name)    
        
    for room1, room2 in main_connections:
        node1 = f"R{room1}"
        node2 = f"R{room2}"
        G.add_edge(node1, node2)  
    
    for n1, sn1, n2, sn2 in sub_connections:
        node1 = f"R{n1}" 
        node2 = f"R{n2}"  
        subnode1 = f"{node1} - W{sn1}" 
        subnode2 = f"{node2} - W{sn2}" 
        G.add_edge(subnode1, subnode2)  
    
    net = Network(height="750px", width="100%", notebook=False, directed=False)
    net.from_nx(G)
    
    for node in net.nodes:
        if G.nodes[node['id']].get('main', False):  
            node['size'] = 50  
            node['color'] = "red" 
            node['font'] = {'size': 50, 'color': 'black', 'bold': True}  
            node['label'] = node['id'] 
        else:  # Subnodos
            node['size'] = 40  
            node['color'] = "lightblue" 
            node['font'] = {'size': 40, 'color': 'black', 'bold': True}  
            node['label'] = node['id'] 
    
    for edge in net.edges:
        node1, node2 = edge['from'], edge['to']
        if G.nodes[node1].get('main', False) and G.nodes[node2].get('main', 
                                                                    False):
            edge['color'] = 'red'  
        else:
            edge['color'] = 'black' 
    
    net.repulsion(
        node_distance=150,  
        central_gravity=0.2,  
        spring_length=100,  
        spring_strength=0.05  
    )
    
    net.set_options(""" 
    var options = {
      "physics": {
        "forceAtlas2Based": {
          "gravitationalConstant": -50,
          "centralGravity": 0.005,
          "springLength": 200,
          "springConstant": 0.08
        },
        "minVelocity": 0.75,
        "solver": "forceAtlas2Based"
      }
    }
    """)
    
    net.show(path_storey + '\\' + title_graph)
    
    with open(path_storey + '\\' + title_graph, "r", encoding="utf-8") as f:
        html_content = f.read()
    
    html_content = html_content.replace(
        "<body>",
        f'<body>\n'
        f'<h1 style="text-align: center; font-size: 36px; margin-top: 20px;">'
        f'{page_title}</h1>'
    )
    
    with open(path_storey + '\\' + title_graph, "w", encoding="utf-8") as f:
        f.write(html_content)

