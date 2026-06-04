import networkx as nx

# Your dictionary data
data = {
    "folders": [
        {
            "id": "731195f5-5b98-4364-a269-039568a80613",
            "name": "test",
            "folders": [
                {
                    "id": "c0d3132f-c3b5-4a23-a065-5460efacb88b",
                    "name": "123123",
                    "color": "#93C5FD"
                },
                {
                    "id": "6dd846a0-f454-4541-a13b-be9583b1ac01",
                    "name": "test (1)",
                    "folders": [
                        {
                            "id": "629f5f72-e377-4487-8d03-f6b75e5cd3cc",
                            "name": "asd",
                            "color": "#A5B4FC"
                        }
                    ],
                    "color": "#93C5FD"
                }
            ]
        }
    ]
}


def parse_folders(folder_list, parent_id=None, G=None):

    for folder in folder_list:
        node_id = folder["id"]
        
        # Extract attributes, default to None if missing
        node_attrs = {
            "name": folder.get("name"),
            "color": folder.get("color")
        }
        
        # Add node with metadata
        G.add_node(node_id, **node_attrs)
        
        # Add edge from parent to current folder
        if parent_id:
            G.add_edge(parent_id, node_id)
            
        # Recursively parse nested folders
        if "folders" in folder:
            parse_folders(folder["folders"], parent_id=node_id, G=G)


# Assuming 'G' is the graph initialized and populated from the previous step
def get_folder_path(graph, target_id, return_names=True):
    """
    Finds the path from the root folder to the target ID.
    Returns a list of IDs or a list of folder names.
    """
    if target_id not in graph:
        return target_id
        # return f"ID {target_id} not found in the graph."
    
    path = [target_id]
    current = target_id
    
    # Traverse backwards using predecessors until we hit a root node
    while True:
        predecessors = list(graph.predecessors(current))
        if not predecessors:
            break
        current = predecessors[0]
        path.append(current)
        
    # Reverse to get the path from root -> target
    path.reverse()
    
    if return_names:
        # Convert IDs to their human-readable folder names
        return [graph.nodes[node_id].get("name") for node_id in path]
        
    return path


if __name__ == "__main__":

    # Initialize a directed graph
    G = nx.DiGraph()

    # Run the parser
    parse_folders(data["folders"], G=G)

    # Verify the graph structure
    print("Nodes:", G.nodes(data=True))
    print("Edges:", G.edges())


    # --- Example Usage ---
    # target_node = "629f5f72-e377-4487-8d03-f6b75e5cd3cc"
    target_node = "629f5f72-e377-4487-8d03-f6b75e5cd3ccd"
    print("Name Path:", get_folder_path(G, target_node, return_names=True))
    print("ID Path:", get_folder_path(G, target_node, return_names=False))
