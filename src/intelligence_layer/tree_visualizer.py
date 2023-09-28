from typing import Optional
import uuid
import graphviz # type: ignore
from graphviz import Digraph
from intelligence_layer.classify import TreeNode

def add_node(graph: Digraph, node: TreeNode) -> str:
    new_node_id = str(uuid.uuid4())
    graph.node(new_node_id, label=f"{node.token.token if node.token else 'None'}:{str(round(node.prob, 3)) if node.prob else 'None'}")
    return new_node_id

def graph_path(graph: Digraph, node: TreeNode, parent_id: Optional[str] = None) -> None:
    for child in node.children:
        new_node_id = add_node(graph, child)
        graph.edge(parent_id, new_node_id)
        graph_path(graph, child, new_node_id) 

def graph_nodes(root: TreeNode) -> Digraph:
    graph = graphviz.Digraph('normalized probabilities', node_attr={'shape': 'plaintext'})
    graph.graph_attr['rankdir'] = 'TB'  
    graph.edge_attr.update(arrowhead='vee', arrowsize='1')
    for child in root.children:
        parent_id = add_node(graph, child)
        graph_path(graph, child, parent_id)
    return graph 