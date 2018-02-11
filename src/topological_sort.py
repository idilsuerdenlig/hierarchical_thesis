import networkx as nx
from model_placeholder import PlaceHolder
from mux_block import MuxBlock
from block import Block


def topological_sort(blocks):
    G = nx.DiGraph()

    for block in blocks:
        if not isinstance(block, PlaceHolder):
            G.add_node(block)

    for block in list(G):
        if isinstance(block, MuxBlock):
            for blocklist in block.block_lists:
                for oneblock in blocklist:
                    for ins in oneblock.input_connections:
                        if not isinstance(ins, PlaceHolder):
                            G.add_edge(block, ins)
                    if oneblock.reward_connection is not None and not isinstance(oneblock.reward_connection, PlaceHolder):
                        G.add_edge(block, oneblock.reward_connection)
        if block.input_connections is not None:
            for ins in block.input_connections:
                if not isinstance(ins, PlaceHolder):
                    G.add_edge(block, ins)
        if block.reward_connection is not None and not isinstance(block.reward_connection, PlaceHolder):
            G.add_edge(block, block.reward_connection)

    ordered_block_list = list(reversed(list(nx.topological_sort(G))))

    placeholders = list()
    for block in blocks:
        if isinstance(block, PlaceHolder):
            placeholders.append(block)

    return placeholders+ordered_block_list



