import numpy as np
import graphviz
from IPython.display import display


class NaryTreeEnvironment:
    """
    N-ary Tree environment for RL with categorical actions and improved display
    """

    def __init__(self, depth, branching_factor, start=0):
        self.depth = depth
        self.branching_factor = branching_factor
        self.num_nodes = sum(branching_factor**i for i in range(depth))
        self.agent_position = start
        self.action_map = {
            0: "up",
            **{i + 1: f"child_{i}" for i in range(branching_factor)},
            branching_factor + 1: "stay",
        }

    def move_agent(self, action):
        direction = self.action_map[action]
        current_node = self.agent_position

        if direction == "up":
            new_position = (
                (current_node - 1) // self.branching_factor if current_node > 0 else 0
            )
        elif direction.startswith("child_"):
            child_index = int(direction.split("_")[1])
            new_position = self.branching_factor * current_node + child_index + 1
        elif direction == "stay":
            new_position = current_node
        else:
            raise ValueError(
                f"Invalid action. Use 0 (up), 1-{self.branching_factor} (children), or {self.branching_factor+1} (stay)."
            )

        if self.is_valid_move(new_position):
            self.agent_position = new_position
            return True
        return False

    def is_valid_move(self, position):
        return 0 <= position < self.num_nodes

    def get_state(self):
        return self.agent_position

    def is_leaf(self):
        return self.agent_position >= (
            self.num_nodes - self.branching_factor ** (self.depth - 1)
        ) // (self.branching_factor - 1)

    def reset(self):
        self.agent_position = 0
        return self.get_state()

    def get_possible_actions(self):
        actions = [self.branching_factor + 1]  # 'stay' is always possible
        if self.agent_position > 0:
            actions.append(0)  # 'up' is possible if not at root
        if not self.is_leaf():
            actions.extend(
                range(1, self.branching_factor + 1)
            )  # child moves are possible if not a leaf
        return actions

    def get_node_path(self, start_node, end_node):
        """
        Returns the path of nodes from start_node to end_node.
        """
        if start_node == end_node:
            return [start_node]

        path_to_root_start = []
        path_to_root_end = []

        # Trace path from start_node to root
        node = start_node
        while node >= 0:
            path_to_root_start.append(node)
            node = (node - 1) // self.branching_factor

        # Trace path from end_node to root
        node = end_node
        while node >= 0:
            path_to_root_end.append(node)
            node = (node - 1) // self.branching_factor

        # Find the common ancestor
        while (
            path_to_root_start
            and path_to_root_end
            and path_to_root_start[-1] == path_to_root_end[-1]
        ):
            common_ancestor = path_to_root_start.pop()
            path_to_root_end.pop()

        # Construct the full path
        return path_to_root_start + [common_ancestor] + path_to_root_end[::-1]

    def get_action_path(self, start_node, end_node):
        """
        Returns the sequence of actions to move from start_node to end_node.
        """
        node_path = self.get_node_path(start_node, end_node)
        actions = []

        for i in range(len(node_path) - 1):
            current_node = node_path[i]
            next_node = node_path[i + 1]

            if next_node < current_node:
                actions.append(0)  # Move up
            elif next_node == current_node:
                actions.append(self.branching_factor + 1)  # Stay
            else:
                action = next_node - self.branching_factor * current_node
                actions.append(action)  # Move to child

        actions.append(self.branching_factor + 1) # Stay
        return list(zip(node_path, actions))

    def display(self, highlight_path=None):
        dot = graphviz.Graph()  # Changed to Graph for undirected edges
        dot.attr(rankdir="TB")

        def add_nodes_edges(node, depth=0):
            if node >= self.num_nodes or depth >= self.depth:
                return

            # Node attributes
            if node == self.agent_position:
                dot.node(str(node), str(node), style="filled", color="green")
            elif highlight_path and node in highlight_path:
                dot.node(str(node), str(node), style="filled", color="lightblue")
            else:
                dot.node(str(node), str(node))

            # Add edges to children
            for i in range(self.branching_factor):
                child = self.branching_factor * node + i + 1
                if child < self.num_nodes and depth < self.depth - 1:
                    if (
                        highlight_path
                        and node in highlight_path
                        and child in highlight_path
                    ):
                        dot.edge(str(node), str(child), color="red", penwidth="2")
                    else:
                        dot.edge(str(node), str(child))
                    add_nodes_edges(child, depth + 1)

        add_nodes_edges(0)
        display(dot)


def main():
    """
    Usage.
    """
    # Create a 2-ary tree environment with depth 4
    env = NaryTreeEnvironment(depth=4, branching_factor=2)

    # Choose two nodes
    start_node = 12
    end_node = 6

    # Get the node path
    node_path = env.get_node_path(start_node, end_node)
    print(f"Node path from {start_node} to {end_node}: {node_path}")

    # Get the action path
    action_path = env.get_action_path(start_node, end_node)
    actions = [x[1] for x in action_path][:-1]
    print(actions)

    print(f"Action path from {start_node} to {end_node}: {action_path}")

    # Display the tree with highlighted path
    env.agent_position = start_node
    env.display(highlight_path=node_path)

    for action in actions:
        env.move_agent(action)
        print(f"After action {action}:")
        env.display(highlight_path=node_path)


if __name__ == "__main__":
    main()
