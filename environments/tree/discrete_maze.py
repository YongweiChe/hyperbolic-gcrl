import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
import random

class GridMazeEnvironment:
    def __init__(self, maze):
        self.maze = np.array(maze)
        self.height, self.width = self.maze.shape
        self.num_states = self.height * self.width
        self.valid_indices = np.argwhere(self.maze != 1)
        self.agent_position = self.get_random_start()
        self.action_map = {
            0: "left",
            1: "right",
            2: "up",
            3: "down",
            4: "stay"
        }


    def get_random_start(self):
      x, y = self.valid_indices[np.random.randint(0, self.valid_indices.shape[0])]
      # print(f'i: {x}, {y}')
      return self.flatten_state((x, y))
    
    def get_unflattened_valid_indices(self):
        return [self.flatten_state(tuple(s)) for s in self.valid_indices]
    
    def move_agent(self, action):
        direction = self.action_map[action]
        current_pos = self.unflatten_state(self.agent_position)

        if direction == "left":
            new_position = (current_pos[0], current_pos[1] - 1)
        elif direction == "right":
            new_position = (current_pos[0], current_pos[1] + 1)
        elif direction == "up":
            new_position = (current_pos[0] - 1, current_pos[1])
        elif direction == "down":
            new_position = (current_pos[0] + 1, current_pos[1])
        elif direction == "stay":
            new_position = current_pos
        else:
            raise ValueError(f"Invalid action. Use 0 (left), 1 (right), 2 (up), 3 (down), or 4 (stay).")

        if self.is_valid_move(new_position):
            self.agent_position = self.flatten_state(new_position)
            return True
        return False

    def is_valid_move(self, position):
        row, col = position
        return (0 <= row < self.height and 0 <= col < self.width and self.maze[row, col] != 1)

    def get_state(self):
        return self.agent_position

    def is_goal(self):
        return self.maze[self.unflatten_state(self.agent_position)] == 3

    def reset(self):
        self.agent_position = self.get_random_start()
        return self.get_state()

    def get_possible_actions(self):
        actions = [4]  # 'stay' is always possible
        row, col = self.unflatten_state(self.agent_position)
        if col > 0 and self.maze[row, col-1] != 1:
            actions.append(0)  # left
        if col < self.width-1 and self.maze[row, col+1] != 1:
            actions.append(1)  # right
        if row > 0 and self.maze[row-1, col] != 1:
            actions.append(2)  # up
        if row < self.height-1 and self.maze[row+1, col] != 1:
            actions.append(3)  # down
        return actions

    def flatten_state(self, state):
        return state[0] * self.width + state[1]

    def unflatten_state(self, flat_state):
        return divmod(flat_state, self.width)

    def get_path(self, start, end):
        """
        Returns the path from start to end using BFS.
        """
        queue = [(start, [start])]
        visited = set()

        while queue:
            (node, path) = queue.pop(0)
            if node not in visited:
                if node == end:
                    return path
                visited.add(node)
                for action in range(4):  # Exclude 'stay' action
                    row, col = self.unflatten_state(node)
                    if action == 0 and col > 0 and self.maze[row, col-1] != 1:  # left
                        new_state = self.flatten_state((row, col-1))
                        queue.append((new_state, path + [new_state]))
                    elif action == 1 and col < self.width-1 and self.maze[row, col+1] != 1:  # right
                        new_state = self.flatten_state((row, col+1))
                        queue.append((new_state, path + [new_state]))
                    elif action == 2 and row > 0 and self.maze[row-1, col] != 1:  # up
                        new_state = self.flatten_state((row-1, col))
                        queue.append((new_state, path + [new_state]))
                    elif action == 3 and row < self.height-1 and self.maze[row+1, col] != 1:  # down
                        new_state = self.flatten_state((row+1, col))
                        queue.append((new_state, path + [new_state]))

        return None  # No path found

    def get_action_path(self, start, end):
        """
        Returns the sequence of actions to move from start to end.
        """
        path = self.get_path(start, end)
        if not path:
            return None

        actions = []
        for i in range(len(path) - 1):
            current = self.unflatten_state(path[i])
            next_pos = self.unflatten_state(path[i + 1])
            if next_pos[1] < current[1]:
                actions.append(0)  # left
            elif next_pos[1] > current[1]:
                actions.append(1)  # right
            elif next_pos[0] < current[0]:
                actions.append(2)  # up
            elif next_pos[0] > current[0]:
                actions.append(3)  # down

        return list(zip(path, actions + [4]))  # Add 'stay' action at the end

    def display(self, highlight_path=None):
        plt.figure(figsize=(10, 10))
        plt.imshow(self.maze, cmap='binary')
        plt.title("Grid Maze Environment")
        plt.axis('off')

        # Highlight the path
        if highlight_path:
            path_y, path_x = zip(*[self.unflatten_state(state) for state in highlight_path])
            plt.plot(path_x, path_y, color='blue', linewidth=3, alpha=0.7)

        # Mark agent position
        agent_y, agent_x = self.unflatten_state(self.agent_position)
        print(f'agent: {agent_y}, {agent_x}')
        plt.plot(agent_x, agent_y, 'ro', markersize=15, alpha=0.7)

        display(plt.gcf())
        clear_output(wait=True)
        plt.close()

def get_trajectories(maze_env, num_trajectories):
    trajectories = []

    for _ in range(num_trajectories):
        # Get all valid positions (non-wall cells)
        valid_positions = [maze_env.flatten_state(pos) for pos in zip(*np.where(maze_env.maze != 1))]
        
        # Randomly choose start and end positions
        start_pos = random.choice(valid_positions)
        end_pos = random.choice(valid_positions)
        
        # Get the action path between these positions
        action_path = maze_env.get_action_path(start_pos, end_pos)
        
        # If a valid path exists, add it to the trajectories
        if action_path:
            trajectories.append(action_path)
        
    return trajectories

def main():
    # Create a sample maze
    # 0: empty, 1: wall, 2: start, 3: goal
    maze = np.array([
        [1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 0, 1, 0, 1],
        [1, 0, 0, 0, 1, 0, 1],
        [1, 0, 1, 0, 1, 0, 1],
        [1, 0, 1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1]
    ])

    env = GridMazeEnvironment(maze)

    # Generate trajectories
    trajectories = get_trajectories(env, num_trajectories=5)

    print(f"Generated {len(trajectories)} trajectories")
    for i, traj in enumerate(trajectories):
        print(f"Trajectory {i+1}: {traj}")

    # Display the maze with the first trajectory
    if trajectories:
        env.display(highlight_path=[state for state, _ in trajectories[0]])

if __name__ == "__main__":
    main()