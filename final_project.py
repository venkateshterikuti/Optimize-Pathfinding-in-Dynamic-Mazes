import numpy as np
import sys
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop

class Env:
    def __init__(self, level='easy'):
        print(f"Initializing Env with level: {level}")  # Debugging line
        self.level = level
        self.length = self._set_grid_size(level)
        self.start, self.trap, self.dest = self._generate_points()
        self.state = self.start
        self.episode, self.step, self.total_r = 0, 0, 0
        self.history = []

    def _set_grid_size(self, level):
        sizes = {'easy': 4, 'medium': 6, 'hard': 10}
        return sizes.get(level, 4)  # Default to 'easy' if level is invalid

    def _generate_points(self):
        points = set()
        while len(points) < 3:
            point = (np.random.randint(self.length), np.random.randint(self.length))
            points.add(point)
        return tuple(points)

    def get_actions(self):
        return ['up', 'down', 'left', 'right']

    def render(self, graphical=False):
        if graphical:
            plt.ion()  # Turn on interactive mode
            plt.show()  # Show the window (non-blocking)

            # Clear the current figure
            plt.clf()
            
            ax = plt.gca()  # Get the current axes
            ax.set_xlim(0, self.length)
            ax.set_ylim(0, self.length)

            # Creating the grid
            for x in range(self.length):
                for y in range(self.length):
                    ax.add_patch(patches.Rectangle((x, y), 1, 1, edgecolor='black', facecolor='none'))

            # Marking start, trap, and destination
            ax.add_patch(patches.Rectangle(self.start, 1, 1, facecolor='blue', label='Start'))  # Start
            ax.add_patch(patches.Rectangle(self.trap, 1, 1, facecolor='red', label='Trap'))    # Trap
            ax.add_patch(patches.Rectangle(self.dest, 1, 1, facecolor='green', label='Destination'))  # Destination
            ax.add_patch(patches.Rectangle(self.state, 1, 1, facecolor='orange', label='Agent'))  # Agent

            ax.set_aspect('equal', adjustable='box')
            plt.draw()  # Redraw the current figure
            plt.pause(0.1)  # Pause for a short period to ensure the plot updates
        else:
            # Existing text-based rendering
            view = np.full((self.length, self.length), '_ ')
            for point, char in zip([self.dest, self.trap, self.state], ['* ', 'X ', 'o ']):
                view[point] = char
            display = '\n'.join('  '.join(row) for row in view)
            info = f'EPISODE: {self.episode}, STEP: {self.step}, REWARD: {self.total_r}'
            print(f'{display}\n{info}\n')

    def go(self, action):
        move = {'up': (-1, 0), 'down': (1, 0), 'left': (0, -1), 'right': (0, 1)}.get(action, (0, 0))
        self.state = tuple(np.clip(np.add(self.state, move), 0, self.length - 1))
        reward = -0.1
        if self.state == self.dest:
            reward, done = 1.0, True
        elif self.state == self.trap:
            reward, done = -1.0, True
        else:
            done = False
        self.step += 1
        self.total_r += reward
        self.render()
        if done:
            self.history.append(self.total_r)
            self.episode += 1
            self.step, self.total_r = 0, 0
        return np.array(self.state) - np.array(self.dest), reward, done 

    def restart(self):
        self.state = self.start
        self.render()
        return np.array(self.state) - np.array(self.dest), False
    
class BasicQLearning:
    def __init__(self, states, actions, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.q_table = np.zeros((states, actions))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def choose_action(self, state):
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(range(self.q_table.shape[1]))  # Explore
        else:
            return np.argmax(self.q_table[state])  # Exploit

    def learn(self, state, action, reward, next_state):
        predict = self.q_table[state, action]
        target = reward + self.gamma * np.max(self.q_table[next_state])
        self.q_table[state, action] += self.alpha * (target - predict)


class RL:
    def __init__(self, feature_size, actions, epsilon=0.5, gamma=0.9, lr=0.01, memory_size=200, replace_target_iter=200):
        self.epsilon = epsilon
        self.gamma = gamma
        self.actions = actions
        self.memory = np.zeros((memory_size, 2 * feature_size + 2))
        self.q_eval_model = self._build_model(feature_size, len(actions), lr)
        self.q_target_model = self._build_model(feature_size, len(actions), lr)
        self.replace_target_iter = replace_target_iter
        self.mem_cnt = self.learn_cnt = 0
        self.history = []

    def _build_model(self, feature_size, action_size, lr):
        model = Sequential([
            Dense(50, activation='relu', input_shape=(feature_size,)),
            Dense(action_size)
        ])
        model.compile(loss='mse', optimizer=RMSprop(lr=lr))
        return model

    def store_observation(self, s, a, r, s_):
        index = self.mem_cnt % self.memory.shape[0]
        self.memory[index] = np.hstack((s, [self.actions.index(a), r], s_))
        self.mem_cnt += 1

    def actor(self, observation):
        if np.random.uniform() > self.epsilon:
            return np.random.choice(self.actions)
        q_values = self.q_eval_model.predict(observation[np.newaxis, :])[0]
        return self.actions[np.argmax(q_values)]

    def learn(self):
        if self.learn_cnt % self.replace_target_iter == 0:
            self.q_target_model.set_weights(self.q_eval_model.get_weights())
        batch_index = np.random.choice(self.mem_cnt if self.mem_cnt < self.memory.shape[0] else self.memory.shape[0], size=self.mem_cnt)
        batch = self.memory[batch_index]
        s, s_ = batch[:, :2], batch[:, -2:]
        q_target = self.q_eval_model.predict(s)
        q_next = self.q_target_model.predict(s_)
        q_target[np.arange(batch_index.size), batch[:, 2].astype(int)] = batch[:, 3] + self.gamma * np.max(q_next, axis=1)
        self.q_eval_model.fit(s, q_target, verbose=0)
        self.epsilon = min(self.epsilon + 0.001, 0.9)
        self.history.append(self.q_eval_model.history.history['loss'][0])
        self.learn_cnt += 1

def main(level, iteration, mem_size, delay, model_type):
    print(f"Running {model_type} model at {level} level") 
    game = Env(level)
    
    if model_type == 'dqn':
        model = RL(2, game.get_actions(), memory_size=mem_size)
    elif model_type == 'bqn':
        # Assuming state is represented simply by a number for BasicQLearning
        model = BasicQLearning(states=game.length ** 2, actions=len(game.get_actions()))
    else:
        print("Invalid model type. Please choose 'dqn' or 'bqn'.")
        return

    step = 0
    for episode in range(iteration):
        s, done = game.restart()
        while not done:
            if model_type == 'dqn':
                a = model.actor(s)
                ns, r, done = game.go(a)
                model.store_observation(s, a, r, ns)
                if step > mem_size and step % 5 == 0:
                    model.learn()
            elif model_type == 'bqn':
                # Convert state to a simple number for indexing in Q-table
                state_index = s[0] * game.length + s[1]
                a = model.choose_action(state_index)
                ns, r, done = game.go(game.get_actions()[a])
                next_state_index = ns[0] * game.length + ns[1]
                model.learn(state_index, a, r, next_state_index)
            s = ns
            step += 1

    # Visualization of results
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Game History")
    plt.plot(game.history)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")

    plt.subplot(1, 2, 2)
    plt.title(f"{model_type.upper()} Learning History")
    plt.plot(model.history)
    plt.xlabel("Learning Step")
    plt.ylabel("Loss")

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Maze Environment')
    parser.add_argument('--level', type=str, default='easy', choices=['easy', 'medium', 'hard'], help='Difficulty level of the maze')
    parser.add_argument('--iteration', type=int, default=200, help='Number of training iterations')
    parser.add_argument('--memorysize', type=int, default=2000, help='Size of memory for RL')
    parser.add_argument('--delay', type=float, default=0.1, help='Delay in rendering')
    parser.add_argument('--model', type=str, default='dqn', choices=['dqn', 'bqn'], help='Model type: DQN or Basic Q-Learning')
    args = parser.parse_args()

    main(args.level, args.iteration, args.memorysize, args.delay, args.model)
    
