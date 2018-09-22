'''
###############################################################################################
    SARSA vs. Q Learning (Reinforcement Learning: An Introduction, Sutton, Barto, fig 6.4)
    Created by Jet-Tsyn Lee 23/02/18, last update 26/02/18

    Program is to compare the optimal results found usign a SARSA/Q Learning approch to the
    cliff walking game, presented in the Reinforcement Learning: An Introduction book, Sutton,
    Barto, fig 6.4.
###############################################################################################
'''

import numpy as np
import matplotlib.pyplot as plt


# ACTIONS
# Public variable, action direction corresponding to the direction in a array format
ACT_NORTH = 0
ACT_SOUTH = 1
ACT_EAST = 2
ACT_WEST = 3
action = [ACT_NORTH, ACT_SOUTH, ACT_EAST, ACT_WEST]


# ~~~~~~~~~~  AGENT  ~~~~~~~~~~
# Stores the current agents information and the required actions in the game state (SARSA, Q Learning)
class Agent(object):

    # Constructor
    def __init__(self, policyType, epsilon=0.0, alpha=0.0, gamma=0.0):
        self.alpha = alpha  # Step size
        self.gamma = gamma  # Discounted rate
        self.epsilon = epsilon  # greedy probability
        self.curState = []  # Sets agents current state

        self.policy = str(policyType)

    # Return String
    def __str__(self):
        return self.policy

    # Action movement, using a epsilon-greedy method
    def move(self, state, stateArr):
        # If, probability less than epsilon, pick random action
        if np.random.rand() < self.epsilon:
            a = np.random.choice(action)

        # Else, find maximum Q value
        else:
            qState = stateArr[state[0], state[1], :]
            maxAction = np.argmax(qState)  # Find max value estimate
            # find identical Q values, if more than 1, then randomly pick
            actionArr = np.where(qState == np.argmax(qState))[0]
            if len(actionArr) == 0:
                a = maxAction
            else:
                a = np.random.choice(actionArr)

        # Return action value
        return a


    # SARSA policy  ~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def sarsa(self, stateArr, cliff):

        # Reset Agent to start state
        self.curState = cliff.startState

        # Choose a from s
        setAction = self.move(self.curState, stateArr)

        reward = 0.0  # Sum rewards per step

        # Repeat until goal state is reached
        while self.curState != cliff.goalState:
            # Observe r
            r = cliff.rewardArr[self.curState[0]][self.curState[1]][setAction]
            reward += r

            # Observe s'
            statePri = cliff.next_state[self.curState[0]][self.curState[1]][setAction]

            # choose a' from s'
            actionPri = self.move(statePri, stateArr)

            # State-Action, gamma*Q(s',a')
            qSPriAPri = self.gamma * stateArr[statePri[0], statePri[1], actionPri]

            # Update Q(s,a), SARSA
            stateArr[self.curState[0], self.curState[1], setAction] += \
                self.alpha * (r + qSPriAPri - stateArr[self.curState[0]][self.curState[1]][setAction])

            # s<-s',a<-a'
            self.curState = statePri
            setAction = actionPri

        # Return reward, limited to -100
        return max(reward, -100)


    # Q-Learning policy  ~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def qLearning(self, stateArr, cliff):
        self.curState = cliff.startState
        reward = 0.0

        while self.curState != cliff.goalState:
            # Choose a from s
            setAction = self.move(self.curState, stateArr)

            # Take action a
            # Observe R
            r = cliff.rewardArr[self.curState[0]][self.curState[1]][setAction]
            reward += r  # Cumulative reward

            # Observe s'
            statePri = cliff.next_state[self.curState[0]][self.curState[1]][setAction]

            # set Q(s,a), QLearning
            qSPriAPri = self.gamma * np.max(stateArr[statePri[0], statePri[1], :])

            stateArr[self.curState[0], self.curState[1], setAction] += \
                self.alpha * (r + qSPriAPri - stateArr[self.curState[0]][self.curState[1]][setAction])

            # S <- s'
            self.curState = statePri

        # Return reward, limited to -100
        return max(reward, -100)




# ~~~~~~~~~~  CLIFF  ~~~~~~~~~~
# Stores the variables regarding the game space and all required array infomration to be passed.
class Cliff(object):


    # Constructor
    def __init__(self, height=4, length=12):

        # State array Q, length X width X all actions
        self.stateArr = np.zeros((height, length, len(action)))
        self.startState = [3, 0]
        self.goalState = [3, 11]
        self.cliffArea = [[3, 1],[3, 2],[3, 3],[3, 4],[3, 5],[3, 6],[3, 7],[3, 8],[3, 9],[3, 10]]

        # Reward Array
        self.rewardArr = np.zeros_like(self.stateArr)
        # Set cliff rewards
        self.rewardArr[:, :, :] = -1.0  # Set all reward spaces
        # if action is selected from current posifion to cliff area, reward -100
        self.rewardArr[2, 1:11, ACT_SOUTH] = -100.0
        self.rewardArr[3, 0, ACT_EAST] = -100.0

        # Set a state array to determine the next state positions when moved
        self.next_state = []

        # Loop for every position and set direction
        for iHeight in range(0, height):
            # Add space
            self.next_state.append([])

            for jLength in range(0, length):

                # Create Array for each action to store next state
                stateAction = [[0, 0], [0, 0], [0, 0], [0, 0]]

                # Set state position depending on direction
                stateAction[ACT_NORTH] = [max(iHeight - 1, 0), jLength]
                stateAction[ACT_WEST] = [iHeight, max(jLength - 1, 0)]
                stateAction[ACT_EAST] = [iHeight, min(jLength + 1, length - 1)]
                stateAction[ACT_SOUTH] = [min(iHeight + 1, height - 1), jLength]

                # Set Cliff area, reset to start state if agent moves to position
                if self.rewardArr[iHeight][jLength][ACT_NORTH] <= -100:
                    stateAction[ACT_NORTH] = self.startState
                if self.rewardArr[iHeight][jLength][ACT_EAST] <= -100:
                    stateAction[ACT_EAST] = self.startState
                if self.rewardArr[iHeight][jLength][ACT_WEST] <= -100:
                    stateAction[ACT_WEST] = self.startState
                if self.rewardArr[iHeight][jLength][ACT_SOUTH] <= -100:
                    stateAction[ACT_SOUTH] = self.startState

                # Add dictionary to Next state array
                self.next_state[-1].append(stateAction)




# ~~~~~~~~~~  ENVIRONMENT  ~~~~~~~~~~
# Environment state to run all objects in a single iteration
class Environment(object):

    # Constructor - Stores all objects
    def __init__(self, agents, cliff, episodes=1):
        self.agents = agents
        self.cliff = cliff
        self.episodes = episodes


    # Run game
    def run(self):
        # Reward array
        agentReward = np.zeros((self.episodes, len(self.agents)))

        # Create independent copy of Q array for both policy tests
        stateSarsa = np.copy(self.cliff.stateArr)
        stateQLear = np.copy(self.cliff.stateArr)

        # Loop fo the number of episodes
        for iEpisode in range(self.episodes):

            agtCnt = 0
            for jAgent in self.agents:
                if jAgent.policy == "SARSA":
                    agentReward[iEpisode][agtCnt] += jAgent.sarsa(stateSarsa, self.cliff)
                elif jAgent.policy == "QLearning":
                    agentReward[iEpisode][agtCnt] += jAgent.qLearning(stateQLear, self.cliff)
                agtCnt += 1

        for jAgent in self.agents:
            if jAgent.policy == "SARSA":
                jAgent.qState = stateSarsa
            elif jAgent.policy == "QLearning":
                jAgent.qState = stateQLear

        return agentReward



# ~~~~~~~~~~  MAIN  ~~~~~~~~~
if __name__ == "__main__":


    # ======  CREATE OBJECTS  ======
    # Agents
    ALPHA = 0.1  # Step size
    GAMMA = 1.0  # Learning Rate
    EPSILON = 0.8  # Greedy probability

    # Agents - SARSA and qLearning policy
    agents = [Agent("SARSA", EPSILON, ALPHA, GAMMA),
              Agent("QLearning", EPSILON, ALPHA, GAMMA)]

    # Cliff
    CLIFF_H = 4  # Cliff height
    CLIFF_W = 12  # Cliff Width
    cliff = Cliff(CLIFF_H, CLIFF_W)

    # Test Environment
    EPISODES = 500  # Number of episodes
    ITERATIONS = 10  # Number of iterations
    environment = Environment(agents, cliff, EPISODES)

    # =======  RUN COMPARISON==========

    # Run environment, looping for the number of iterations and average results
    rewards = np.zeros((EPISODES, len(agents)))
    for iter in range(0, ITERATIONS):
        print("Running for Iteration:", iter)
        rewards += environment.run()

    # Average rewards
    rewards /= ITERATIONS

    # ======  SMOOTH RESULTS  =======
    sucEpisodes = 10  # number of successful episodes
    # Smoothed rewards, set walues to -100 to drop the first 10 values
    smoothedReward = np.zeros_like(rewards)
    smoothedReward[:] = -100

    # Calculate the mean of the results from the previous 10 episodes
    for iRow in range(sucEpisodes, EPISODES):
        for jAgent in range(len(agents)):
            sum = 0.0
            for kAdd in range(sucEpisodes, 0, -1):
                sum += rewards[iRow - kAdd][jAgent]

            smoothedReward[iRow][jAgent] = sum / sucEpisodes


    # Print optimal path
    for agent in agents:
        optPath = []
        for i in range(0, CLIFF_H):
            optPath.append([])

            for j in range(0, CLIFF_W):
                if [i, j] == cliff.goalState:
                    optPath[-1].append('G')
                    continue

                if [i, j] in cliff.cliffArea:
                    optPath[-1].append('C')
                    continue

                maxAct = np.argmax(agent.qState[i, j, :])
                if maxAct == ACT_NORTH:
                    optPath[-1].append('N')
                elif maxAct == ACT_SOUTH:
                    optPath[-1].append('S')
                elif maxAct == ACT_WEST:
                    optPath[-1].append('W')
                elif maxAct == ACT_EAST:
                    optPath[-1].append('E')

        # Show table
        optimalTable = plt.subplots()
        plt.title(str(agent) + " - Optimal Route")
        plt.axis("off")
        optimalTable = plt.table(cellText=optPath, loc=1)

    plt.show()


    # ======  GRAPH  ======
    # Plot Data to graph
    plt.title("Cliffwalking - SARSA vs. Q-Learning, Epsilon="+str(EPSILON))
    plt.plot(smoothedReward)
    plt.ylabel('Sum of rewards during episode')
    plt.xlabel('Episode')
    plt.legend(agents, loc=4)
    plt.show()
