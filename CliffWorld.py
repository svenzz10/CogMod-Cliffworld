import numpy as np
import random as random
import matplotlib.pyplot as plt

ROWS = 4 # number of rows
COLS = 12 # number of cols
S = (3, 0) #start position of agent
G = (3, 11) #goal position

class Cliff:

    # iniatialize board
    def __init__(self):
        self.end = False
        self.pos = S
        self.board = np.zeros([4, 12])
        self.steps = 0
        # add cliff marked as -1
        self.board[3, 1:11] = -1

    #move agent position with the correct action
    def nextPosition(self, action):
        if action == "up":
            nxtPos = (self.pos[0] - 1, self.pos[1])
        elif action == "down":
            nxtPos = (self.pos[0] + 1, self.pos[1])
        elif action == "left":
            nxtPos = (self.pos[0], self.pos[1] - 1)
        else:
            nxtPos = (self.pos[0], self.pos[1] + 1)
        # check legitimacy
        if nxtPos[0] >= 0 and nxtPos[0] <= 3:
            if nxtPos[1] >= 0 and nxtPos[1] <= 11:
                self.pos = nxtPos

        #end game states
        #check if next position is a goal state
        if self.pos == G:
            self.end = True
            print("Game ends reaching goal")
        #check if next position is a cliff
        if self.board[self.pos] == -1:
            self.end = True
            print("Game ends falling off cliff")
        return self.pos

    def getReward(self):
        # give reward just like Example 6.6 from book by Sutton
        if self.pos == G:
            return -1
        if self.board[self.pos] == 0:
            return -1
        #cliff has a reward of -100
        return -100

class Agent:
    def __init__(self, epsilon=0.3, lr=0.1, sarsa=True, softmax=False, temp=0.9):
        self.cliff = Cliff() #create new board
        self.actions = ["up", "left", "right", "down"] #define actions
        self.states = []  # record position and action of each episode
        self.pos = S #set agent position to start position
        self.epsilon = epsilon #epsilson rate
        self.lr = lr #learning rate
        self.sarsa = sarsa #if true sarsa is used, if false q-learning is used
        self.softmax = softmax #if true softmax is used, if false e-greedy is used
        self.state_actions = {} #define state actions
        self.temp = temp
        for i in range(ROWS):
            for j in range(COLS):
                self.state_actions[(i, j)] = {}
                for a in self.actions:
                    self.state_actions[(i, j)][a] = 0

    #Compute softmax values for each sets of scores in x.
    def calcSoftmax(self, values, temp):
        #temp specificies hardness
        #temp < 1 = harder -> algorithm gets more confident
        #temp > 1 = softer -> more equiprobable
        tempValues = [x / temp for x in values] #implement temperature for manipulating softness of softmax function
        bottom = sum(np.exp(tempValues)) #bottom of Boltzmann equation
        softmax = np.exp(tempValues)/bottom #full softmax Boltzmann equation
        return softmax

    def chooseAction(self):
        #softmax
        if self.softmax:
            if self.epsilon==0: #only used for calculating optimal policy -> calculateOptimalPolicy()
                max_next_reward = -999 # initialize on -999 to make sure first value will be max
                for a in self.actions: #loop through the 4 actions
                    current_position = self.pos
                    next_reward = self.state_actions[current_position][a] #check the next reward
                    if next_reward >= max_next_reward: # select the action with the max reward
                        action = a
                        max_next_reward = next_reward
                return action

            rewardValues = []
            for a in self.actions:
                rewardValues.append(self.state_actions[self.pos][a])

            softmaxProbabilities = self.calcSoftmax(rewardValues, self.temp)
            action = np.random.choice(a=self.actions, p=softmaxProbabilities)
            return action

        # epsilon-greedy    
        else:
            max_next_reward = -999 # initialize on -999 to make sure first value will be max
            action = ""


            if np.random.uniform(0, 1) <= self.epsilon: #epsilon chance to get a random action
                action = np.random.choice(self.actions)
            else:
                # greedy action
                for a in self.actions: #loop through the 4 actions
                    current_position = self.pos
                    next_reward = self.state_actions[current_position][a] #check the next reward
                    if next_reward >= max_next_reward: # select the action with the max reward
                        action = a
                        max_next_reward = next_reward
            return action

    def reset(self):
        self.states = [] #reset states
        self.cliff = Cliff() #make new board
        self.pos = S #reset player position

    def learn(self, iterations=10):
        reward_cache = list() #define reward cache
        step_cache = list() #define step cache
        for _ in range(iterations):
            while True:
                reward_cum = 0 # cumulative reward of the episode
                step_cum = 0 # keeps number of iterations untill the end of the game

                curr_state = self.pos #get current state
                action = self.chooseAction() #choose action

                # next position
                self.cliff.pos = self.cliff.nextPosition(action)
                self.pos = self.cliff.pos
                step_cum += 1

                cur_reward = self.cliff.getReward() #get reward
                reward_cum += cur_reward

                next_action = self.chooseAction() #determine next action used for SARSA
                next_state = self.pos
                if self.sarsa:
                    current_value = self.state_actions[curr_state][action] #get state action
                    next_state_value = self.state_actions[next_state][next_action] # differs from q-learning uses the next action determined by policy
                    # reward = Q(S,A) + lr * (state_reward + Q'(S,A) - Q(S,A))
                    reward = current_value + self.lr * (cur_reward + next_state_value - current_value)
                    self.state_actions[curr_state][action] = round(reward, 3) #define reward per Q(S,A)
                else:
                    current_value = self.state_actions[curr_state][action] #get current value
                    maximum_state_value = np.max(list(self.state_actions[self.pos].values()))  # maximum
                    #q-learning uses off-policy next maximum state value to calculate reward
                    reward = current_value + self.lr * (cur_reward + maximum_state_value - current_value)
                    self.state_actions[curr_state][action] = round(reward, 3) #define reward per Q(S,A)
                if self.cliff.end:
                    reward_cache.append(reward_cum) #append the cumulative reward to the reward cache
                    step_cache.append(step_cum) #append the cumulative step to the step cache
                    break

            self.reset()
        return reward_cache, step_cache #return reward and step cache used for plotting


def showRoute(states):
    board = np.zeros([4, 12])
    # add cliff marked as -1
    board[3, 1:11] = -1
    for i in range(0, ROWS):
        print('-------------------------------------------------')
        out = '| '
        for j in range(0, COLS):
            token = '0'
            if board[i, j] == -1:
                token = '*'
            if (i, j) in states: #for every Q(S,A) define path
                token = 'P'
            if (i, j) == G:
                token = 'G'
            out += token + ' | '
        print(out)
    print('-------------------------------------------------') 

#visualizes the reward convergence
def plot_cumreward_normalized(reward_cache_qlearning, reward_cache_SARSA):
    cum_rewards_q = []
    rewards_mean = np.array(reward_cache_qlearning).mean()
    rewards_std = np.array(reward_cache_qlearning).std()
    count = 0 # used to determine the batches
    cur_reward = 0 # accumulate reward for the batch
    for cache in reward_cache_qlearning:
        count = count + 1
        cur_reward += cache
        if(count == 10):
            # normalize the sample
            normalized_reward = (cur_reward - rewards_mean)/rewards_std
            cum_rewards_q.append(normalized_reward)
            cur_reward = 0
            count = 0
            
    cum_rewards_SARSA = []
    rewards_mean = np.array(reward_cache_SARSA).mean()
    rewards_std = np.array(reward_cache_SARSA).std()
    normalized_reward = (cur_reward - rewards_mean)/rewards_std
    count = 0 # used to determine the stepsize
    cur_reward = 0 # accumulate reward for the batch
    for cache in reward_cache_SARSA:
        count = count + 1
        cur_reward += cache
        if(count == 10):
            # normalize the sample
            normalized_reward = (cur_reward - rewards_mean)/rewards_std
            cum_rewards_SARSA.append(normalized_reward)
            cur_reward = 0
            count = 0      
    # prepare the graph  

    """Uncomment this to create a plot about Q-learning vs SARSA"""
    plt.plot(cum_rewards_q, label = "Q-Learning")
    plt.plot(cum_rewards_SARSA, label = "SARSA")
    plt.ylabel('Cumulative Rewards per Cluster of Episodes')
    plt.xlabel('Episodes (cluster of 10)')
    plt.title("Q-Learning/SARSA Performance")
    plt.legend(loc='lower right', ncol=2, mode="expand", borderaxespad=0.)
    plt.savefig('performance-qlearning-sarsa')

    """Uncomment this to create a plot about softmax vs e-greedy"""
    # plt.plot(cum_rewards_q, label = "Q-Learning E-greedy")
    # plt.plot(cum_rewards_SARSA, label = "Q-Learning Softmax")
    # plt.ylabel('Cumulative Rewards per Cluster of Episodes')
    # plt.xlabel('Episodes (cluster of 10)')
    # plt.title("Action Selectors Performance E-Greedy vs Softmax")
    # plt.legend(loc='lower right', ncol=2, mode="expand", borderaxespad=0.)
    # plt.savefig('performance-action-selectors')

def plot_cumreward_normalized_temperatures(reward_cache1, reward_cache2, reward_cache3, reward_cache4):
    cum_rewards_1 = []
    rewards_mean = np.array(reward_cache1).mean()
    rewards_std = np.array(reward_cache1).std()
    count = 0 # used to determine the batches
    cur_reward = 0 # accumulate reward for the batch
    for cache in reward_cache1:
        count = count + 1
        cur_reward += cache
        if(count == 10):
            # normalize the sample
            normalized_reward = (cur_reward - rewards_mean)/rewards_std
            cum_rewards_1.append(normalized_reward)
            cur_reward = 0
            count = 0

    cum_rewards_2 = []
    rewards_mean = np.array(reward_cache2).mean()
    rewards_std = np.array(reward_cache2).std()
    count = 0 # used to determine the batches
    cur_reward = 0 # add rewards for the batch of 10
    for cache in reward_cache2:
        count = count + 1
        cur_reward += cache
        if(count == 10):
            # normalize the sample
            normalized_reward = (cur_reward - rewards_mean)/rewards_std
            cum_rewards_2.append(normalized_reward)
            cur_reward = 0
            count = 0    

    cum_rewards_3 = []
    rewards_mean = np.array(reward_cache3).mean()
    rewards_std = np.array(reward_cache3).std()
    count = 0 # used to determine the batches
    cur_reward = 0 # add rewards for the batch of 10
    for cache in reward_cache3:
        count = count + 1
        cur_reward += cache
        if(count == 10):
            # normalize the sample
            normalized_reward = (cur_reward - rewards_mean)/rewards_std
            cum_rewards_3.append(normalized_reward)
            cur_reward = 0
            count = 0    

    cum_rewards_4 = []
    rewards_mean = np.array(reward_cache4).mean()
    rewards_std = np.array(reward_cache4).std()
    count = 0 # used to determine the batches
    cur_reward = 0 # accumulate reward for the sample size
    for cache in reward_cache4:
        count = count + 1
        cur_reward += cache
        if(count == 10):
            # normalize the sample
            normalized_reward = (cur_reward - rewards_mean)/rewards_std
            cum_rewards_4.append(normalized_reward)
            cur_reward = 0
            count = 0 
    
    # prepare the graph  

    plt.plot(cum_rewards_1, label = "SARSA with softmax T=0.5")
    plt.plot(cum_rewards_2, label = "SARSA with softmax T=0.9")
    plt.plot(cum_rewards_3, label = "SARSA with softmax T=5")
    plt.plot(cum_rewards_4, label = "SARSA with softmax T=15")
    plt.ylabel('Cumulative Rewards per Cluster of Episodes')
    plt.xlabel('Episodes (cluster of 10)')
    plt.title("SARSA Performance Softmax Temperatures")
    plt.legend(loc='lower right')
    plt.savefig('performance-temperatures')

def calculateOptimalPolicy(ag):
    states = []

    ag_optimal = Agent(epsilon=0) #initialize test agent for optimal policy
    #set epsilon to 0 so that the agent uses state_actions from ag
    ag_optimal.state_actions = ag.state_actions  #set agent state actions to optimal

    while True: #run till all states are checked
        curr_state = ag_optimal.pos
        action = ag_optimal.chooseAction()
        states.append(curr_state) #append state so that we can use this later to graph the route
        print("current position {} |action {}".format(curr_state, action)) #print taken action

        # next position
        ag_optimal.cliff.pos = ag_optimal.cliff.nextPosition(action)
        ag_optimal.pos = ag_optimal.cliff.pos

        if ag_optimal.cliff.end:
            break
    
    #show the taken route
    showRoute(states)

if __name__ == "__main__":
    #Use this code to play around if you want;
    
    c = Cliff() #create board

    #ag = Agent(epsilon = 0.1, lr=0.1, sarsa=True, softmax=False)
    # ag.learn(500)
    # calculateOptimalPolicy(ag)
    
    """To Generate Figure Performance SARSA with softmax different temperatures
        uncomment to use
    """
    # ag = Agent(epsilon=0.1, lr=0.1, sarsa=False, softmax=True, temp=0.5) #settings for agent
    # reward_cache_temp05, _ = ag.learn(iterations=500) #used for training the agent 
    # calculateOptimalPolicy(ag) #calculating optimal policy

    # ag = Agent(epsilon=0.1, lr=0.1, sarsa=True, softmax=True, temp=0.9) #settings for agent
    # reward_cache_temp09, _ = ag.learn(iterations=500) #used for training the agent 
    # calculateOptimalPolicy(ag)

    # ag = Agent(epsilon=0.1, lr=0.1, sarsa=False, softmax=True, temp=5) #settings for agent
    # reward_cache_temp5, _ = ag.learn(iterations=500) #used for training the agent 
    # calculateOptimalPolicy(ag) #calculating optimal policy

    # ag = Agent(epsilon=0.1, lr=0.1, sarsa=False, softmax=True, temp=15) #settings for agent
    # reward_cache_temp15, _ = ag.learn(iterations=500) #used for training the agent 
    # calculateOptimalPolicy(ag) #calculating optimal policy

    # plot_cumreward_normalized_temperatures(reward_cache_temp05, reward_cache_temp09, reward_cache_temp5, reward_cache_temp15)

    """To Generate Figure Q-learning softmax vs e-greedy
        uncomment to use
    """
    # ag = Agent(epsilon=0.1, lr=0.1, sarsa=False, softmax=False) #settings for agent
    # reward_cache_q_learning, _ = ag.learn(iterations=500) #used for training the agent 
    # calculateOptimalPolicy(ag)

    # ag = Agent(epsilon=0.1, lr=0.1, sarsa=False, softmax=True) #settings for agent
    # reward_cache_sarsa, _ = ag.learn(iterations=500) #used for training the agent 
    # calculateOptimalPolicy(ag) #calculating optimal policy

    # plot_cumreward_normalized(reward_cache_q_learning, reward_cache_sarsa)
    
    """To Generate Figure Performance Q-learning vs Sarsa
        uncomment to use
    """
    ag = Agent(epsilon=0.1, lr=0.1, sarsa=False, softmax=False) #settings for agent
    reward_cache_q_learning, _ = ag.learn(iterations=500) #used for training the agent 
    calculateOptimalPolicy(ag)

    ag = Agent(epsilon=0.1, lr=0.1, sarsa=True, softmax=False) #settings for agent
    reward_cache_sarsa, _ = ag.learn(iterations=500) #used for training the agent 
    calculateOptimalPolicy(ag) #calculating optimal policy

    plot_cumreward_normalized(reward_cache_q_learning, reward_cache_sarsa)
