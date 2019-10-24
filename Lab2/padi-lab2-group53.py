#!/usr/bin/env python
# coding: utf-8

# # Learning and Decision Making

# ## Laboratory 2: The Taxi Problem
# 
# In the end of the lab, you should export the notebook to a Python script (File >> Download as >> Python (.py)). Your file should be named `padi-lab2-groupXX.py`, where the `XX` corresponds to your group number and should be submitted to the e-mail <adi.tecnico@gmail.com>. 
# 
# Make sure...
# 
# * **... that the subject is of the form `[<group n.>] LAB <lab n.>`.** 
# 
# * **... to strictly respect the specifications in each activity, in terms of the intended inputs, outputs and naming conventions.** 
# 
# In particular, after completing the activities you should be able to replicate the examples provided (although this, in itself, is no guarantee that the activities are correctly completed).
# 
# ### 1. The MDP Model 
# 
# Consider once again the taxi domain described in the Homework which you modeled using a Markov decision process. In this lab you will interact with larger version of the same problem. You will use an MDP based on the aforementioned domain and investigate how to evaluate, solve and simulate a Markov decision problem. The domain is represented in the diagram below.
# 
# <img src="taxi.png" width="200px">
# 
# In the taxi domain above,
# 
# * The taxi can be in any of the 25 cells in the diagram. The passenger can be at any of the 4 marked locations ($Y$, $B$, $G$, $R$) or in the taxi. Additionally, the passenger wishes to go to one of the 4 possible destinations. The total number of states, in this case, is $25\times 5\times 4$.
# * At each step, the agent (taxi driver) may move in any of the four directions -- south, north, east and west. It can also pickup the passenger or drop off the passenger. 
# * The goal of the taxi driver is to pickup the passenger and drop it at the passenger's desired destination.
# 
# **Throughout the lab, use $\gamma=0.99$.**
# 
# $$\diamond$$

# In this first activity, you will implement an MDP model in Python. You will start by loading the MDP information from a `numpy` binary file, using the `numpy` function `load`. The file contains the list of states, actions, the transition probability matrices and cost function.
# 
# ---
# 
# #### Activity 1.        
# 
# Write a function named `load_mdp` that receives, as input, a string corresponding to the name of the file with the MDP information, and a real number $\gamma$ between $0$ and $1$. The loaded file contains 4 arrays:
# 
# * An array `S` that contains all the states in the MDP. There is a total of $501$ states describing the possible taxi-passenger configurations. Those states are represented as strings of the form `"(x, y, p, d)"`, where $(x,y)$ represents the position of the taxi in the grid, $p$ represents the position of the passenger ($R$, $G$, $Y$, $B$, or in the taxi), and $d$ the destination of the passenger ($R$, $G$, $Y$, $B$). There is one additional absorbing state called `"Final"` to which the MDP transitions after reaching the goal.
# * An array `A` that contains all the actions in the MDP. Each action is represented as a string `"South"`, `"North"`, and so on.
# * An array `P` containing 5 $501\times 501$ sub-arrays, each corresponding to the transition probability matrix for one action.
# * An array `c` containing the cost function for the MDP.
# 
# Your function should create the MDP as a tuple `(S, A, (Pa, a = 0, ..., 5), c, g)`, where `S` is a tuple containing the states in the MDP represented as strings (see above), `A` is a tuple containing the actions in the MDP represented as strings (see above), `P` is a tuple with 6 elements, where `P[a]` is an np.array corresponding to the transition probability matrix for action `a`, `c` is an np.array corresponding to the cost function for the MDP, and `g` is a float, corresponding to the discount and provided as the argument $\gamma$ of your function. Your function should return the MDP tuple.
# 
# **Note**: Don't forget to import `numpy`.
# 
# ---

# In[1]:


import numpy as np
import random as rand

def load_mdp(filename, y):
    content = np.load(filename)
    S = content['S']
    A = content['A']
    c = content['c']
    P = []
    for m in content['P']:
        P.append(m)
    P = tuple(P)
    return (S, A, P, c, y)
    
M = load_mdp("taxi.npz", 0.99)
print(M[3])


# We provide below an example of application of the function with the file `taxi.npz` that you can use as a first "sanity check" for your code.
# 
# ```python
# import numpy.random as rand
# 
# M = load_mdp('taxi.npz', 0.99)
# 
# rand.seed(42)
# 
# # States
# print('Number of states:', len(M[0]))
# 
# # Random state
# s = rand.randint(len(M[0]))
# print('Random state:', M[0][s])
# 
# # Final state
# print('Final state:', M[0][-1])
# 
# # Actions
# print('Number of actions:', len(M[1]))
# 
# # Random action
# a = rand.randint(len(M[1]))
# print('Random action:', M[1][a])
# 
# # Transition probabilities
# print('Transition probabilities for the selected state/action:')
# print(M[2][a][s, :])
# 
# # Cost
# print('Cost for the selected state/action:')
# print(M[3][s, a])
# 
# # Discount
# print('Discount:', M[4])
# ```
# 
# Output:
# 
# ```
# Number of states: 501
# Random state: (1, 0, 0, 2)
# Final state: Final
# Number of actions: 6
# Random action: West
# Transition probabilities for the selected state/action:
# [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
#   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
#   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
#   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
#   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
#   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  1.  0.  0.  0.  0.  0.
#   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
#   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
#   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
#   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
#   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
#   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
#   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
#   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
#   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
#   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
#   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
#   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
#   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
#   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
#   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
#   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
#   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
#   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
#   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
#   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
#   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
#   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
# Cost for the selected state/action:
# 0.7
# Discount: 0.99
# ```

# ### 2. Prediction
# 
# You are now going to evaluate a given policy, computing the corresponding cost-to-go.

# ---
# 
# #### Activity 2.
# 
# You will now describe the policy that, at each state $x$, always moves the taxi down (South). Recall that the action "South" corresponds to the action index $0$. Your policy should be a `numpy` array named `pol` with as many rows as states and as many columns as actions, where `pol[s,a]` should contain the probability of action `a` in state `s` according to the desired policy. 
# 
# ---

# In[2]:


pol = np.zeros((len(M[0]), len(M[1])))
for i in range(len(M[0])):
    pol[i, 0] = 1
print(pol)


# ---
# 
# #### Activity 3.
# 
# You will now write a function called `evaluate_pol` that evaluates a given policy. Your function should receive, as an input, an MDP described as a tuple like that of **Activity 1** and a policy described as an array like that of **Activity 2** and return a `numpy` array corresponding to the cost-to-go function associated with the given policy.
# 
# ---

# In[3]:


import math

def evaluate_pol(MDP, pol):
    y = MDP[4] #discount
    c = MDP[3] #cost
    Jpi = np.zeros((len(MDP[0]), 1))
    I = np.eye(len(MDP[0]))
    for i in range(len(MDP[1])):
        P = MDP[2][i] * y
        res = np.subtract(I, P)

        Jpi[:,0] = np.linalg.inv(res).dot(c[:,np.argmax(pol[i])])
    #print(Jpi)
    return Jpi

Jpi = evaluate_pol(M, pol)

rand.seed(42)

s = rand.randint(0,len(M[0]))
print('Cost to go at state %s:' % M[0][s], Jpi[s])

s = rand.randint(0,len(M[0]))
print('Cost to go at state %s:' % M[0][s], Jpi[s])

s = rand.randint(0,len(M[0]))
print('Cost to go at state %s:' % M[0][s], Jpi[s])


# As an example, you can evaluate the policy from **Activity 2** in the MDP from **Activity 1**.
# 
# ```python
# Jpi = evaluate_pol(M, pol)
# 
# rand.seed(42)
# 
# s = rand.randint(len(M[0]))
# print('Cost to go at state %s:' % M[0][s], Jpi[s])
# 
# s = rand.randint(len(M[0]))
# print('Cost to go at state %s:' % M[0][s], Jpi[s])
# 
# s = rand.randint(len(M[0]))
# print('Cost to go at state %s:' % M[0][s], Jpi[s])
# ```
# 
# Output: 
# ```
# Cost to go at state (1, 0, 0, 2): [70.]
# Cost to go at state (4, 1, 3, 3): [70.]
# Cost to go at state (3, 2, 2, 0): [70.]
# ```

# ### 3. Control
# 
# In this section you are going to compare value and policy iteration, both in terms of time and number of iterations.

# ---
# 
# #### Activity 4
# 
# In this activity you will show that the policy in Activity 3 is _not_ optimal. For that purpose, you will use value iteration to compute the optimal cost-to-go, $J^*$, and show that $J^*\neq J^\pi$. 
# 
# Write a function called `value_iteration` that receives as input an MDP represented as a tuple like that of **Activity 1** and returns an `numpy` array corresponding to the optimal cost-to-go function associated with that MDP. Before returning, your function should print:
# 
# * The time it took to run, in the format `Execution time: xxx seconds`, where `xxx` represents the number of seconds rounded up to $3$ decimal places.
# * The number of iterations, in the format `N. iterations: xxx`, where `xxx` represents the number of iterations.
# 
# **Note 1:** Stop the algorithm when the error between iterations is smaller than $10^{-8}$.
# 
# **Note 2:** You may find useful the function ``time()`` from the module ``time``.
# 
# ---

# In[10]:


import time
def value_iteration(MDP):
    c = MDP[3]
    y = MDP[4]
    P = MDP[2]
    error = 1
    i = 0
    Jpi = evaluate_pol(MDP, pol)
    Jstar = np.zeros((len(MDP[0])))
    Q = np.zeros((len(MDP[0]), len(MDP[1])))
    t = time.time()
    while(error > 1e-8):
        for column in range(len(MDP[1])):
            Q[:,column] = c[:,column] + y * P[column].dot(Jstar)
        
        #minimum for each action
        J = np.amin(Q, axis=1)
        i += 1
        
        #normalization to return one single value instead of an array
        error = np.linalg.norm(J - Jstar)
        Jstar = J
    t1 = time.time()
    total = t1-t
    print("Time:", round(total, 3))
    print('Iterations: ', i)
    return Q

value_iteration(M)


# For example, the optimal cost-to-go for the MDP from **Activity 1** is can be computed as follows.
# 
# ```python
# Jopt = value_iteration(M)
# 
# rand.seed(42)
# 
# s = rand.randint(len(M[0]))
# print('Cost to go at state %s:' % M[0][s], Jopt[s])
# 
# s = rand.randint(len(M[0]))
# print('Cost to go at state %s:' % M[0][s], Jopt[s])
# 
# s = rand.randint(len(M[0]))
# print('Cost to go at state %s:' % M[0][s], Jopt[s])
# 
# print('\nIs the policy from Activity 2 optimal?', np.all(np.isclose(Jopt, Jpi)))
# ```
# 
# Output:
# ```
# Execution time: 0.031 seconds
# N. iterations: 18
# Cost to go at state (1, 0, 0, 2): [4.1]
# Cost to go at state (4, 1, 3, 3): [4.76]
# Cost to go at state (3, 2, 2, 0): [6.69]
# 
# Is the policy from Activity 2 optimal? False
# ```

# ---
# 
# #### Activity 5
# 
# You will now compute the optimal policy using policy iteration. Write a function called `policy_iteration` that receives as input an MDP represented as a tuple like that of **Activity 1** and returns an `numpy` array corresponding to the optimal policy associated with that MDP. Your function should print the time it takes to run before returning, in the format `Execution time: xxx seconds`, where `xxx` represents the number of seconds rounded up to $3$ decimal places.
# 
# **Note:** If you find that numerical errors affect your computations (especially when comparing two values/arrays) you may use the `numpy` function `isclose` with adequately set absolute and relative tolerance parameters (e.g., $10^{-8}$).
# 
# ---

# In[14]:


def policy_iteration(MDP):
    c = MDP[3]
    P = MDP[2]
    y = MDP[4]
    
    e = 1
    i = 0
    Jpi = np.zeros((len(MDP[0]), 1))
    
    Q = np.zeros((len(MDP[0]), len(MDP[1])))
    p = np.ones((len(MDP[0]), len(MDP[1])))
    cpi = np.zeros((len(MDP[0])))
    ppi = np.zeros((len(MDP[0])))
    pinew = np.zeros((len(MDP[0]), len(MDP[1])))


    diff = True
    while diff:
        for action in range(len(MDP[1])):
            cpi += np.diag(p[:,action]).dot(c[:,action].reshape(501))
            ppi += np.diag(p[:,action]).dot(p[:,action].reshape(501))
        Jpi = np.linalg.inv(np.eye(len(MDP[0])) - y * ppi).dot(cpi)
        for action in range(len(MDP[1])):
            Q[:,action] = c[:,action] + y * P[action].dot(Jpi)
        print(pinew)
        for action in range(len(MDP[1])):
            pinew[:, action] = np.isclose(Q[:,action], np.amin(Q,axis=1), atol=1e-10,rtol=1e-10).astype(int)
        pinew = pinew / np.sum(pinew, axis=1, keepdims = True)
        diff = (p == pinew).all() #does only one iteration
        #print(p)
        #print(pinew)
        #diff = False
        p = pinew
        i += 1
    print(i)
    return p

popt = policy_iteration(M)
print(popt)


rand.seed(42)


# For example, the optimal policy for the MDP from **Activity 1** is can be computed as follows.
# 
# ```python
# popt = policy_iteration(M)
# 
# rand.seed(42)
# 
# # Select random state, and action using the policy computed
# s = rand.randint(len(M[0]))
# a = rand.choice(len(M[1]), p=popt[s, :])
# print('Policy at state %s: %s' % (M[0][s], M[1][a]))
# 
# # Select random state, and action using the policy computed
# s = rand.randint(len(M[0]))
# a = rand.choice(len(M[1]), p=popt[s, :])
# print('Policy at state %s: %s' % (M[0][s], M[1][a]))
# 
# # Select random state, and action using the policy computed
# s = rand.randint(len(M[0]))
# a = rand.choice(len(M[1]), p=popt[s, :])
# print('Policy at state %s: %s' % (M[0][s], M[1][a]))
# ```
# 
# Output:
# ```
# Execution time: 0.089 seconds
# N. iterations: 3
# Policy at state (1, 0, 0, 2): North
# Policy at state (2, 3, 2, 2): West
# Policy at state (1, 4, 2, 0): West
# ```

# ### 4. Simulation
# 
# Finally, in this section you will check whether the theoretical computations of the cost-to-go actually correspond to the cost incurred by an agent following a policy.

# ---
# 
# #### Activity 6
# 
# Write a function `simulate` that receives, as inputs
# 
# * An MDP represented as a tuple like that of **Activity 1**;
# * A policy, represented as an `numpy` array like that of **Activity 2**;
# * An integer, corresponding to a state index
# 
# Your function should return, as an output, a float corresponding to the estimated cost-to-go associated with the provided policy at the provided state. To estimate such cost-to-go, your function should:
# 
# * Generate **100** trajectories of 10,000 steps each, starting in the provided state and following the provided policy. 
# * For each trajectory, compute the accumulated (discounted) cost. 
# * Compute the average cost over the 100 trajectories.
# 
# **Note 1:** You may find useful to import the numpy module `numpy.random`.
# 
# **Note 2:** Each simulation may take a bit of time, don't despair â˜ºï¸.
# 
# ---

# In[8]:


import numpy.random as rand
def simulate(MDP, pol, index):
    y = MDP[4]
    cost = 0
    totalCost = 0
    for i in range(100):
        state = index
        action = rand.choice(len(pol[0]), p=pol[state])
        cost = MDP[3][state, action]
        nextState = np.argmax(MDP[2][action][state,:])
        state = nextState
        totalCost += cost
        for j in range(10000):
            action = rand.choice(len(pol[0]), p=pol[state])
            nextState = np.argmax(MDP[2][action][state,:])
            state = nextState
            cost = MDP[3][state, action]
            totalCost += MDP[4] ** j * cost    
    return totalCost / 100 

s = rand.randint(1,len(M[0]))
print(simulate(M, pol, s))


# For example, we can use this function to estimate the values of some random states and compare them with those from **Activity 4**.
# 
# ```python
# 
# rand.seed(42)
# 
# # Select random state, and evaluate for the optimal policy
# s = rand.randint(len(M[0]))
# print('Cost-to-go for state %s:' % M[0][s])
# print('\tTheoretical:', Jopt[s])
# print('\tEmpirical:', simulate(M, popt, s))
# 
# # Select random state, and evaluate for the optimal policy
# s = rand.randint(len(M[0]))
# print('Cost-to-go for state %s:' % M[0][s])
# print('\tTheoretical:', Jopt[s])
# print('\tEmpirical:', simulate(M, popt, s))
# 
# # Select random state, and evaluate for the optimal policy
# s = rand.randint(len(M[0]))
# print('Cost-to-go for state %s:' % M[0][s])
# print('\tTheoretical:', Jopt[s])
# print('\tEmpirical:', simulate(M, popt, s))
# ```
# 
# Output:
# ````
# Cost-to-go for state (1, 0, 0, 2):
# 	Theoretical: [ 4.1]
# 	Empirical: 4.39338954193
# Cost-to-go for state (3, 1, 4, 1):
# 	Theoretical: [ 4.1]
# 	Empirical: 4.09638954193
# Cost-to-go for state (3, 2, 2, 2):
# 	Theoretical: [ 4.1]
# 	Empirical: 4.3816865569
# ```
