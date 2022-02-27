#!/usr/bin/env python
# coding: utf-8

# In[139]:


import sys
import pennylane as qml
from pennylane import numpy as np


# In[140]:


def hamiltonian_coeffs_and_obs(graph):
    """Creates an ordered list of coefficients and observables used to construct
    the UDMIS Hamiltonian.
    Args:
        - graph (list((float, float))): A list of x,y coordinates. e.g. graph = [(1.0, 1.1), (4.5, 3.1)]
    Returns:
        - coeffs (list): List of coefficients for elementary parts of the UDMIS Hamiltonian
        - obs (list(qml.ops)): List of qml.ops
    """

    num_vertices = len(graph)
    E, num_edges = edges(graph)
    u = 1.35
    obs = []
    coeffs = []

    # QHACK #
    # QHACK #
    for i in range(num_vertices):
        coeffs.append(-1/2.)
        obs.append(qml.PauliZ(i))

        coeffs.append(-1/2.)
        obs.append(qml.Identity(i))
        
    #Coeff gets divided by two (half in i,j, half in j,i) and then by four (each n_i divided by two)
    for vertex_i in range(num_vertices-1):
        for vertex_j in range(vertex_i+1,num_vertices):
            if   E[vertex_i, vertex_j] != 0:
                coeffs.append(+u/4)
                obs.append(qml.PauliZ(vertex_i)@qml.PauliZ(vertex_j))

                coeffs.append(+u/4)
                obs.append(qml.PauliZ(vertex_i))

                coeffs.append(+u/4)
                obs.append(qml.PauliZ(vertex_j))

                coeffs.append(+u/4)
                obs.append(qml.Identity(vertex_i) @ qml.Identity(vertex_j))
                
            
        
        
        
        
    #H = qml.Hamiltonian(coeffs, obs)    

    # create the Hamiltonian coeffs and obs variables here

    # QHACK #

    return coeffs, obs
    #print(H)
    #print(obs)
    #print(coeffs)


# In[141]:


def edges(graph):
    """Creates a matrix of bools that are interpreted as the existence/non-existence (True/False)
    of edges between vertices (i,j).
    Args:
        - graph (list((float, float))): A list of x,y coordinates. e.g. graph = [(1.0, 1.1), (4.5, 3.1)]
    Returns:
        - num_edges (int): The total number of edges in the graph
        - E (np.ndarray): A Matrix of edges
    """

    # DO NOT MODIFY anything in this code block
    num_vertices = len(graph)
    E = np.zeros((num_vertices, num_vertices), dtype=bool)
    for vertex_i in range(num_vertices - 1):
        xi, yi = graph[vertex_i]  # coordinates

        for vertex_j in range(vertex_i + 1, num_vertices):
            xj, yj = graph[vertex_j]  # coordinates
            dij = np.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)
            E[vertex_i, vertex_j] = 1 if dij <= 1.0 else 0

    return E, np.sum(E, axis=(0, 1))


# In[255]:


def variational_circuit(params, num_vertices):
    
    #def U_B(beta,alpha):
     #   for wire in range(num_vertices):
      #      qml.RX(2 * beta*alpha, wires=wire)


# unitary operator U_C with parameter gamma
    
    for i in range(num_vertices):
        qml.RY(params[i][0],wires=i)
    for j in range(num_vertices-3):       
        for i in range(0,num_vertices-2,2):
            qml.CZ(wires=[i,i+1])
            qml.RY(params[i][1],wires=i)
            qml.RY(params[i+1][2],wires=i+1)
            #qml.Hadamard(wires=j)
            #qml.Hadamard(wires=j+1)
            #qml.Hadamard(wires=j+2)
            #qml.Hadamard(wires=j+3)
            #qml.RZ(params[i+1][2],wires=j+1)
            #qml.CNOT(wires=[j,j+1])
            #qml.CNOT(wires=j+1,j+2)
            #qml.RZ(wires=j+2)
            #qml.CNOT(wires=j+1,j+2)
            
            
        for i in range(1,num_vertices-1,2):
            qml.CZ(wires=[i,i+1])
            qml.RY(params[i][3],wires=i)
            qml.RY(params[i+1][4],wires=i+1)
            
            
            
        qml.Hadamard(wires=j)
        qml.Hadamard(wires=j+1)
        qml.Hadamard(wires=j+2)
            #qml.RZ(params[i+1][2],wir   
            
            
            
    
            
            
   
    """A variational circuit.
    Args:
        - params (np.ndarray): your variational parameters
        - num_vertices (int): The number of vertices in the graph. Also used for number of wires.
    """

    # QHACK #
    #for i in range(num_vertices):
     #   qml.RX(params[i],wires=i)
        
        #qml.CNOT(wires=[i,i+1])
      #  qml.RY(2*params[i],wires=i)
        
    #qml.RY(params[num_vertices-1],wires=num_vertices-1)
        
    


# unitary operator U_C with parameter gamma
    
    

    # create your variational circuit here

    # QHACK #


# In[283]:


def train_circuit(num_vertices, H):
    """Trains a quantum circuit to learn the ground state of the UDMIS Hamiltonian.
    Args:
        - num_vertices (int): The number of vertices/wires in the graph
        - H (qml.Hamiltonian): The result of qml.Hamiltonian(coeffs, obs)
    Returns:
        - E / num_vertices (float): The ground state energy density.
    """

    dev = qml.device("default.qubit", wires=num_vertices)

    @qml.qnode(dev)
    def cost(params):
        """The energy expectation value of a Hamiltonian"""
        variational_circuit(params, num_vertices)
        return qml.expval(H)

    # QHACK #

    # define your trainable parameters and optimizer here
    # change the number of training iterations, `epochs`, if you want to
    # just be aware of the 80s time limit!
    #p=
    #for i in range(2):
    #    p.append(4.45)
     #   p.append(2.22)
     #   p.append(.033)
        
    init_params = np.zeros((num_vertices,5))
    opt= qml.AdagradOptimizer(.33)
    params=init_params
    epochs = 790

    # QHACK #

    for i in range(epochs):
        params, E = opt.step_and_cost(cost, params)
        #print(params)
    #for i in range(epochs):
     #   params, E = opt.step_and_cost(cost, params)
      #  print(params)
        
        
    return E / float(num_vertices)
    #print( qml.qaoa.cost.max_independent_set(graph))


# In[286]:


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block
    #inputs =  np.array([1.971141343837512,1.3175151958757692,1.7743411711379486,0.19884568690295978,1.2411294078770752,4.077451769506551,1.1991102459020841,3.7875893202521675,3.0946479758878165,2.389909455806816,3.2889545392433455,0.5826862970005925])
    inputs =  np.array([4.676495520782796,2.1112909562162763,1.5485936307414583,7.361431893916296,6.832710422152228,5.513534564521993,2.611816925831103,6.054858463854695,5.222428162220638,4.552144597036618,9.630745026546073,7.387891596606659])

    num_vertices = int(len(inputs) / 2)
    x = inputs[:num_vertices]
    y = inputs[num_vertices:]
    graph = []
    for n in range(num_vertices):
        graph.append((x[n].item(), y[n].item()))

    coeffs, obs = hamiltonian_coeffs_and_obs(graph)
    H = qml.Hamiltonian(coeffs, obs)

    energy_density = train_circuit(num_vertices, H)
    print(f"{energy_density:.6f}")


# In[ ]:


## 


# In[ ]:




