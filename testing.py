#######################################################
#######################################################
#######################################################
# The following script is for testing algorithms for 
# the Online Weight-Arrival Minimum Spanning Tree 
# problem with weight predictions.
# 



import pandas as pd
import random

#######################################################
#######################################################
###          We need some extra functions           ###
#######################################################
#######################################################

def get_vertices(edge_list):
    vertices = []
    for i in range(len(edge_list)):
        if edge_list[i][0] not in vertices:
            vertices.append(edge_list[i][0])
        if edge_list[i][1] not in vertices:
            vertices.append(edge_list[i][1])
    return vertices

def adjacency_matrix(edge_list):
    max = -1
    vertices = get_vertices(edge_list)
    for vertex in vertices:
        if max <= vertex:
            max = vertex
    size = max
    ad_matrix = [[0 for _ in range(size)] for _ in range(size)]
    for i in range(len(edge_list)):
        source = int(edge_list[i][0])
        target = int(edge_list[i][1])
        ad_matrix[source-1][target-1] = 1
        ad_matrix[target-1][source-1] = 1
    return ad_matrix

def adjacency_matrix_edge_cut(edge_list,tree):
    max = -1
    vertices = get_vertices(edge_list)
    for vertex in vertices:
        if max <= vertex:
            max = vertex
    size = max
    ad_matrix = [[0 for _ in range(size)] for _ in range(size)]
    for i in range(len(tree)):
        source = int(tree[i][0])
        target = int(tree[i][1])
        ad_matrix[source-1][target-1] = 1
        ad_matrix[target-1][source-1] = 1
    return ad_matrix

def mergeSort(weights,indices):
    if len(weights) > 1:
        mid = len(weights)//2
        left_weights = weights[:mid]
        left_indices = indices[:mid]
        right_weights = weights[mid:]
        right_indices = indices[mid:]

        left_weights, left_indices = mergeSort(left_weights,left_indices)
        right_weights, right_indices = mergeSort(right_weights,right_indices)

        i = 0
        j = 0
        k = 0
        while (i < len(left_weights)) and (j < len(right_weights)):
            if left_weights[i] <= right_weights[j]:
                weights[k] = left_weights[i]
                indices[k] = left_indices[i]
                i += 1
            else: 
                weights[k] = right_weights[j]
                indices[k] = right_indices[j]
                j += 1
            k += 1
        
        while i < len(left_weights):
            weights[k] = left_weights[i]
            indices[k] = left_indices[i]
            i += 1
            k += 1

        while j < len(right_weights):
            weights[k] = right_weights[j]
            indices[k] = right_indices[j]
            j += 1
            k += 1
    
    return weights, indices

def sort_edges(edge_list,w):
    weights = []
    indices = []
    for i in range(len(edge_list)):
        weights.append(edge_list[i][w])
        indices.append(i)
    _, indices = mergeSort(weights,indices)
    sorted_edge_list = []
    for index in indices:
        sorted_edge_list.append([int(edge_list[index][0]),int(edge_list[index][1]),edge_list[index][2],edge_list[index][3]]) 
    return sorted_edge_list


# DFS returns a bool and a history (bool, history), where
# the bool states whether or not there is a cycle 
# and the history records one cycle.
# We will mainly use this in GFtP, for checking
# cycles constructed by newly revealed edges outside T.
# Since there can only ever be one cycle in T,
# we are sure to find this using this implementation of DFS.
def DFS(ad_matrix):
    history = []
    visited = [False for _ in range(len(ad_matrix))]
    for i in range(len(ad_matrix)):
        if visited[i] == False:
            history.append(i+1)
            bit,history = DFS_sub(ad_matrix,i,visited,-1,history)
            if bit == True:
                return True, history
    return False, history

def DFS_sub(ad_matrix, start, visited, parent,history):
    visited[start] = True
    for i in range(len(ad_matrix)):
        if ad_matrix[start][i] == 1:
            if visited[i] == False:
                history.append(i+1)
                bit, history = DFS_sub(ad_matrix,i,visited,start,history)
                if bit == True:
                    return True, history
            elif parent != i:
                history.append(i+1)
                # The following five lines are added in case of cycles like [1,2,3,4,2], where the first 1 is superflouos
                start_index = -1
                for j in range(len(history)-1):
                    if history[j] == i + 1:
                        start_index = j
                        break
                sub_history = [history[l] for l in range(start_index,len(history))] 
                return True, sub_history
    history.pop()
    return False, history

def kruskal(edge_list,ad_matrix,weights): 
    assert weights in {2,3}
    edge_list_sorted = sort_edges(edge_list,weights)
    index = 0
    n = len(get_vertices(edge_list))
    num_edges_accepted = 0
    mst = []
    tree_ad_matrix = [[0 for _ in range(len(ad_matrix))] for _ in range(len(ad_matrix))]
    while num_edges_accepted < n-1:
        next_edge = edge_list_sorted[index]
        tree_ad_matrix[int(next_edge[0])-1][int(next_edge[1])-1] = 1
        tree_ad_matrix[int(next_edge[1])-1][int(next_edge[0])-1] = 1
        bit, _ = DFS(tree_ad_matrix)
        if not bit:
            mst.append(next_edge)
            num_edges_accepted += 1
        else:
            tree_ad_matrix[int(next_edge[0])-1][int(next_edge[1])-1] = 0
            tree_ad_matrix[int(next_edge[1])-1][int(next_edge[0])-1] = 0            
        index += 1
    return mst, tree_ad_matrix


#######################################################
#######################################################
###              Checking the instance              ###
#######################################################
#######################################################

# is_connected is basically another (simpler) implementation of DFS
# starting in only one vertex.
def is_connected(ad_matrix):
    visited = [0 for _ in range(len(ad_matrix))]
    for i in range(len(ad_matrix)):
        biti = 0
        for j in range(len(ad_matrix)):
            if ad_matrix[i][j] == 1:
                biti = 1
        if biti == 0:
            visited[i] = 1
    ### Only one initial vertex - if connected, this will reach all other edges
    visited = is_connected_sub(ad_matrix,0,visited,1)
    for i in range(len(visited)):
        if visited[i] == 0:
            return False
    return True

def is_connected_sub(ad_matrix,start,visited,num):
    visited[start] = num
    for i in range(len(ad_matrix)):
        if ad_matrix[start][i] == 1:
            if visited[i] == 0:
                visited = is_connected_sub(ad_matrix,i,visited,num)
    return visited

#######################################################
#######################################################
###                  Code for OPT                   ###
#######################################################
#######################################################

def OPT(edge_list,ad_matrix):
    return kruskal(edge_list,ad_matrix,2)

#######################################################
#######################################################
###                  Code for FtP                   ###
#######################################################
#######################################################

def FtP(edge_list,ad_matrix):
    return kruskal(edge_list,ad_matrix,3)

#######################################################
#######################################################
###                  Code for GFtP                  ###
#######################################################
#######################################################

### This implementation assumes that the edges in edge_list
### are ordered such that the algorithm recieves the edges
### one at a time, from top to bottom as ordered in edge_list.

def is_in_edge_list(row,edge_list):
    index = -1
    for i in range(len(edge_list)):
        if (int(edge_list[i][0]) == int(row[0])) and (int(edge_list[i][1]) == int(row[1])) or (int(edge_list[i][0]) == int(row[1]) and int(edge_list[i][1]) == int(row[0])) :
            index = i
    return index

def create_C_prime(tree,unseen,cycle):
    C_prime = []
    for i in range(len(tree)): #Iterate through rows in the tree
        f = tree[i][0]
        t = tree[i][1]
        for j in range(len(cycle)-1):
            # If edge is contained in the cycle, and the edge is still unseen, we add it to C'
            if (((cycle[j] == f) and (cycle[j+1] == t)) or ((cycle[j] == t) and (cycle[j+1] == f))) and (is_in_edge_list(tree[i],unseen) != -1): 
                C_prime.append([int(tree[i][0]),int(tree[i][1]),tree[i][2],tree[i][3]])
    return C_prime

def GFtP(edge_list,ad_matrix):
    n = len(get_vertices(edge_list))
    T, tree_ad_matrix = FtP(edge_list,ad_matrix) # Initial guess
    U = edge_list.copy() #Collection of unseen edges
    num_revealed = 0 # A counter for number of revealed edges
    num_accepted = 0 # A counter for number of accepted edges
    while len(U) != 0 and num_accepted < n-1: # While somes edges are still unseenp
        U.pop(0) # The topmost edge in U is no longer unseen
        e_next = edge_list[num_revealed]
        index = is_in_edge_list(e_next,T) #Is the next edge contained in T?
        
        ### if the index is different from -1, then the next
        ### edge is in T, in which case we accept it. We simply
        ### skip this in this implementation.
        if index == -1: # next edge is not in T
            T.append([int(e_next[0]),int(e_next[1]),e_next[2],e_next[3]])
            tree_ad_matrix[int(e_next[0])-1][int(e_next[1])-1] = 1
            tree_ad_matrix[int(e_next[1])-1][int(e_next[0])-1] = 1
            _, cycle = DFS(tree_ad_matrix)
            unseen_edges_in_cycle = create_C_prime(T,U,cycle)
            if len(unseen_edges_in_cycle) != 0:
                e_max = float('-inf')
                drop_index = -1
                for i in range(len(unseen_edges_in_cycle)):
                    if e_max <= unseen_edges_in_cycle[i][3]:
                        e_max = unseen_edges_in_cycle[i][3]
                        drop_index = is_in_edge_list(unseen_edges_in_cycle[i],T)
                if e_next[2] <= e_max:
                    tree_ad_matrix[int(T[drop_index][0])-1][int(T[drop_index][1])-1] = 0
                    tree_ad_matrix[int(T[drop_index][1])-1][int(T[drop_index][0])-1] = 0
                    T.pop(drop_index)
                    num_accepted += 1
                else:
                    tree_ad_matrix[int(e_next[0])-1][int(e_next[1])-1] = 0
                    tree_ad_matrix[int(e_next[1])-1][int(e_next[0])-1] = 0
                    T.pop(len(T)-1)
            else:
                tree_ad_matrix[int(e_next[0])-1][int(e_next[1])-1] = 0
                tree_ad_matrix[int(e_next[1])-1][int(e_next[0])-1] = 0
                T.pop(len(T)-1)
        else:
            # In this case, the just-revealed edge is contained in T, and we accept it.
            num_accepted += 1
        num_revealed += 1
    return T


#######################################################
#######################################################
###                 Third algorithm                 ###
#######################################################
#######################################################

def edge_cut(edge_list,tree,edge):
    index = is_in_edge_list(edge,tree)
    max = -1
    vertices = get_vertices(edge_list)
    for vertex in vertices:
        if max <= vertex:
            max = vertex
    size = max
    tree_internal = tree.copy()
    tree_internal.pop(index)

    v_from = int(edge[0]) - 1 # These will be used as indices, and hence we subtract 1
    v_to = int(edge[1]) - 1

    connected_component_v_from = [0 for _ in range(size)]
    connected_component_v_to = [0 for _ in range(size)]
    tree_internal_matrix = adjacency_matrix_edge_cut(edge_list,tree_internal)

    connected_component_v_from = is_connected_sub(tree_internal_matrix,v_from,connected_component_v_from,1)
    connected_component_v_to = is_connected_sub(tree_internal_matrix,v_to,connected_component_v_to,1)
    U = []
    V = []
    for i in range(size):
        if connected_component_v_from[i] == 1:
            U.append(i+1)
        else:
            V.append(i+1)
    return U,V

def third_algorithm(edge_list,ad_matrix):
    n = len(get_vertices(edge_list)) # Number of vertices
    T, tree_ad_matrix = FtP(edge_list,ad_matrix) # FtPs solution
    U = edge_list.copy() # Collection of unseen edges
    mark = [float('inf') for _ in range(len(edge_list))] #Mark of the edges.
    num_revealed = 0
    num_accepted = 0

    while len(U) != 0 and num_accepted < n-1:
        U.pop(0)
        if mark[num_revealed] == float('-inf'):
            mark[num_revealed] = edge_list[num_revealed][2]
        e_next = edge_list[num_revealed]
        index = is_in_edge_list(e_next,T) # Is the next edge in T. If not, then index = -1, otherwise, index is the index in T in which the next edge es located
        if index == -1:
            T.append([int(e_next[0]),int(e_next[1]),e_next[2],e_next[3]])
            tree_ad_matrix[int(e_next[0])-1][int(e_next[1])-1] = 1
            tree_ad_matrix[int(e_next[1])-1][int(e_next[0])-1] = 1
            _, cycle = DFS(tree_ad_matrix)
            unseen_edges_in_cycle = create_C_prime(T,U,cycle)
            if len(unseen_edges_in_cycle) != 0:
                e_max = float('-inf')
                drop_index = -1
                for i in range(len(unseen_edges_in_cycle)):
                    if e_max <= unseen_edges_in_cycle[i][3]:
                        e_max = unseen_edges_in_cycle[i][3]
                        drop_index = is_in_edge_list(unseen_edges_in_cycle[i],T)
                if e_next[2] <= e_max:
                    tree_ad_matrix[int(T[drop_index][0])-1][int(T[drop_index][1])-1] = 0
                    tree_ad_matrix[int(T[drop_index][1])-1][int(T[drop_index][0])-1] = 0
                    T.pop(drop_index)
                    num_accepted += 1
                else:
                    tree_ad_matrix[int(e_next[0])-1][int(e_next[1])-1] = 0
                    tree_ad_matrix[int(e_next[1])-1][int(e_next[0])-1] = 0
                    T.pop(len(T)-1)
            else:
                tree_ad_matrix[int(e_next[0])-1][int(e_next[1])-1] = 0
                tree_ad_matrix[int(e_next[1])-1][int(e_next[0])-1] = 0
                T.pop(len(T)-1)
        else:
            if e_next[2] > e_next[3]:
                component_1,component_2 = edge_cut(edge_list,T,e_next)
                uv_edges = []
                for i in range(len(U)):
                    edge = U[i]
                    v_from = edge[0]
                    v_to = edge[1]
                    if ((v_from in component_1) and (v_to in component_2)) or ((v_from in component_2) and (v_to in component_1)):
                        uv_edges.append([edge[0],edge[1],edge[2],edge[3]])
                if len(uv_edges) != 0:
                    e_min = float('inf')
                    e_min_index = -1
                    for j in range(len(uv_edges)):
                        pred_weight = uv_edges[j][3]
                        if pred_weight <= e_min:
                            e_min = pred_weight
                            e_min_index = j
                    if (e_min < mark[num_revealed]) and (e_min < e_next[2]):
                        tree_ad_matrix[int(T[index][0])-1][int(T[index][1])-1] = 0
                        tree_ad_matrix[int(T[index][1])-1][int(T[index][0])-1] = 0
                        T.pop(index)

                        tree_ad_matrix[int(uv_edges[e_min_index][0])-1][int(uv_edges[e_min_index][1])-1] = 1
                        tree_ad_matrix[int(uv_edges[e_min_index][1])-1][int(uv_edges[e_min_index][0])-1] = 1
                        T.append([int(uv_edges[e_min_index][0]),int(uv_edges[e_min_index][1]),uv_edges[e_min_index][2],uv_edges[e_min_index][3]])
                        e_min_index_in_edge_list = is_in_edge_list(uv_edges[e_min_index],edge_list)
                        mark[e_min_index_in_edge_list] = min(mark[num_revealed],e_next[2])
                    else: 
                        num_accepted += 1
                else:
                    num_accepted += 1
            else:
                num_accepted += 1
        num_revealed += 1
    return T


#######################################################
#######################################################
###            Code for Measures and Cost           ###
#######################################################
#######################################################

def compute_eta(edge_list):
    num_vertices = len(get_vertices(edge_list))
    pred_errors = [0 for _ in range(len(edge_list))]
    for i in range(len(edge_list)):
        pred_errors[i] = abs(edge_list[i][2] - edge_list[i][3])
    pred_errors.sort()
    error = 0
    for i in range(num_vertices-1):
        error += pred_errors[len(pred_errors)-1-i]
    return error

def cost(edge_list):
    cost = 0
    for i in range(len(edge_list)):
        cost += edge_list[i][2]
    return cost


#######################################################
#######################################################
###               Getting permutations              ###
#######################################################
#######################################################

def random_perm(k: int):
    perm = [-1 for _ in range(k)]
    indices_remaining = [i for i in range(k)]

    for i in range(k):
        num = random.choice(indices_remaining)
        perm[i] = num
        indices_remaining.remove(num)
    return perm

def shuffle(edge_list,permutation):
    new_edge_list = []
    for i in range(len(edge_list)):
        row = edge_list[permutation[i]]
        new_edge_list.append([int(row[0]),int(row[1]),row[2],row[3]])
    return new_edge_list

def perm_as_string(permutation : list):
    st = ''
    st = st + '['
    for i in range(len(permutation)):
        st = st + str(permutation[i])
        if not i == len(permutation)-1:
            st = st + ', '
    st = st + ']'
    return st
        

#######################################################
#######################################################
###            Erdös-Rényi-Gilbert model            ###
#######################################################
#######################################################

def erdos_renyi_gilbert(n,p,type):
    assert 0 <= p <= 1
    graph = []
    for i in range(n):
        for j in range(i+1,n):
            unif_num = random.uniform(0,1)
            if unif_num <= p:
                true_weight = -1
                pred_weight = -1
                while not ((0 < true_weight < 1) and (0 < pred_weight < 1)):
                    match type:
                        case 'normalWeights_normalPreds_independent':
                            true_weight = random.normalvariate(0.5,0.15)
                            pred_weight = random.normalvariate(0.5,0.15)
                        case 'normalWeights_normalPreds_not_indepndent':
                            true_weight = random.normalvariate(0.5,0.15)
                            pred_weight = random.normalvariate(true_weight,0.15)
                        case 'normalWeights_unifPreds':
                            true_weight = random.normalvariate(0.5,0.15)
                            pred_weight = random.uniform(0,1)
                        case 'unifWeights_normalPreds_independent':
                            true_weight = random.uniform(0,1)
                            pred_weight = random.normalvariate(0.5,0.15)
                        case 'unifWeights_normalPreds_not_indepndent':
                            true_weight = random.uniform(0,1)
                            pred_weight = random.normalvariate(true_weight,0.15)
                        case 'unifWeights_unifPreds':
                            true_weight = random.uniform(0,1)
                            pred_weight = random.uniform(0,1)
                graph.append([i+1,j+1,true_weight,pred_weight])
    return graph


#######################################################
#######################################################
###         Get largest connected component         ###
#######################################################
#######################################################

def get_largest_connected_component(graph,graph_matrix):
    visited = [0 for _ in range(len(graph_matrix))]
    num = 1
    visited = is_connected_sub(graph_matrix,0,visited,num)
    num_components = 1
    for i in range(len(visited)):
        if visited[i] == 0:
            num += 1
            num_components += 1
            visited = is_connected_sub(graph_matrix,i,visited,num)
    num_occurences = [0 for _ in range(num_components)]
    for obs in range(len(visited)):
        num_occurences[visited[obs]-1] += 1
    max = -1
    index = -1
    for l in range(len(num_occurences)):
        if num_occurences[l] > max:
            max = num_occurences[l]
            index = l
    vertex = -1
    for o in range(len(visited)):
        if visited[o] == index+1:
            vertex = o
            break
    visited = [0 for _ in range(len(graph_matrix))]
    visited = is_connected_sub(graph_matrix,vertex,visited,1)
    connected_subgraph = graph.copy()
    for row in range(len(graph)-1,-1,-1):
        if visited[graph[row][0]-1] == 0:
            connected_subgraph.pop(row)
    graph_new = connected_subgraph.copy()
    graph_matrix_new = adjacency_matrix(graph_new)
    return graph_new, graph_matrix_new


#######################################################
#######################################################
###                       Main                      ###
#######################################################
#######################################################

synth_data = {'normalWeights_normalPreds_independent','normalWeights_normalPreds_not_indepndent','normalWeights_unifPreds','unifWeights_normalPreds_independent','unifWeights_normalPreds_not_indepndent','unifWeights_unifPreds'}

for type in synth_data:
    delta = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    num_tests = 1
    for p in range(len(delta)):

        print('------------------------------------------')
        print('Now computing samples for p =',delta[p])
        print()

        for i in range(1000):

            print('i:',i,end='\r')

            # Generating random graph
            graph = erdos_renyi_gilbert(25,delta[p],type)

            # Saving random graph
            graph_pandas = pd.DataFrame(
                [
                edge for edge in graph
                ],
                columns = ['From','To','True weight','Predicted weight']
            )
            graph_pandas.to_csv('results/' + type + '/test_' + str(num_tests) + '.csv',index = False)

            # Creating adjacency matrix
            graph_matrix = adjacency_matrix(graph)

            # Checking for connectedness
            is_it_connected = True
            if not is_connected(graph_matrix):
                # If not connected, we run on largest connected component
                graph, graph_matrix = get_largest_connected_component(graph,graph_matrix)
                is_it_connected = False

            # Running all four algorithms on edge list permuted by uniformly randomly chosen permutation
            perm = random_perm(len(graph))
            graph_random = shuffle(graph,perm)
            opt, _ = OPT(graph_random,graph_matrix)
            ftp, _ = FtP(graph_random,graph_matrix)
            gftp = GFtP(graph_random,graph_matrix)
            third_alg = third_algorithm(graph_random,graph_matrix)
            c_opt = cost(opt)
            c_ftp = cost(ftp)
            c_gftp = cost(gftp)
            c_third_alg = cost(third_alg)

            # Computing eta and epsilon
            eta = compute_eta(graph)
            epsilon = eta / c_opt

            # Saving a comprehensive summary of the test instance, ensuring reproducability
            with open('results/' + type + '/test_' + str(num_tests) + '.txt', 'w') as f:
                f.write('Summary of test number: ' + str(num_tests) + '\n')
                f.write('---------------------------------------------------\n')
                f.write('n: ' + str(25) + '\n')
                f.write('p:' + str(delta[p]) + '\n')
                if is_it_connected:
                    f.write('Graph is connected\n')
                else:
                    f.write('Graph is not connected\n')
                f.write('Eta: ' + str(eta) + '\n')
                f.write('Epsilon: ' + str(epsilon) + '\n'),
                f.write('OPTs cost: ' + str(c_opt) + '\n')
                f.write('FTPs cost: ' + str(c_ftp) + '\n')
                f.write('GFtPs cost: ' + str(c_gftp) + '\n')
                f.write('Third algorithms cost: ' + str(c_third_alg) + '\n')
                f.write('Permutation: ' + perm_as_string(perm) + '\n')
                f.write('---------------------------------------------------\n')
            num_tests += 1
                