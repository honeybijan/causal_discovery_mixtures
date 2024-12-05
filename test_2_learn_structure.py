from data_generating import *
from utils import *
from learn_structure import *

# Test 2: for one "y" graph
def latex_adj_matrix(correct_matrix, adj_matrix):
    result= ""
    i = 0
    for row in adj_matrix:
        row_str_list= ["$V_{}$".format(i)]
        j= 0
        for value in row:
            if correct_matrix[i, j] == 1:
                color = "blue"
            else:
                color = "red"
            if value == 0:
                color = "black"
            row_str_list.append("\color{{{}}}{{{}}}".format(color, value))
            j +=1
        result += " & ".join(row_str_list)
        result += "\\\\\n"
        i += 1
    return result

def generate_tikz_graph(n, correct_adj, recovered_adj_percentage):
    result = ""
    for i in range(n):
        result += "\\node[state] (V{}) {{$V_{}$}};\n".format(i, i)
    for i in range(n):
        for j in range(i):
            if correct_adj[i, j] == 1:
                color = "blue"
                bend = "right = 0"
            else:
                color = "red"
                if (i==0 or j ==0) and i - j > 1:
                    bend = "right = 30"
                elif (i==1 or j ==1) and i - j > 1:
                    bend = "left = 30"
                elif i - j > 1:
                    bend = "right = 50"
                else:
                    bend = "right = 0"
            opc = recovered_adj_percentage[i,j]
            if opc > 0:
                result += "\path[ultra thick] (V{}) edge[-, color={}, opacity = {}, bend {}] (V{});\n".format(i, color, opc, bend, j)
    return result


n=7

correct_adj_matrix = np.zeros((n, n))
correct_adj_matrix[0,2] = 1
correct_adj_matrix[1,2] = 1
for i in range(2,n-1):
    correct_adj_matrix[i, i+1] = 1
correct_adj_matrix = correct_adj_matrix + np.transpose(correct_adj_matrix)

#eps = .01
#print("With epsilon = {}".format(eps))
attempts = 100
results = np.zeros((n,n))
for i in range(attempts):
    data = generate_data_y_graph(n, 10000)
    if i % 10 == 0:
        print("Checkpoint: Iteration {}".format(i))
    results += find_adj_structure_hyp_test(n, data, k=2, p = .0005)#, eps = eps)
results = results / attempts

print(generate_tikz_graph(n, correct_adj_matrix, results))
print(latex_adj_matrix(correct_adj_matrix, results))