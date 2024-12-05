import numpy as np

def positive_negative_to_binary(v):
    return (v + 1)/2


def binary_var_from_parents(datapoints, list_of_parents, noise_probability = .1):
    list_of_parents = np.array(list_of_parents)
    number_of_parents = list_of_parents.shape[0]
    sum_of_parents = np.zeros(datapoints)
    for parent in list_of_parents:
        sum_of_parents += parent
    ps = (sum_of_parents + 1) / (number_of_parents + 2)
    result = np.random.binomial(1,p=ps)
    return result

def generate_data_y_graph(n, datapoints):
  U = binary_var_from_parents(datapoints, [])# np.random.binomial(1, .5, datapoints)
  V0 = binary_var_from_parents(datapoints, [U])
  V1 = binary_var_from_parents(datapoints, [U])
  V2 = binary_var_from_parents(datapoints, [V0, V1, U])
  list_of_Vs = [V0, V1, V2]
  for i in range(n-3):
    list_of_Vs.append(binary_var_from_parents(datapoints, [list_of_Vs[-1], U]))
  return np.transpose(list_of_Vs)

def generate_data_connected(datapoints):
  U = binary_var_from_parents(datapoints, [])
  V0 = binary_var_from_parents(datapoints, [U])
  V1 = binary_var_from_parents(datapoints, [U, V0])
  V2 = binary_var_from_parents(datapoints, [U, V1])
  V3 = binary_var_from_parents(datapoints, [U, V2])
  list_of_Vs = [V0, V1, V2, V3]
  return np.transpose(list_of_Vs)

def generate_data_not_connected(datapoints):
  U = binary_var_from_parents(datapoints, [])
  V0 = binary_var_from_parents(datapoints, [U])
  V1 = binary_var_from_parents(datapoints, [U, V0])
  V2 = binary_var_from_parents(datapoints, [U])
  V3 = binary_var_from_parents(datapoints, [U, V2])
  list_of_Vs = [V0, V1, V2, V3]
  return np.transpose(list_of_Vs)
