"""
    Assume input tensor is of the form:
    tensor = [outlook,temp,humidity,windy,play]
    here play is the target variable (class)
    remaining four are explanatory variables

"""
import torch
import math

"""Calculate the entropy of the entire dataset"""
# input:tensor
# output:int/float
def get_entropy_of_dataset(tensor:torch.Tensor):
    
    dataset = tensor[:,-1]
    a = len([x for x in dataset if x == 1])
    b = len([x for x in dataset if x == 0])
    if a == 0 or b == 0:
        return 0
    
    entropy = -(a/(a+b)) * math.log2(a/(a+b)) - (b/(a+b)) * math.log2(b/(a+b))
    return entropy

"""Return avg_info of the attribute provided as parameter"""
# input:tensor,attribute number 
# output:int/float
def get_avg_info_of_attribute(tensor:torch.Tensor, attribute:int):
    
    values = tensor[:,attribute]
    total = len(values)
    unique_values = torch.unique(values)
    avg_info = 0

    for value in unique_values:
        subset = tensor[values == value]
        subset_len = len(subset)

        if subset_len == 0:
            continue

        info_value = (subset_len / total) * get_entropy_of_dataset(subset)
        avg_info += info_value

    return avg_info


"""Return Information Gain of the attribute provided as parameter"""
# input:tensor,attribute number
# output:int/float
def get_information_gain(tensor:torch.Tensor, attribute:int):
    
    information_gain = get_entropy_of_dataset(tensor) - get_avg_info_of_attribute(tensor,attribute)
    return information_gain


# input: tensor
# output: ({dict},int)
def get_selected_attribute(tensor:torch.Tensor):
    """
    Return a tuple with the first element as a dictionary which has IG of all columns
    and the second element as an integer representing attribute number of selected attribute

    example : ({0: 0.123, 1: 0.768, 2: 1.23} , 2)
    """
    
    number_of_columns = tensor.shape[1] - 1
    dict = {}
    for i in range(number_of_columns):
        dict[i] = get_information_gain(tensor,i)
    result = max(dict, key=dict.get)
    return dict , result
    








