"""
    Assume input tensor is of the form:
    tensor = [outlook,temp,humidity,windy,play]
    here play is the target variable (class)
    remaining four are explanatory variables

"""
import torch

"""Calculate the entropy of the entire dataset"""
# input:tensor
# output:int/float
def get_entropy_of_dataset(tensor:torch.Tensor):
    # TODO
    play=tensor[:, -1]
    unique_classes, class_counts=torch.unique(play, return_counts=True)
    prob=class_counts.float()/len(play)
    S=-torch.sum(prob*torch.log2(prob))
    return S.item()
    pass

"""Return avg_info of the attribute provided as parameter"""
# input:tensor,attribute number 
# output:int/float
def get_avg_info_of_attribute(tensor: torch.Tensor, attribute: int):
    attribute_values = tensor[:, attribute]
    unique_values, value_counts = torch.unique(attribute_values, return_counts=True)
    avg_info = 0

    for value, count in zip(unique_values, value_counts):
        value_mask = (attribute_values == value)
        value_occurrences = torch.sum(value_mask)
        
        if value_occurrences > 0:
            value_subset = tensor[value_mask]
            value_subset_entropy = get_entropy_of_dataset(value_subset)
            avg_info=avg_info+(value_occurrences / len(attribute_values)) * value_subset_entropy

    return avg_info.item()


    pass

"""Return Information Gain of the attribute provided as parameter"""
# input:tensor,attribute number
# output:int/float
def get_information_gain(tensor:torch.Tensor, attribute:int):
    # TODO
    entropy_of_dataset=get_entropy_of_dataset(tensor)
    avg_info=get_avg_info_of_attribute(tensor, attribute)
    info_gain=entropy_of_dataset-avg_info
    return info_gain
    pass

# input: tensor
# output: ({dict},int)
def get_selected_attribute(tensor:torch.Tensor):
    """
    Return a tuple with the first element as a dictionary which has IG of all columns
    and the second element as an integer representing attribute number of selected attribute

    example : ({0: 0.123, 1: 0.768, 2: 1.23} , 2)
    """
    # TODO
    num_attr = tensor.shape[1] - 1  
    attr_info_gains = {}
    
    for attr in range(num_attr):
        info_gain = get_information_gain(tensor, attr)
        attr_info_gains[attr] = info_gain
    
    selected_attribute = max(attr_info_gains, key=attr_info_gains.get)
    return attr_info_gains, selected_attribute
    pass