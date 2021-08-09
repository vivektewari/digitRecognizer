from funcs import count_parameters,get_dict_from_class
from config import *
model = model(**get_dict_from_class(model_param))
#checkpoint = torch.load(pre_trained_model)['model_state_dict']
count_parameters(model)