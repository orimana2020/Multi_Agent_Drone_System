import numpy as np


def func(args_list=None):
    args = ['a','b','c']
    if args_list:
        args += args_list
           
    print(args)    
func()


# print(['a','v'] + [np.array([[1,2,3],[4,5,6]])])