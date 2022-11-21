path = "./datasets/or1W/train"

import os
import numpy as np

path_list = os.listdir(path)

new_list1 = []
new_list2 = []

for x in path_list:
    list1 = x.split("_")[0]
    list2 = x.split("_")[1]
    new_list1.append(list1)
    new_list2.append(list2)

new_list1 = np.array(new_list1)
new_list2 = np.array(new_list2)
for t in new_list2:
    a = np.where(new_list2 == t)[0]
    if len(a) > 1:
        print(a)
        print(t)
        print(" ")