import os
import glob
import matplotlib.pyplot as plt
import numpy as np

plt.figure()
plt.suptitle('train data')


data_dir        = os.path.join('input', 'cifar-10-raw', 'train')
class_dir_list  = os.listdir(data_dir)
summary_x       = []
summary_y       = []
for class_dir in class_dir_list:
    class_n     = len(os.listdir(os.path.join(data_dir, class_dir)))
    summary_x.append(class_dir)
    summary_y.append(class_n)

summary_x_np    = np.array(summary_x)
summary_y_np    = np.array(summary_y)
plt.bar(summary_x_np, summary_y_np, color='orange')
for x, y in zip(summary_x_np, summary_y_np):
    print(f'{x}: {y}')
    plt.text(x, y, y, ha='center', va='bottom')
plt.tick_params(labelrotation=-20)
plt.xlabel('class')
plt.ylabel('nums')
plt.savefig('train_data.png')
