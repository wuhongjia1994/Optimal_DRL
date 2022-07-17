import csv
import numpy as np
import matplotlib.pyplot as plt
import os
def load_reward(file_name):
    data = []
    with open(file_name) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            a = row['average_reward']
            a = a.split(' ')
            while True:
                try:
                    a.remove('')
                except ValueError:
                    break
            if a[0] is '[': a.remove('[')

            if a[-1] is ']': a.remove(']')

            try:
                a[0] = a[0].split('[')[1]
            except:
                pass
            try:
                a[-1] = a[-1].split(']')[0]
            except:
                pass
            data.append(np.asarray(a,dtype=float))
    return np.asarray(data[1:], dtype=float)

data1 = load_reward('data/2s3z_2022_7_13_19_16_23/progress.csv')
plt.figure(figsize=(8, 6))
y1 = np.full(100, 2169.99, dtype=None)
y2= np.full(100, 2564.56, dtype=None)
y3 = np.full(100, 3035.96, dtype=None)
for i in range(3):
    plt.plot(data1[:,i],label='MC%d'%i)
plt.plot(y1, color='purple', label='SE', linestyle='--')
plt.plot(y2, color='purple', linestyle='--')
plt.plot(y3, color='purple', linestyle='--')
#my_x_ticks = np.arange(-5, 5, 0.5)
#my_y_ticks = np.arange(1200, 2400, 80)

#plt.yticks(my_y_ticks)

plt.xlabel('Episode')
plt.ylabel('The Utility of MC')
plt.legend()
plt.tight_layout()
figure_save_path = "file_fig"
if not os.path.exists(figure_save_path):
    os.makedirs(figure_save_path) # 如果不存在目录figure_save_path，则创建
plt.savefig(os.path.join(figure_save_path, 'IPPO.png'), dpi=500)#第一个是指存储路径，第二个是图片名字

plt.show()







