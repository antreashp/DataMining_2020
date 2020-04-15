import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


x = np.load('bined_x_win1.npy')

y = np.load('bined_y_win1.npy')

meh = Counter(y[:]) 
for i,val in meh.items():
    print(i,val)
exit()
print(x.shape)
plt.figure(1)
plt.ylabel('valence_counts')
plt.xlabel('value*100')
meh = Counter(x[:,0])
counts = np.zeros((100))
for i in list(meh.keys()):
    print(i)
    counts[int(float(i)*100)-1] = meh[i]
plt.bar(range(100),counts)

plt.figure(2)
plt.ylabel('valence')
plt.ylabel('records')
plt.scatter(range(x.shape[0]),x[:,0])
plt.show()

