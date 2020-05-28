import matplotlib.pyplot as plt
import numpy as np


res_full = np.load('./res_full.npy', allow_pickle=True)
res_major_729 = np.load('./res_major_729.npy', allow_pickle=True)
res_major_512 = np.load('./res_major_512.npy', allow_pickle=True)

plt.title('EP accuracy')
plt.figure(figsize=(14, 8))

names = ['full, train', 'major(729), train', 'major(512), train', \
         'full, val(full)', 'major(729), val(full)', 'major(512), val(full)', \
         'full, val(minor)', 'major(729), val(minor)', 'major(512), val(minor)']

plt.plot(res_full[0], 'b-')
plt.plot(res_major_729[0], 'r-')
plt.plot(res_major_512[0], 'g-')
plt.plot(res_full[2], 'b--')
plt.plot(res_major_729[2], 'r--')
plt.plot(res_major_512[2], 'g--')
plt.plot(res_full[4], 'b:')
plt.plot(res_major_729[4], 'r:')
plt.plot(res_major_512[4], 'g:')

plt.legend(names, loc='lower right')
plt.xlabel('iterations')
length = len(res_full[0])
plt.xticks(np.arange(length), [str(1000 * (i+1)) for i in range(length)])

plt.savefig('./ep_acc.jpg')