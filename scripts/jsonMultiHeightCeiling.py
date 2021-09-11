import json
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

x_total = []
label_total = []
height = 0

with open('../data provided/json model files/recon_regen.json') as json_file:
    struct = json.load(json_file)

    for h in struct['Wall']:
        height = h['height']

    for i,ceil in enumerate(struct['MultiHeightCeiling']):
        vertice = np.vstack([[obj['x'],obj['y'], height-ceil['thickness']] for obj in ceil['BaseProfile']])
        vertice = np.concatenate((vertice,np.reshape(vertice[0],(-1,3))),axis=0)
        for vid, v in enumerate(vertice[:-1]):
            startpoint = v
            endpoint = vertice[vid + 1]
            num_points = int(np.linalg.norm(startpoint- endpoint) / 0.003)
            x = np.linspace(startpoint, endpoint, num_points)
            x_total.append(x)
            label = np.ones((len(x),1),dtype=int) * i
            label_total.append(label)
    x_total = np.vstack(x_total)
    label_total = np.vstack(label_total)



label_id = []
for i in range(len(label_total)):
    if label_total[i][0] not in label_id:
        label_id.append(label_total[i][0])

fig = plt.figure()
ax = plt.axes(projection = '3d')

# for i in label_id:
#     xp = []
#     yp = []
#     zp = []
#     xp.append(0)
#     yp.append(0)
#     zp.append(0)
#     for j in range(len(x_total)):
#         if i == label_total[j][0]:
#             xp.append(x_total[j][0])
#             yp.append(x_total[j][1])
#             zp.append(x_total[j][2])
#     # ax.plot(xp, yp, zp)
#     plt.plot(xp,yp)

# plt.show()

# print(height)