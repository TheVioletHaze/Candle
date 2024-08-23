import numpy as np

#triangle (B,C,D from drawing)
v0_list=[1, 2, 0]
v1_list=[2, 0, 0]
v2_list=[0, 4, 1]

v0 = np.array(v0_list)
v1 = np.array(v1_list)
v2 = np.array(v2_list)

#Ray


AB = v1-v0
AC = v2-v0
print(f"AB: {str(AB)} AB: {str(AC)}")

N=np.cross(AC, AB)

print(f"Normal: {str(N)}")

D=-np.dot(N, v0)
print(f"D: {str(D)}")

