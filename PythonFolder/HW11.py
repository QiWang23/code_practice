# Face recognition by using Principal Component Analysis#
import os
import numpy as np
from numpy.linalg import eig,norm
from skimage.io import imread
import matplotlib.pyplot as plt

os.chdir("C:/Users/wqemi/OneDrive - Florida State Students/Documents/FSU/STA 5635/HW")

def create_dataset(folder):
    data = []
    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        image = imread(path)
        if image is not None:
            data.append(image)
    data = np.array(data)
    shape = data.shape
    data = data.reshape((shape[0],shape[1]*shape[2]))
    return data

def distance(x,p):
    proj = np.zeros(x.shape)
    for i in range(p.shape[1]):
        proj += x.dot(p[:,i].reshape((-1,1))).dot(p[:,i].reshape(1,-1))
    return norm(x-proj,axis=1)

x = create_dataset('faces')
y = create_dataset('background')

mx = np.mean(x,axis=0)
X = x-mx
Y = y-mx
sigma = X.T.dot(X)
w, v = eig(sigma)
idx = np.argsort(w)[::-1] # decreasing order of eigenvalues
P = v[:,[0,2]] # PC obtained by the first and third principal component
projx = X.dot(P)
projy = Y.dot(P)

P10 = v[:,:10] #the 10 largest PCs
distx = distance(X,P)
disty = distance(Y,P)

# Problem a
plt.figure()
plt.title("Eigenvalues without 2 Largest Ones")
plt.scatter(idx[2:],w[idx[2:]])
plt.plot(idx[2:],w[idx[2:]])
plt.savefig('a.eps', format='eps')

# Problem b
plt.figure()
plt.title("Projection of Faces")
plt.scatter(projx[:,0],projx[:,1],s=3)
plt.savefig('b.eps', format='eps')

# Problem c
plt.figure()
plt.title("Projection")
plt.scatter(projx[:,0],projx[:,1],color='black',s=2)
plt.scatter(projy[:,0],projy[:,1],color='red',s=2)
plt.legend(["faces","background"])
plt.savefig('c.eps', format='eps')

# Problem d
plt.figure()
plt.title("Distances to the Plane")
plt.scatter(projx[:,0],distx,color='blue',s=2)
plt.scatter(projy[:,0],disty,color='red',s=2)
plt.xlabel("Projections on the the first PC")
plt.ylabel("Distances")
plt.legend(["faces","background"])
plt.savefig('d.eps', format='eps')

# Problem e
plt.figure()
plt.title("Histogram of distance")
plt.hist(distx, bins='auto', color = 'red')
plt.hist(disty, bins='auto', color = 'blue')
plt.legend(["faces","background"])
plt.savefig('e.eps', format='eps')