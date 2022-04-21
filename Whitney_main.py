
from PIL import Image, ImageDraw
import numpy as np
from math import sqrt,pi,cos,sin
import datetime as dat
import os

#Approx. area of black pixels
def Area(pic):
	w,h=pic.size
	area=0.
	for i in range(w):
		for j in range(h):
			area+=255.-pic.getpixel((i,j))
	return area/(255.*w*h)

# This block is optional
fg='empty.bmp' 
IMR=os.path.join('./images/',fg)
im=Image.open(IMR)
wi,he=im.size   

img=255*np.ones((wi,he))

fx=lambda x: (x-float(wi)*0.5)/100.
fy=lambda y: (-y+float(he)*0.5)/100.

f_x=lambda x: x*100.+float(wi)*0.5
f_y=lambda y: -1.*(y*100.-float(he)*0.5)
#for i in range(wi):
#	for j in range(wi):
#		pixv=im.getpixel((i,j))
#		img[i][j]=pixv


# ######################
#
#  START AMCF
#
#
#Nvidia module
#
import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule

tt=np.linspace(0,np.pi,13)
tt=list(tt)
tt.pop()
tt=np.array(tt)

global N
N=len(tt)

RR=1.0
radpix=5
# Import all (string) kernels for AMCF 
import mod_amcf_15 as amcf
amcf.init_ker(N,wi,he)
#
#
#            Create arrays
#
#  Initial curve (lemniscate)
M=np.empty((N,2))
for i in range(N):
	de=1/(1+cos(tt[i])**2)
	M[i][0]=-RR*sin(tt[i])*de
	M[i][1]=RR*cos(tt[i])*sin(tt[i])*de
#
#
A=np.zeros((N,N))
np.fill_diagonal(A,-1)
np.fill_diagonal(A[1:N,2:N],1)
A[0][:]=np.ones(N)
A[N-1][0]=1
A_inv=np.linalg.inv(A)
#
# Pass containers to GPU
M_gpu=gpuarray.to_gpu(M.astype(np.float64))
Tan_gpu=gpuarray.to_gpu(np.zeros((N,2)).astype(np.float64))
Nor_gpu=gpuarray.to_gpu(np.zeros((N,2)).astype(np.float64))
K_gpu=gpuarray.to_gpu(np.zeros((N,2)).astype(np.float64))
A_invgpu=gpuarray.to_gpu(A_inv.astype(np.float64))
b_gpu=gpuarray.to_gpu(np.zeros(N).astype(np.float64))
a_gpu=gpuarray.to_gpu(np.zeros(N).astype(np.float64))
d_gpu=gpuarray.to_gpu(np.zeros(N).astype(np.float64))
v_gpu=gpuarray.to_gpu(np.zeros(N).astype(np.float64))

im_gpu=gpuarray.to_gpu(img.astype(np.float64))

Pix_gpu=gpuarray.to_gpu(np.zeros(N).astype(np.float64))
Sol=gpuarray.to_gpu(np.zeros((N,N)).astype(np.float64))

bu_gpu=gpuarray.to_gpu(np.zeros(2*N).astype(np.float64))
Usys=gpuarray.to_gpu(np.zeros((2*N,2*N)).astype(np.float64))
#
# Load kernel strings as GPU functions
mod = SourceModule(amcf.TanNorCurv)
TNK=mod.get_function("TNK")
DistF = SourceModule(amcf.distanceV)
Dist=DistF.get_function("Dist")
Evolution=SourceModule(amcf.Evolution)
Evolve=Evolution.get_function("Evolution")
BL=SourceModule(amcf.BLin)
BLin=BL.get_function("BLinear")
MxV=SourceModule(amcf.MMxV)
MultiMxV=MxV.get_function("MultiMxV")
Upt=SourceModule(amcf.UPT)
UpTan=Upt.get_function("Up")

BSys=SourceModule(amcf.Bu,no_extern_c=True)
Bu=BSys.get_function("Bu")
ASys=SourceModule(amcf.Au,no_extern_c=True)
Au=ASys.get_function("Au")


PixV=SourceModule(amcf.pixval)
PixVal=PixV.get_function('pixv')

#Parameters
h=np.float64(1./N)
dt=np.float64(h**2/N)
mu=np.float64(0.64)

q_gpu=np.float64(0)
qp=np.array([-0.3,0])
qpos_gpu=gpuarray.to_gpu(qp.astype(np.float64)) #charge position
Tlim=196 #Max iterations
if os.path.isdir('data')==0:
	os.mkdir('data')
route='data'
print 'start...'
Ini=dat.datetime.now()
for tk in range(Tlim):
	bu_gpu=gpuarray.to_gpu(np.zeros(2*N).astype(np.float64))
	Usys=gpuarray.to_gpu(np.zeros((2*N,2*N)).astype(np.float64))
	name='time_'+str(tk).zfill(15)+'.txt'
	#
	if (tk%5)==0:
		f=open(os.path.join(route,name),'w')
		MM=M_gpu.get()
		MM[0][0]=0
		MM[0][1]=0
		#
		x=[MM[i][0] for i in range(N)]
		y=[MM[i][1] for i in range(N)]
		for i in range(N):
			print>>f,MM[i][0],MM[i][1]
		f.close()
	Dist(M_gpu,d_gpu,block=(2,N,1))
	C_lengpu=np.float64(gpuarray.sum(d_gpu).get())
	
	BLin(d_gpu,h,C_lengpu,b_gpu,block=(N,1,1))
	
	# Solve system for a's (tangential coponents)
	MultiMxV(A_invgpu,b_gpu,a_gpu,block=(N,1,1))
	# Compute Tangent, Normal and Curvature vectors (TNK)
	TNK(M_gpu,Tan_gpu,K_gpu,Nor_gpu,mu,block=(2,N,1))
	
	PixVal(M_gpu,Pix_gpu,im_gpu,block=(N,1,1))
	# Set 2N linear system
	Bu(M_gpu,K_gpu,Nor_gpu,q_gpu,qpos_gpu,Pix_gpu,bu_gpu,block=(2*N,1,1),grid=(1,1))
	Au(M_gpu,Nor_gpu,Usys,q_gpu,qpos_gpu,Pix_gpu,block=(2*N,1,1),grid=(1,1))
	bu=bu_gpu.get()
	U=Usys.get()
	
	ss=np.linalg.solve(U,bu)
	sss=ss[:N]
	Sol_gpu=gpuarray.to_gpu(sss.astype(np.float64))
	# Compute a*T and v*N
	UpTan(Tan_gpu,a_gpu,block=(2,N,1))
	UpTan(Nor_gpu,Sol_gpu,block=(2,N,1))
	#Update cordinates
	Evolve(M_gpu,Tan_gpu,Nor_gpu,dt,block=(2,N,1))
	
	Px=Pix_gpu.get()
	# Stop if all points have reached the black area
	if (len([Px[i] for i in range(N) if Px[i]<=190])==N) :
		break
Fin=dat.datetime.now()
print 'time: ',dat.timedelta.total_seconds(Fin-Ini)

MM=M_gpu.get()
x=[MM[i][0] for i in range(N)]
y=[MM[i][1] for i in range(N)]
x.append(x[0])
y.append(y[0])
# 
name='time_'+str(tk).zfill(15)+'.txt'
f=open(os.path.join(route,name),'w')
for i in range(N):
	print>>f,MM[i][0],MM[i][1]
f.close()
# ######################
# Plot Output files in data directory
