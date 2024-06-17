
import numpy as np
import matplotlib.pyplot as plt


input = np.array([[0, 0, -1], [ 0, 1, -1], [1, 0, -1], [1, 1, -1]])
targ  = np.array([0, 1, 1, 1])

learn_rate = 0.1

weights = np.random.random(size=3)*(0.02)-0.01

######################### ##########################
#                Main training loop
######################### ##########################
for t in range(0,10):

	print("*** Iteration : ",t," ***")
	######################### ##########################
	# Testing the result of the network with a forward
	######################### ##########################

	for i in range(0,4):
		# Forward phase
		h = 0.0
		for j in range(0,3):
			h = h + weights[j]*input[i][j]
		
		if(h > 0):
			output = 1
		else:
			output = 0
	

		print("Input  : ", input[i])
		print("Target : ", targ[i])
		print("Output : ", output, "\n")
	x = np.linspace(0,1,100)
	plt.plot(x, -weights[0]/weights[1]*x + weights[2]/weights[1])
	plt.scatter([0,0,1,1],[0,1,0,1], c=targ)
	plt.xlim(-0.1,1.1)
	plt.ylim(-0.1,1.1)
	plt.show()


	######################### ##########################
	#             Training on all data once
	######################### ##########################
	for i in range(0,4):
	
		# Forward step
		h = 0.0
		for j in range(0,3):
			h = h + weights[j]*input[i][j]
		
		if(h > 0):
			output = 1
		else:
			output = 0
		
		#Back-propagation phase
		for j in range(0,3):
			weights[j] = weights[j] - learn_rate*(output-targ[i])*input[i][j]
			



			









