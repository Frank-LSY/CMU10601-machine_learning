import numpy as np

X = np.array([[1,1],
              [1,2],
              [1,3],
              [1,4],
              [1,5]])
y = np.array([3,8,9,12,15])

# a = X.transpose().dot(X)
# print('a: ',a)
# b = X.transpose().dot(y)
# print('b: ',b)

# theta = np.linalg.inv(X.transpose().dot(X)).dot(X.transpose()).dot(y)

# theta = np.array([0,2])
# for j in range(2):
# 	g = np.array([0.,0.])
# 	for i in range(len(X)):
# 		# print('a: ',theta.transpose().dot(X[i]))
# 		# print('b: ',theta.transpose().dot(X[i])-y[i])
# 		grad = (theta.transpose().dot(X[i])-y[i])*(X[i])
# 		g += grad
# 	print(0.4*g)
# 	theta = theta - 0.01*0.4*g
# 	print(j,': ',theta)

# a = (2.236*1-0.068-3)+(2.236*2-0.068-8)*2+(2.236*3-0.068-9)*3+(2.236*4-0.068-12)*4+(2.236*5-0.068-15)*5
# print(a)
# print(2.236-a*0.01*0.4)

a = np.array([	[0,1,2],
				[1,0,2],
				[2,1,0]]
	)
o = np.array([3,6,9])
b = a.transpose().dot(a)
c = np.linalg.inv(b)
d = c.dot(a.transpose())
e = d.dot(o)
print(e)





