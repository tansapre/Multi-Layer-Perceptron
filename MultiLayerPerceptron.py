import sys
import numpy as np
from random import seed
import math
from math import exp
from random import random


cmd = sys.argv
a,b,c = cmd[1],cmd[2],cmd[3]
df = np.loadtxt(a,delimiter=",", dtype=float)

#print(df)
x_trains = list(df)
x_train = []
for i in range(len(df)):
    sin_x1 = math.sin(df[i][0])
    sin_x2 = math.sin(df[i][1])
    new_df = [df[i][0],df[i][1],sin_x1,sin_x2]
    x_train.append(new_df)
#print(new_df)

y_train = np.loadtxt(b,delimiter=",", dtype=float)
x_test = np.loadtxt(c,delimiter=",", dtype=float)
x_tests =[]

for i in range(len(x_test)):
    sin_x1 = math.sin(x_test[i][0])
    sin_x2 = math.sin(x_test[i][1])
    new_df = [x_test[i][0],x_test[i][1],sin_x1,sin_x2]
    x_tests.append(new_df)

#print(df)
for i in range(len(y_train)):
    x_train[i].append(int(y_train[i]))
             
def fwd_prog(nw, ro):
	ip = ro
    #print(ip)
	for lyr in nw:
		new_ip = []
		for nurn in lyr:
			actvn = activate_nurn(nurn['wts'], ip)
			nurn['op'] = tanh_act(actvn)
			new_ip.append(nurn['op'])
		ip = new_ip
	return ip


def propogate_back(nw, expctd):
	for i in reversed(range(len(nw))):
		lyr = nw[i]
		ers = list()
		if i != len(nw)-1:
			for j in range(len(lyr)):
				a_erro = 0.0
				for nurn in nw[i + 1]:
					a_erro += (nurn['wts'][j] * nurn['dt'])
				ers.append(a_erro)
		else:
			for j in range(len(lyr)):
				nurn = lyr[j]
				ers.append(nurn['op'] - expctd[j])
		for j in range(len(lyr)):
			nurn = lyr[j]
			nurn['dt'] = tanh_act_derivative(nurn['op']) * ers[j]

def activate_nurn(wts, ip):
	actvn = wts[-1]
	for i in range(len(wts)-1):
		actvn += ip[i] * wts[i]
	return actvn


def tanh_act(actvn):
	return (exp(2*actvn) - 1.0) / (exp(2*actvn)+1.0)

def init_nw(n_ip, n_hidden, no_outp):
	nw = list()
	hidden_lyr = [{'wts':[random() for i in range(n_ip + 1)]} for i in range(n_hidden)]
	nw.append(hidden_lyr)
	op_lyr = [{'wts':[random() for i in range(n_hidden + 1)]} for i in range(no_outp)]
	nw.append(op_lyr)
	return nw

def tanh_act_derivative(op):
	return (1.0 - (op ** 2))


def update_wts(nw, ro, learning_rate):
	for i in range(len(nw)):
		ip = ro[:-1]
		if i != 0:
			ip = [nurn['op'] for nurn in nw[i - 1]]
		for nurn in nw[i]:
			for j in range(len(ip)):
				nurn['wts'][j] -= learning_rate * nurn['dt'] * ip[j]
			nurn['wts'][-1] -=  nurn['dt'] * learning_rate


def train_nw(nw, train, learning_rate, no_epoch, no_outp):
	for epoch in range(no_epoch):
		sum_err = 0
		for ro in train:
			outps = fwd_prog(nw, ro)
			expctd = [0 for i in range(no_outp)]
			expctd[ro[-1]] = 1
			sum_err += sum([(expctd[i]-outps[i])**2 for i in range(len(expctd))])
			propogate_back(nw, expctd)
			update_wts(nw, ro, learning_rate)
		

        
def predict(nw, ro):
	outps = fwd_prog(nw, ro)
	return outps.index(max(outps))


seed(1)
dset = x_train
n_ip = len(dset[0]) - 1
no_outp = len(set([ro[-1] for ro in dset]))
nw = init_nw(n_ip,6,no_outp+1)
train_nw(nw, dset, 0.02, 200, no_outp + 1)

count1 = 0
count2 = 0
preds = []
for ro in x_tests:
	pedc = predict(nw, ro)
	
	preds.append(pedc)
	if ro[-1] == pedc:
		count1 += 1
	count2 += 1
#print('accuracy',(count1/count2)*100)



with open(r'test_predictions.csv', 'w') as fp:
    for item in preds:
        fp.write("%s\n" % item)