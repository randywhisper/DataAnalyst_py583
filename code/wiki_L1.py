#------->Project-CS 583<----------
#------Author: Randy Wang---------
#---Mail:randywhisper@gmail.com---

#execute the structed python file and get the dataList
execfile("structed_wiki.py")
import copy

#exectue 5-fold cross-validation
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(penalty = 'L1')
clf_countUnd = LogisticRegression(penalty = 'L1')
clf_countDir = LogisticRegression(penalty = 'L1')
clf_proUnd = LogisticRegression(penalty = 'L1')
clf_proDir = LogisticRegression(penalty = 'L1')
clf_existUnd = LogisticRegression(penalty = 'L1')
clf_existDir = LogisticRegression(penalty = 'L1')
clf_modeUnd = LogisticRegression(penalty = 'L1')
clf_modeDir = LogisticRegression(penalty = 'L1')

k_fold = cross_validation.KFold(len(dataList),n_folds=5,indices=True)


import numpy as np
def struct():
	scoreContent = 0
	scoreCountUnd = scoreCountDir = 0
	scoreProUnd = scoreProDir = 0
	scoreExistUnd = scoreExistDir = 0
	scoreModeUnd = scoreModeDir = 0
	for train_indices,test_indices in k_fold:
#define the the class_label and the word_attribute of the train and test data
		trainAtt = []
		trainLab = []
		testAtt = []
		testLab = []
		countAttUnd = []
		countTestAttUnd = []
#construct the list of the train list
		for i in range(len(train_indices)):
			trainAtt.append(dataList[train_indices[i]][2])
			trainLab.append(dataList[train_indices[i]][1])
#construct the list of the test list
		for i in range(len(test_indices)):
			testAtt.append(dataList[test_indices[i]][2])
			testLab.append(dataList[test_indices[i]][1])
#content-only classification
		X = np.array(trainAtt,dtype='f')
		y = np.array(trainLab,dtype='f')
		m = np.array(testAtt,dtype ='f')
		n_test = np.array(testLab,dtype ='f')
		clf.fit(X,y)
		testPre = clf.predict(m)		
		scoreContent = scoreContent + clf.score(m,n_test)
		print(scoreContent)

#creating relational features of the training instances	
		countAttUnd = copy.deepcopy(trainAtt)
		countAttDir = copy.deepcopy(trainAtt)
		proporAttUnd = copy.deepcopy(trainAtt)
		proporAttDir = copy.deepcopy(trainAtt)
		existAttUnd = copy.deepcopy(trainAtt)
		existAttDir = copy.deepcopy(trainAtt)
		modeAttUnd = copy.deepcopy(trainAtt)
		modeAttDir = copy.deepcopy(trainAtt)
   		countTestAttUnd = copy.deepcopy(testAtt)
		countTestAttDir = copy.deepcopy(testAtt)
		proporTestAttUnd = copy.deepcopy(testAtt)
		proporTestAttDir = copy.deepcopy(testAtt)
		existTestAttUnd = copy.deepcopy(testAtt)
		existTestAttDir = copy.deepcopy(testAtt)
		modeTestAttUnd = copy.deepcopy(testAtt)
		modeTestAttDir = copy.deepcopy(testAtt)
		

		for i in range(len(train_indices)):
			citeTrain = []
			citingTrain = []
			for m in range(19):
				citeTrain.append(0)
				citingTrain.append(0)
			total = totalCited = totalCiting = 0.0
			maxTotal = maxCited = maxCiting = 0
			index = indexCited = indexCiting = 0
			for m in range(len(dataList[train_indices[i]][3][0])):
				for n in range(len(train_indices)):
					if(dataList[train_indices[i]][3][0][m] == dataList[train_indices[n]][0]):
						if(dataList[train_indices[n]][1] == 1):
							citingTrain[0] = citingTrain[0] + 1
						elif(dataList[train_indices[n]][1] == 2):
							citingTrain[1] = citingTrain[1] + 1
						elif(dataList[train_indices[n]][1] == 3):
							citingTrain[2] = citingTrain[2] + 1
						elif(dataList[train_indices[n]][1] == 4):
							citingTrain[3] = citingTrain[3] + 1
						elif(dataList[train_indices[n]][1] == 5):
							citingTrain[4] = citingTrain[4] + 1
						elif(dataList[train_indices[n]][1] == 6):
							citingTrain[5] = citingTrain[5] + 1
						elif(dataList[train_indices[n]][1] == 7):
							citingTrain[6] = citingTrain[6] + 1
						elif(dataList[train_indices[n]][1] == 8):
							citingTrain[7] = citingTrain[7] + 1
						elif(dataList[train_indices[n]][1] == 9):
							citingTrain[8] = citingTrain[8] + 1
						elif(dataList[train_indices[n]][1] == 10):
							citingTrain[9] = citingTrain[9] + 1
						elif(dataList[train_indices[n]][1] == 11):
							citingTrain[10] = citingTrain[10] + 1
						elif(dataList[train_indices[n]][1] == 12):
							citingTrain[11] = citingTrain[11] + 1
						elif(dataList[train_indices[n]][1] == 13):
							citingTrain[12] = citingTrain[12] + 1
						elif(dataList[train_indices[n]][1] == 14):
							citingTrain[13] = citingTrain[13] + 1
						elif(dataList[train_indices[n]][1] == 15):
							citingTrain[14] = citingTrain[14] + 1
						elif(dataList[train_indices[n]][1] == 16):
							citingTrain[15] = citingTrain[15] + 1
						elif(dataList[train_indices[n]][1] == 17):
							citingTrain[16] = citingTrain[16] + 1
						elif(dataList[train_indices[n]][1] == 18):
							citingTrain[17] = citingTrain[17] + 1
						elif(dataList[train_indices[n]][1] == 19):
							citingTrain[18] = citingTrain[18] + 1
						break
			for p in range(len(dataList[train_indices[i]][3][1])):
				for q in range(len(train_indices)):
					if(dataList[train_indices[i]][3][1][p] == dataList[train_indices[q]][0]):
						if(dataList[train_indices[q]][1] == 1):
							citeTrain[0] = citeTrain[0] +1
						elif(dataList[train_indices[q]][1] == 2):
							citeTrain[1] = citeTrain[1] +1
						elif(dataList[train_indices[q]][1] == 3):
							citeTrain[2] = citeTrain[2] +1
						elif(dataList[train_indices[q]][1] == 4):
							citeTrain[3] = citeTrain[3] +1
						elif(dataList[train_indices[q]][1] == 5):
							citeTrain[4] = citeTrain[4] +1
						elif(dataList[train_indices[q]][1] == 6):
							citeTrain[5] = citeTrain[5] +1
						elif(dataList[train_indices[q]][1] == 7):
							citeTrain[6] = citeTrain[6] +1
						elif(dataList[train_indices[q]][1] == 8):
							citeTrain[7] = citeTrain[7] +1
						elif(dataList[train_indices[q]][1] == 9):
							citeTrain[8] = citeTrain[8] +1
						elif(dataList[train_indices[q]][1] == 10):
							citeTrain[9] = citeTrain[9] +1
						elif(dataList[train_indices[q]][1] == 11):
							citeTrain[10] = citeTrain[10] +1
						elif(dataList[train_indices[q]][1] == 12):
							citeTrain[11] = citeTrain[11] +1
						elif(dataList[train_indices[q]][1] == 13):
							citeTrain[12] = citeTrain[12] +1
						elif(dataList[train_indices[q]][1] == 14):
							citeTrain[13] = citeTrain[13] +1
						elif(dataList[train_indices[q]][1] == 15):
							citeTrain[14] = citeTrain[14] +1
						elif(dataList[train_indices[q]][1] == 16):
							citeTrain[15] = citeTrain[15] +1
						elif(dataList[train_indices[q]][1] == 17):
							citeTrain[16] = citeTrain[16] +1
						elif(dataList[train_indices[q]][1] == 18):
							citeTrain[17] = citeTrain[17] +1
						elif(dataList[train_indices[q]][1] == 19):
							citeTrain[18] = citeTrain[18] +1
						break
			for u in range(19):
				total = total + (citeTrain[u]+citingTrain[u])
				totalCited = totalCited + citeTrain[u]
				totalCiting = totalCiting + citingTrain[u]
			for u in range(19):
				countAttUnd[i].append(citeTrain[u]+citingTrain[u])	
				countAttDir[i].append(citeTrain[u])
				if(total == 0):
					proporAttUnd[i].append(0)
				else:
					proporAttUnd[i].append(round((citeTrain[u]+citingTrain[u])/total,3))
				if(totalCited == 0):
					proporAttDir[i].append(0)
				else:
					proporAttDir[i].append(round(citeTrain[u]/totalCited,3))
				if(citeTrain[u]+citingTrain[u] == 0):
					existAttUnd[i].append(0)
				else:
					existAttUnd[i].append(1)
				if(citeTrain[u] == 0):
					existAttDir[i].append(0)
				else:
					existAttDir[i].append(1)
				if((citeTrain[u]+citingTrain[u]) > maxTotal):
					index = u	
					maxTotal = citeTrain[u]+citingTrain[u]
				if(citeTrain[u] > maxCited):
					indexCited = u
					maxCited = citeTrain[u]
			modeAttUnd[i].append(index)
			modeAttDir[i].append(indexCited)
			for u in range(19):
				countAttDir[i].append(citingTrain[u])
				if(totalCiting == 0):
					proporAttDir[i].append(0)
				else:
					proporAttDir[i].append(round(citingTrain[u]/totalCiting,3))
				if(citingTrain[u] == 0):
						existAttDir[i].append(0)
				else:
						existAttDir[i].append(1)
				if(citingTrain[u] > maxCiting):
					indexCiting = u
					maxCiting = citingTrain[u]
			modeAttDir[i].append(indexCiting)
		X_countUnd = np.array(countAttUnd,dtype='f')
		X_countDir = np.array(countAttDir,dtype='f')
		X_proUnd = np.array(proporAttUnd,dtype='f')
		X_proDir = np.array(proporAttDir,dtype='f')
		X_existUnd = np.array(existAttUnd,dtype='f')
		X_existDir = np.array(existAttDir,dtype='f')
		X_modeUnd = np.array(modeAttUnd,dtype='f')
		X_modeDir = np.array(modeAttDir,dtype='f')
		clf_countUnd.fit(X_countUnd,y)
		clf_countDir.fit(X_countDir,y)
		clf_proUnd.fit(X_proUnd,y)
		clf_proDir.fit(X_proDir,y)
		clf_existUnd.fit(X_existUnd,y)
		clf_existDir.fit(X_existDir,y)
		clf_modeUnd.fit(X_modeUnd,y)
		clf_modeDir.fit(X_modeDir,y)

#creating the relational features of testing instances
		index_test = 0
		testPreStart = testPre
		testPreNew =  testPre
		while( 0 <= index_test <10):
			for i in range(len(test_indices)):
					citeTest = []
					citingTest = []
					for m in range(19):
						citeTest.append(0)
						citingTest.append(0)
#					totalTest = totalCitedTest = totalCitingTest = 0.0
#					maxTotalTest = maxCitedTest = maxCitingTest = 0
#					indexTest = indexCitedTest = indexCitingTest = 0
					for m in range(len(dataList[test_indices[i]][3][0])):
						for n in range(len(train_indices)):
							if(dataList[test_indices[i]][3][0][m] == dataList[train_indices[n]][0]):
								if(dataList[train_indices[n]][1] == 1):
									citingTest[0] = citingTest[0] + 1
								elif(dataList[train_indices[n]][1] == 2):
									citingTest[1] = citingTest[1] + 1
								elif(dataList[train_indices[n]][1] == 3):
									citingTest[2] = citingTest[2] + 1
								elif(dataList[train_indices[n]][1] == 4):
									citingTest[3] = citingTest[3] + 1
								elif(dataList[train_indices[n]][1] == 5):
									citingTest[4] = citingTest[4] + 1
								elif(dataList[train_indices[n]][1] == 6):
									citingTest[5] = citingTest[5] + 1
								elif(dataList[train_indices[n]][1] == 7):
									citingTest[6] = citingTest[6] + 1
								elif(dataList[train_indices[n]][1] == 8):
									citingTest[7] = citingTest[7] + 1
								elif(dataList[train_indices[n]][1] == 9):
									citingTest[8] = citingTest[8] + 1
								elif(dataList[train_indices[n]][1] == 10):
									citingTest[9] = citingTest[9] + 1
								elif(dataList[train_indices[n]][1] == 11):
									citingTest[10] = citingTest[10] + 1
								elif(dataList[train_indices[n]][1] == 12):
									citingTest[11] = citingTest[11] + 1
								elif(dataList[train_indices[n]][1] == 13):
									citingTest[12] = citingTest[12] + 1
								elif(dataList[train_indices[n]][1] == 14):
									citingTest[13] = citingTest[13] + 1
								elif(dataList[train_indices[n]][1] == 15):
									citingTest[14] = citingTest[14] + 1
								elif(dataList[train_indices[n]][1] == 16):
									citingTest[15] = citingTest[15] + 1
								elif(dataList[train_indices[n]][1] == 17):
									citingTest[16] = citingTest[16] + 1
								elif(dataList[train_indices[n]][1] == 18):
									citingTest[17] = citingTest[17] + 1
								elif(dataList[train_indices[n]][1] == 19):
									citingTest[18] = citingTest[18] + 1
								break
						for n in range(len(test_indices)):
							if(dataList[test_indices[i]][3][0][m] == dataList[test_indices[n]][0]):
								if(testPreStart[n] == 1):
									citingTest[0] = citingTest[0] + 1
								elif(testPreStart[n] == 2):
									citingTest[1] = citingTest[1] + 1
								elif(testPreStart[n] == 3):
									citingTest[2] = citingTest[2] + 1
								elif(testPreStart[n] == 4):
									citingTest[3] = citingTest[3] + 1
								elif(testPreStart[n] == 5):
									citingTest[4] = citingTest[4] + 1
								elif(testPreStart[n] == 6):
									citingTest[5] = citingTest[5] + 1
								elif(testPreStart[n] == 7):
									citingTest[6] = citingTest[6] + 1
								elif(testPreStart[n] == 8):
									citingTest[7] = citingTest[7] + 1
								elif(testPreStart[n] == 9):
									citingTest[8] = citingTest[8] + 1
								elif(testPreStart[n] == 10):
									citingTest[9] = citingTest[9] + 1
								elif(testPreStart[n] == 11):
									citingTest[10] = citingTest[10] + 1
								elif(testPreStart[n] == 12):
									citingTest[11] = citingTest[11] + 1
								elif(testPreStart[n] == 13):
									citingTest[12] = citingTest[12] + 1
								elif(testPreStart[n] == 14):
									citingTest[13] = citingTest[13] + 1
								elif(testPreStart[n] == 15):
									citingTest[14] = citingTest[14] + 1
								elif(testPreStart[n] == 16):
									citingTest[15] = citingTest[15] + 1
								elif(testPreStart[n] == 17):
									citingTest[16] = citingTest[16] + 1
								elif(testPreStart[n] == 18):
									citingTest[17] = citingTest[17] + 1
								elif(testPreStart[n] == 19):
									citingTest[18] = citingTest[18] + 1
								break
					for p in range(len(dataList[test_indices[i]][3][1])):
						for q in range(len(train_indices)):
							if(dataList[test_indices[i]][3][1][p] == dataList[train_indices[q]][0]):
								if(dataList[train_indices[q]][1] == 1):
									citeTest[0] = citeTest[0] +1
								elif(dataList[train_indices[q]][1] == 2):
									citeTest[1] = citeTest[1] +1
								elif(dataList[train_indices[q]][1] == 3):
									citeTest[2] = citeTest[2] +1
								elif(dataList[train_indices[q]][1] == 4):
									citeTest[3] = citeTest[3] +1
								elif(dataList[train_indices[q]][1] == 5):
									citeTest[4] = citeTest[4] +1
								elif(dataList[train_indices[q]][1] == 6):
									citeTest[5] = citeTest[5] +1
								elif(dataList[train_indices[q]][1] == 7):
									citeTest[6] = citeTest[6] +1
								elif(dataList[train_indices[q]][1] == 8):
									citeTest[7] = citeTest[7] +1
								elif(dataList[train_indices[q]][1] == 9):
									citeTest[8] = citeTest[8] +1
								elif(dataList[train_indices[q]][1] == 10):
									citeTest[9] = citeTest[9] +1
								elif(dataList[train_indices[q]][1] == 11):
									citeTest[10] = citeTest[10] +1
								elif(dataList[train_indices[q]][1] == 12):
									citeTest[11] = citeTest[11] +1
								elif(dataList[train_indices[q]][1] == 13):
									citeTest[12] = citeTest[12] +1
								elif(dataList[train_indices[q]][1] == 14):
									citeTest[13] = citeTest[13] +1
								elif(dataList[train_indices[q]][1] == 15):
									citeTest[14] = citeTest[14] +1
								elif(dataList[train_indices[q]][1] == 16):
									citeTest[15] = citeTest[15] +1
								elif(dataList[train_indices[q]][1] == 17):
									citeTest[16] = citeTest[16] +1
								elif(dataList[train_indices[q]][1] == 18):
									citeTest[17] = citeTest[17] +1
								elif(dataList[train_indices[q]][1] == 19):
									citeTest[18] = citeTest[18] +1
								break
						for q in range(len(test_indices)):
							if(dataList[test_indices[i]][3][1][p] == dataList[test_indices[q]][0]):
								if(testPreStart[q] == 1):
									citeTest[0] = citeTest[0] + 1
								elif(testPreStart[q] == 2):
									citeTest[1] = citeTest[1] + 1
								elif(testPreStart[q] == 3):
									citeTest[2] = citeTest[2] + 1
								elif(testPreStart[q] == 4):
									citeTest[3] = citeTest[3] + 1
								elif(testPreStart[q] == 5):
									citeTest[4] = citeTest[4] + 1
								elif(testPreStart[q] == 6):
									citeTest[5] = citeTest[5] + 1
								elif(testPreStart[q] == 7):
									citeTest[6] = citeTest[6] + 1
								elif(testPreStart[q] == 8):
									citeTest[7] = citeTest[7] + 1
								elif(testPreStart[q] == 9):
									citeTest[8] = citeTest[8] + 1
								elif(testPreStart[q] == 10):
									citeTest[9] = citeTest[9] + 1
								elif(testPreStart[q] == 11):
									citeTest[10] = citeTest[10] + 1
								elif(testPreStart[q] == 12):
									citeTest[11] = citeTest[11] + 1
								elif(testPreStart[q] == 13):
									citeTest[12] = citeTest[12] + 1
								elif(testPreStart[q] == 14):
									citeTest[13] = citeTest[13] + 1
								elif(testPreStart[q] == 15):
									citeTest[14] = citeTest[14] + 1
								elif(testPreStart[q] == 16):
									citeTest[15] = citeTest[15] + 1
								elif(testPreStart[q] == 17):
									citeTest[16] = citeTest[16] + 1
								elif(testPreStart[q] == 18):
									citeTest[17] = citeTest[17] + 1
								elif(testPreStart[q] == 19):
									citeTest[18] = citeTest[18] + 1
								break
#					for u in range(6):
#						totalTest = totalTest + (citeTest[u]+citingTest[u])
#						totalCitedTest = totalCitedTest + citeTest[u]
#						totalCitingTest = totalCitingTest + citingTest[u]]
					for u in range(19):
						if(index_test == 0):
							countTestAttUnd[i].append(float(citeTest[u]+citingTest[u]))	
						else:
							countTestAttUnd[i][4973+u] = float(citeTest[u]+citingTest[u])
#						countTestAttDir[i].append(citeTest[u])
#						if(total == 0):
#							proporTestAttUnd[i].append(0)
#						else:
#							proporTestAttUnd[i].append(round((citeTest[u]+citingTest[u])/totalTest,3))
#						if(totalCited == 0):
#							proporTestAttDir[i].append(0)
#						else:
#							proporTestAttDir[i].append(round(citeTest[u]/totalCitedTest,3))
#						if(citeTest[u]+citingTest[u] == 0):
#							existTestAttUnd[i].append(0)
#						else:
#							existTestAttUnd[i].append(1)
#						if(citeTest[u] == 0):
#							existTestAttDir[i].append(0)
#						else:
#							existTestAttDir[i].append(1)
#						if((citeTest[u]+citingTest[u]) > maxTotalTest):
#							indexTest = u	
#							maxTotalTest = citeTest[u]+citingTest[u]
#						if(citeTest[u] > maxCitedTest):
#							indexCitedTest = u
#							maxCitedTest = citeTest[u]
#					modeAttUnd[i].append(indexTest)
#					modeAttDir[i].append(indexCitedTest)
#					for u in range(6):
#						countTestAttDir[i].append(citingTest[u])
#						if(totalCitingTest == 0):
#							proporTestAttDir[i].append(0)
#						else:
#							proporTestAttDir[i].append(round(citingTest[u]/totalCitingTest,3))
#						if(citingTest[u] == 0):
#								existTestAttDir[i].append(0)
#						else:
#								existTestAttDir[i].append(1)
#						if(citingTest[u] > maxCitingTest):
#							indexCitingTest = u
#							maxCitingTest = citingTest[u]
#					modeTestAttDir[i].append(indexCitingTest)
			index_test = index_test + 1
			m_test = np.array(countTestAttUnd,dtype='f')	
			testPreNew = clf_countUnd.predict(m_test)		
			if(np.allclose(testPreNew,testPreStart)):
				break
			else:
				testPreStart = testPreNew
		scoreCountUnd = scoreCountUnd + clf_countUnd.score(m_test,n_test)
		print(scoreCountUnd)


		index_test = 0
		testPreStart = testPre
		testPreNew =  testPre
		while( 0 <= index_test <10):
			for i in range(len(test_indices)):
					citeTest = []
					citingTest = []
					for m in range(19):
						citeTest.append(0)
						citingTest.append(0)
#					totalTest = totalCitedTest = totalCitingTest = 0.0
#					maxTotalTest = maxCitedTest = maxCitingTest = 0
#					indexTest = indexCitedTest = indexCitingTest = 0
					for m in range(len(dataList[test_indices[i]][3][0])):
						for n in range(len(train_indices)):
							if(dataList[test_indices[i]][3][0][m] == dataList[train_indices[n]][0]):
								if(dataList[train_indices[n]][1] == 1):
									citingTest[0] = citingTest[0] + 1
								elif(dataList[train_indices[n]][1] == 2):
									citingTest[1] = citingTest[1] + 1
								elif(dataList[train_indices[n]][1] == 3):
									citingTest[2] = citingTest[2] + 1
								elif(dataList[train_indices[n]][1] == 4):
									citingTest[3] = citingTest[3] + 1
								elif(dataList[train_indices[n]][1] == 5):
									citingTest[4] = citingTest[4] + 1
								elif(dataList[train_indices[n]][1] == 6):
									citingTest[5] = citingTest[5] + 1
								elif(dataList[train_indices[n]][1] == 7):
									citingTest[6] = citingTest[6] + 1
								elif(dataList[train_indices[n]][1] == 8):
									citingTest[7] = citingTest[7] + 1
								elif(dataList[train_indices[n]][1] == 9):
									citingTest[8] = citingTest[8] + 1
								elif(dataList[train_indices[n]][1] == 10):
									citingTest[9] = citingTest[9] + 1
								elif(dataList[train_indices[n]][1] == 11):
									citingTest[10] = citingTest[10] + 1
								elif(dataList[train_indices[n]][1] == 12):
									citingTest[11] = citingTest[11] + 1
								elif(dataList[train_indices[n]][1] == 13):
									citingTest[12] = citingTest[12] + 1
								elif(dataList[train_indices[n]][1] == 14):
									citingTest[13] = citingTest[13] + 1
								elif(dataList[train_indices[n]][1] == 15):
									citingTest[14] = citingTest[14] + 1
								elif(dataList[train_indices[n]][1] == 16):
									citingTest[15] = citingTest[15] + 1
								elif(dataList[train_indices[n]][1] == 17):
									citingTest[16] = citingTest[16] + 1
								elif(dataList[train_indices[n]][1] == 18):
									citingTest[17] = citingTest[17] + 1
								elif(dataList[train_indices[n]][1] == 19):
									citingTest[18] = citingTest[18] + 1
								break
						for n in range(len(test_indices)):
							if(dataList[test_indices[i]][3][0][m] == dataList[test_indices[n]][0]):
								if(testPreStart[n] == 1):
									citingTest[0] = citingTest[0] + 1
								elif(testPreStart[n] == 2):
									citingTest[1] = citingTest[1] + 1
								elif(testPreStart[n] == 3):
									citingTest[2] = citingTest[2] + 1
								elif(testPreStart[n] == 4):
									citingTest[3] = citingTest[3] + 1
								elif(testPreStart[n] == 5):
									citingTest[4] = citingTest[4] + 1
								elif(testPreStart[n] == 6):
									citingTest[5] = citingTest[5] + 1
								elif(testPreStart[n] == 7):
									citingTest[6] = citingTest[6] + 1
								elif(testPreStart[n] == 8):
									citingTest[7] = citingTest[7] + 1
								elif(testPreStart[n] == 9):
									citingTest[8] = citingTest[8] + 1
								elif(testPreStart[n] == 10):
									citingTest[9] = citingTest[9] + 1
								elif(testPreStart[n] == 11):
									citingTest[10] = citingTest[10] + 1
								elif(testPreStart[n] == 12):
									citingTest[11] = citingTest[11] + 1
								elif(testPreStart[n] == 13):
									citingTest[12] = citingTest[12] + 1
								elif(testPreStart[n] == 14):
									citingTest[13] = citingTest[13] + 1
								elif(testPreStart[n] == 15):
									citingTest[14] = citingTest[14] + 1
								elif(testPreStart[n] == 16):
									citingTest[15] = citingTest[15] + 1
								elif(testPreStart[n] == 17):
									citingTest[16] = citingTest[16] + 1
								elif(testPreStart[n] == 18):
									citingTest[17] = citingTest[17] + 1
								elif(testPreStart[n] == 19):
									citingTest[18] = citingTest[18] + 1
								break
					for p in range(len(dataList[test_indices[i]][3][1])):
						for q in range(len(train_indices)):
							if(dataList[test_indices[i]][3][1][p] == dataList[train_indices[q]][0]):
								if(dataList[train_indices[q]][1] == 1):
									citeTest[0] = citeTest[0] +1
								elif(dataList[train_indices[q]][1] == 2):
									citeTest[1] = citeTest[1] +1
								elif(dataList[train_indices[q]][1] == 3):
									citeTest[2] = citeTest[2] +1
								elif(dataList[train_indices[q]][1] == 4):
									citeTest[3] = citeTest[3] +1
								elif(dataList[train_indices[q]][1] == 5):
									citeTest[4] = citeTest[4] +1
								elif(dataList[train_indices[q]][1] == 6):
									citeTest[5] = citeTest[5] +1
								elif(dataList[train_indices[q]][1] == 7):
									citeTest[6] = citeTest[6] +1
								elif(dataList[train_indices[q]][1] == 8):
									citeTest[7] = citeTest[7] +1
								elif(dataList[train_indices[q]][1] == 9):
									citeTest[8] = citeTest[8] +1
								elif(dataList[train_indices[q]][1] == 10):
									citeTest[9] = citeTest[9] +1
								elif(dataList[train_indices[q]][1] == 11):
									citeTest[10] = citeTest[10] +1
								elif(dataList[train_indices[q]][1] == 12):
									citeTest[11] = citeTest[11] +1
								elif(dataList[train_indices[q]][1] == 13):
									citeTest[12] = citeTest[12] +1
								elif(dataList[train_indices[q]][1] == 14):
									citeTest[13] = citeTest[13] +1
								elif(dataList[train_indices[q]][1] == 15):
									citeTest[14] = citeTest[14] +1
								elif(dataList[train_indices[q]][1] == 16):
									citeTest[15] = citeTest[15] +1
								elif(dataList[train_indices[q]][1] == 17):
									citeTest[16] = citeTest[16] +1
								elif(dataList[train_indices[q]][1] == 18):
									citeTest[17] = citeTest[17] +1
								elif(dataList[train_indices[q]][1] == 19):
									citeTest[18] = citeTest[18] +1
								break
						for q in range(len(test_indices)):
							if(dataList[test_indices[i]][3][1][p] == dataList[test_indices[q]][0]):
								if(testPreStart[q] == 1):
									citeTest[0] = citeTest[0] + 1
								elif(testPreStart[q] == 2):
									citeTest[1] = citeTest[1] + 1
								elif(testPreStart[q] == 3):
									citeTest[2] = citeTest[2] + 1
								elif(testPreStart[q] == 4):
									citeTest[3] = citeTest[3] + 1
								elif(testPreStart[q] == 5):
									citeTest[4] = citeTest[4] + 1
								elif(testPreStart[q] == 6):
									citeTest[5] = citeTest[5] + 1
								elif(testPreStart[q] == 7):
									citeTest[6] = citeTest[6] + 1
								elif(testPreStart[q] == 8):
									citeTest[7] = citeTest[7] + 1
								elif(testPreStart[q] == 9):
									citeTest[8] = citeTest[8] + 1
								elif(testPreStart[q] == 10):
									citeTest[9] = citeTest[9] + 1
								elif(testPreStart[q] == 11):
									citeTest[10] = citeTest[10] + 1
								elif(testPreStart[q] == 12):
									citeTest[11] = citeTest[11] + 1
								elif(testPreStart[q] == 13):
									citeTest[12] = citeTest[12] + 1
								elif(testPreStart[q] == 14):
									citeTest[13] = citeTest[13] + 1
								elif(testPreStart[q] == 15):
									citeTest[14] = citeTest[14] + 1
								elif(testPreStart[q] == 16):
									citeTest[15] = citeTest[15] + 1
								elif(testPreStart[q] == 17):
									citeTest[16] = citeTest[16] + 1
								elif(testPreStart[q] == 18):
									citeTest[17] = citeTest[17] + 1
								elif(testPreStart[q] == 19):
									citeTest[18] = citeTest[18] + 1
								break
#					for u in range(6):
#						totalTest = totalTest + (citeTest[u]+citingTest[u])
#						totalCitedTest = totalCitedTest + citeTest[u]
#						totalCitingTest = totalCitingTest + citingTest[u]]
					for u in range(19):
						if(index_test == 0):
							countTestAttDir[i].append(float(citeTest[u]))	
						else:
							countTestAttDir[i][4973+u] = float(citeTest[u])
#						countTestAttDir[i].append(citeTest[u])
#						if(total == 0):
#							proporTestAttUnd[i].append(0)
#						else:
#							proporTestAttUnd[i].append(round((citeTest[u]+citingTest[u])/totalTest,3))
#						if(totalCited == 0):
#							proporTestAttDir[i].append(0)
#						else:
#							proporTestAttDir[i].append(round(citeTest[u]/totalCitedTest,3))
#						if(citeTest[u]+citingTest[u] == 0):
#							existTestAttUnd[i].append(0)
#						else:
#							existTestAttUnd[i].append(1)
#						if(citeTest[u] == 0):
#							existTestAttDir[i].append(0)
#						else:
#							existTestAttDir[i].append(1)
#						if((citeTest[u]+citingTest[u]) > maxTotalTest):
#							indexTest = u	
#							maxTotalTest = citeTest[u]+citingTest[u]
#						if(citeTest[u] > maxCitedTest):
#							indexCitedTest = u
#							maxCitedTest = citeTest[u]
#					modeAttUnd[i].append(indexTest)
#					modeAttDir[i].append(indexCitedTest)
					for u in range(19):
						if(index_test == 0):
							countTestAttDir[i].append(citingTest[u])
						else:
							countTestAttDir[i][4992+u] = float(citingTest[u])
#						if(totalCitingTest == 0):
#							proporTestAttDir[i].append(0)
#						else:
#							proporTestAttDir[i].append(round(citingTest[u]/totalCitingTest,3))
#						if(citingTest[u] == 0):
#								existTestAttDir[i].append(0)
#						else:
#								existTestAttDir[i].append(1)
#						if(citingTest[u] > maxCitingTest):
#							indexCitingTest = u
#							maxCitingTest = citingTest[u]
#					modeTestAttDir[i].append(indexCitingTest)
			index_test = index_test + 1
			m_test = np.array(countTestAttDir,dtype='f')	
			testPreNew = clf_countDir.predict(m_test)		
			if(np.allclose(testPreNew,testPreStart)):
				break
			else:
				testPreStart = testPreNew
		scoreCountDir = scoreCountDir + clf_countDir.score(m_test,n_test)
		print(scoreCountDir)


		index_test = 0
		testPreStart = testPre
		testPreNew =  testPre
		while( 0 <= index_test <10):
			for i in range(len(test_indices)):
					citeTest = []
					citingTest = []
					for m in range(19):
						citeTest.append(0)
						citingTest.append(0)
					totalTest = totalCitedTest = totalCitingTest = 0.0
					maxTotalTest = maxCitedTest = maxCitingTest = 0
					indexTest = indexCitedTest = indexCitingTest = 0
					for m in range(len(dataList[test_indices[i]][3][0])):
						for n in range(len(train_indices)):
							if(dataList[test_indices[i]][3][0][m] == dataList[train_indices[n]][0]):
								if(dataList[train_indices[n]][1] == 1):
									citingTest[0] = citingTest[0] + 1
								elif(dataList[train_indices[n]][1] == 2):
									citingTest[1] = citingTest[1] + 1
								elif(dataList[train_indices[n]][1] == 3):
									citingTest[2] = citingTest[2] + 1
								elif(dataList[train_indices[n]][1] == 4):
									citingTest[3] = citingTest[3] + 1
								elif(dataList[train_indices[n]][1] == 5):
									citingTest[4] = citingTest[4] + 1
								elif(dataList[train_indices[n]][1] == 6):
									citingTest[5] = citingTest[5] + 1
								elif(dataList[train_indices[n]][1] == 7):
									citingTest[6] = citingTest[6] + 1
								elif(dataList[train_indices[n]][1] == 8):
									citingTest[7] = citingTest[7] + 1
								elif(dataList[train_indices[n]][1] == 9):
									citingTest[8] = citingTest[8] + 1
								elif(dataList[train_indices[n]][1] == 10):
									citingTest[9] = citingTest[9] + 1
								elif(dataList[train_indices[n]][1] == 11):
									citingTest[10] = citingTest[10] + 1
								elif(dataList[train_indices[n]][1] == 12):
									citingTest[11] = citingTest[11] + 1
								elif(dataList[train_indices[n]][1] == 13):
									citingTest[12] = citingTest[12] + 1
								elif(dataList[train_indices[n]][1] == 14):
									citingTest[13] = citingTest[13] + 1
								elif(dataList[train_indices[n]][1] == 15):
									citingTest[14] = citingTest[14] + 1
								elif(dataList[train_indices[n]][1] == 16):
									citingTest[15] = citingTest[15] + 1
								elif(dataList[train_indices[n]][1] == 17):
									citingTest[16] = citingTest[16] + 1
								elif(dataList[train_indices[n]][1] == 18):
									citingTest[17] = citingTest[17] + 1
								elif(dataList[train_indices[n]][1] == 19):
									citingTest[18] = citingTest[18] + 1
								break
						for n in range(len(test_indices)):
							if(dataList[test_indices[i]][3][0][m] == dataList[test_indices[n]][0]):
								if(testPreStart[n] == 1):
									citingTest[0] = citingTest[0] + 1
								elif(testPreStart[n] == 2):
									citingTest[1] = citingTest[1] + 1
								elif(testPreStart[n] == 3):
									citingTest[2] = citingTest[2] + 1
								elif(testPreStart[n] == 4):
									citingTest[3] = citingTest[3] + 1
								elif(testPreStart[n] == 5):
									citingTest[4] = citingTest[4] + 1
								elif(testPreStart[n] == 6):
									citingTest[5] = citingTest[5] + 1
								elif(testPreStart[n] == 7):
									citingTest[6] = citingTest[6] + 1
								elif(testPreStart[n] == 8):
									citingTest[7] = citingTest[7] + 1
								elif(testPreStart[n] == 9):
									citingTest[8] = citingTest[8] + 1
								elif(testPreStart[n] == 10):
									citingTest[9] = citingTest[9] + 1
								elif(testPreStart[n] == 11):
									citingTest[10] = citingTest[10] + 1
								elif(testPreStart[n] == 12):
									citingTest[11] = citingTest[11] + 1
								elif(testPreStart[n] == 13):
									citingTest[12] = citingTest[12] + 1
								elif(testPreStart[n] == 14):
									citingTest[13] = citingTest[13] + 1
								elif(testPreStart[n] == 15):
									citingTest[14] = citingTest[14] + 1
								elif(testPreStart[n] == 16):
									citingTest[15] = citingTest[15] + 1
								elif(testPreStart[n] == 17):
									citingTest[16] = citingTest[16] + 1
								elif(testPreStart[n] == 18):
									citingTest[17] = citingTest[17] + 1
								elif(testPreStart[n] == 19):
									citingTest[18] = citingTest[18] + 1
								break
					for p in range(len(dataList[test_indices[i]][3][1])):
						for q in range(len(train_indices)):
							if(dataList[test_indices[i]][3][1][p] == dataList[train_indices[q]][0]):
								if(dataList[train_indices[q]][1] == 1):
									citeTest[0] = citeTest[0] +1
								elif(dataList[train_indices[q]][1] == 2):
									citeTest[1] = citeTest[1] +1
								elif(dataList[train_indices[q]][1] == 3):
									citeTest[2] = citeTest[2] +1
								elif(dataList[train_indices[q]][1] == 4):
									citeTest[3] = citeTest[3] +1
								elif(dataList[train_indices[q]][1] == 5):
									citeTest[4] = citeTest[4] +1
								elif(dataList[train_indices[q]][1] == 6):
									citeTest[5] = citeTest[5] +1
								elif(dataList[train_indices[q]][1] == 7):
									citeTest[6] = citeTest[6] +1
								elif(dataList[train_indices[q]][1] == 8):
									citeTest[7] = citeTest[7] +1
								elif(dataList[train_indices[q]][1] == 9):
									citeTest[8] = citeTest[8] +1
								elif(dataList[train_indices[q]][1] == 10):
									citeTest[9] = citeTest[9] +1
								elif(dataList[train_indices[q]][1] == 11):
									citeTest[10] = citeTest[10] +1
								elif(dataList[train_indices[q]][1] == 12):
									citeTest[11] = citeTest[11] +1
								elif(dataList[train_indices[q]][1] == 13):
									citeTest[12] = citeTest[12] +1
								elif(dataList[train_indices[q]][1] == 14):
									citeTest[13] = citeTest[13] +1
								elif(dataList[train_indices[q]][1] == 15):
									citeTest[14] = citeTest[14] +1
								elif(dataList[train_indices[q]][1] == 16):
									citeTest[15] = citeTest[15] +1
								elif(dataList[train_indices[q]][1] == 17):
									citeTest[16] = citeTest[16] +1
								elif(dataList[train_indices[q]][1] == 18):
									citeTest[17] = citeTest[17] +1
								elif(dataList[train_indices[q]][1] == 19):
									citeTest[18] = citeTest[18] +1
								break
						for q in range(len(test_indices)):
							if(dataList[test_indices[i]][3][1][p] == dataList[test_indices[q]][0]):
								if(testPreStart[q] == 1):
									citeTest[0] = citeTest[0] + 1
								elif(testPreStart[q] == 2):
									citeTest[1] = citeTest[1] + 1
								elif(testPreStart[q] == 3):
									citeTest[2] = citeTest[2] + 1
								elif(testPreStart[q] == 4):
									citeTest[3] = citeTest[3] + 1
								elif(testPreStart[q] == 5):
									citeTest[4] = citeTest[4] + 1
								elif(testPreStart[q] == 6):
									citeTest[5] = citeTest[5] + 1
								elif(testPreStart[q] == 7):
									citeTest[6] = citeTest[6] + 1
								elif(testPreStart[q] == 8):
									citeTest[7] = citeTest[7] + 1
								elif(testPreStart[q] == 9):
									citeTest[8] = citeTest[8] + 1
								elif(testPreStart[q] == 10):
									citeTest[9] = citeTest[9] + 1
								elif(testPreStart[q] == 11):
									citeTest[10] = citeTest[10] + 1
								elif(testPreStart[q] == 12):
									citeTest[11] = citeTest[11] + 1
								elif(testPreStart[q] == 13):
									citeTest[12] = citeTest[12] + 1
								elif(testPreStart[q] == 14):
									citeTest[13] = citeTest[13] + 1
								elif(testPreStart[q] == 15):
									citeTest[14] = citeTest[14] + 1
								elif(testPreStart[q] == 16):
									citeTest[15] = citeTest[15] + 1
								elif(testPreStart[q] == 17):
									citeTest[16] = citeTest[16] + 1
								elif(testPreStart[q] == 18):
									citeTest[17] = citeTest[17] + 1
								elif(testPreStart[q] == 19):
									citeTest[18] = citeTest[18] + 1
								break
					for u in range(19):
						totalTest = totalTest + (citeTest[u]+citingTest[u])
#						totalCitedTest = totalCitedTest + citeTest[u]
#						totalCitingTest = totalCitingTest + citingTest[u]]
					for u in range(19):
#						if(index_test == 0):
#							countTestAttUnd[i].append(float(citeTest[u]+citingTest[u]))	
#						else:
#							countTestAttUnd[i][3703+u] = float(citeTest[u]+citingTest[u])
#						countTestAttDir[i].append(citeTest[u])
						if(index_test == 0):
							if(totalTest == 0):
								proporTestAttUnd[i].append(0)
							else:
								proporTestAttUnd[i].append(round((citeTest[u]+citingTest[u])/totalTest,3))
						else:
							if(totalTest == 0):
								proporTestAttUnd[i][4973+u] = 0
							else:
								proporTestAttUnd[i][4973+u] = round((citeTest[u]+citingTest[u])/totalTest,3)
#						if(totalCited == 0):
#							proporTestAttDir[i].append(0)
#						else:
#							proporTestAttDir[i].append(round(citeTest[u]/totalCitedTest,3))
#						if(citeTest[u]+citingTest[u] == 0):
#							existTestAttUnd[i].append(0)
#						else:
#							existTestAttUnd[i].append(1)
#						if(citeTest[u] == 0):
#							existTestAttDir[i].append(0)
#						else:
#							existTestAttDir[i].append(1)
#						if((citeTest[u]+citingTest[u]) > maxTotalTest):
#							indexTest = u	
#							maxTotalTest = citeTest[u]+citingTest[u]
#						if(citeTest[u] > maxCitedTest):
#							indexCitedTest = u
#							maxCitedTest = citeTest[u]
#					modeAttUnd[i].append(indexTest)
#					modeAttDir[i].append(indexCitedTest)
#					for u in range(6):
#						countTestAttDir[i].append(citingTest[u])
#						if(totalCitingTest == 0):
#							proporTestAttDir[i].append(0)
#						else:
#							proporTestAttDir[i].append(round(citingTest[u]/totalCitingTest,3))
#						if(citingTest[u] == 0):
#								existTestAttDir[i].append(0)
#						else:
#								existTestAttDir[i].append(1)
#						if(citingTest[u] > maxCitingTest):
#							indexCitingTest = u
#							maxCitingTest = citingTest[u]
#					modeTestAttDir[i].append(indexCitingTest)
			index_test = index_test + 1
			m_test = np.array(proporTestAttUnd,dtype='f')	
			testPreNew = clf_proUnd.predict(m_test)		
			if(np.allclose(testPreNew,testPreStart)):
				break
			else:
				testPreStart = testPreNew
		scoreProUnd = scoreProUnd + clf_proUnd.score(m_test,n_test)
		print(scoreProUnd)



		index_test = 0
		testPreStart = testPre
		testPreNew =  testPre
		while( 0 <= index_test <10):
			for i in range(len(test_indices)):
					citeTest = []
					citingTest = []
					for m in range(19):
						citeTest.append(0)
						citingTest.append(0)
					totalTest = totalCitedTest = totalCitingTest = 0.0
					maxTotalTest = maxCitedTest = maxCitingTest = 0
					indexTest = indexCitedTest = indexCitingTest = 0
					for m in range(len(dataList[test_indices[i]][3][0])):
						for n in range(len(train_indices)):
							if(dataList[test_indices[i]][3][0][m] == dataList[train_indices[n]][0]):
								if(dataList[train_indices[n]][1] == 1):
									citingTest[0] = citingTest[0] + 1
								elif(dataList[train_indices[n]][1] == 2):
									citingTest[1] = citingTest[1] + 1
								elif(dataList[train_indices[n]][1] == 3):
									citingTest[2] = citingTest[2] + 1
								elif(dataList[train_indices[n]][1] == 4):
									citingTest[3] = citingTest[3] + 1
								elif(dataList[train_indices[n]][1] == 5):
									citingTest[4] = citingTest[4] + 1
								elif(dataList[train_indices[n]][1] == 6):
									citingTest[5] = citingTest[5] + 1
								elif(dataList[train_indices[n]][1] == 7):
									citingTest[6] = citingTest[6] + 1
								elif(dataList[train_indices[n]][1] == 8):
									citingTest[7] = citingTest[7] + 1
								elif(dataList[train_indices[n]][1] == 9):
									citingTest[8] = citingTest[8] + 1
								elif(dataList[train_indices[n]][1] == 10):
									citingTest[9] = citingTest[9] + 1
								elif(dataList[train_indices[n]][1] == 11):
									citingTest[10] = citingTest[10] + 1
								elif(dataList[train_indices[n]][1] == 12):
									citingTest[11] = citingTest[11] + 1
								elif(dataList[train_indices[n]][1] == 13):
									citingTest[12] = citingTest[12] + 1
								elif(dataList[train_indices[n]][1] == 14):
									citingTest[13] = citingTest[13] + 1
								elif(dataList[train_indices[n]][1] == 15):
									citingTest[14] = citingTest[14] + 1
								elif(dataList[train_indices[n]][1] == 16):
									citingTest[15] = citingTest[15] + 1
								elif(dataList[train_indices[n]][1] == 17):
									citingTest[16] = citingTest[16] + 1
								elif(dataList[train_indices[n]][1] == 18):
									citingTest[17] = citingTest[17] + 1
								elif(dataList[train_indices[n]][1] == 19):
									citingTest[18] = citingTest[18] + 1
								break
						for n in range(len(test_indices)):
							if(dataList[test_indices[i]][3][0][m] == dataList[test_indices[n]][0]):
								if(testPreStart[n] == 1):
									citingTest[0] = citingTest[0] + 1
								elif(testPreStart[n] == 2):
									citingTest[1] = citingTest[1] + 1
								elif(testPreStart[n] == 3):
									citingTest[2] = citingTest[2] + 1
								elif(testPreStart[n] == 4):
									citingTest[3] = citingTest[3] + 1
								elif(testPreStart[n] == 5):
									citingTest[4] = citingTest[4] + 1
								elif(testPreStart[n] == 6):
									citingTest[5] = citingTest[5] + 1
								elif(testPreStart[n] == 7):
									citingTest[6] = citingTest[6] + 1
								elif(testPreStart[n] == 8):
									citingTest[7] = citingTest[7] + 1
								elif(testPreStart[n] == 9):
									citingTest[8] = citingTest[8] + 1
								elif(testPreStart[n] == 10):
									citingTest[9] = citingTest[9] + 1
								elif(testPreStart[n] == 11):
									citingTest[10] = citingTest[10] + 1
								elif(testPreStart[n] == 12):
									citingTest[11] = citingTest[11] + 1
								elif(testPreStart[n] == 13):
									citingTest[12] = citingTest[12] + 1
								elif(testPreStart[n] == 14):
									citingTest[13] = citingTest[13] + 1
								elif(testPreStart[n] == 15):
									citingTest[14] = citingTest[14] + 1
								elif(testPreStart[n] == 16):
									citingTest[15] = citingTest[15] + 1
								elif(testPreStart[n] == 17):
									citingTest[16] = citingTest[16] + 1
								elif(testPreStart[n] == 18):
									citingTest[17] = citingTest[17] + 1
								elif(testPreStart[n] == 19):
									citingTest[18] = citingTest[18] + 1
								break
					for p in range(len(dataList[test_indices[i]][3][1])):
						for q in range(len(train_indices)):
							if(dataList[test_indices[i]][3][1][p] == dataList[train_indices[q]][0]):
								if(dataList[train_indices[q]][1] == 1):
									citeTest[0] = citeTest[0] +1
								elif(dataList[train_indices[q]][1] == 2):
									citeTest[1] = citeTest[1] +1
								elif(dataList[train_indices[q]][1] == 3):
									citeTest[2] = citeTest[2] +1
								elif(dataList[train_indices[q]][1] == 4):
									citeTest[3] = citeTest[3] +1
								elif(dataList[train_indices[q]][1] == 5):
									citeTest[4] = citeTest[4] +1
								elif(dataList[train_indices[q]][1] == 6):
									citeTest[5] = citeTest[5] +1
								elif(dataList[train_indices[q]][1] == 7):
									citeTest[6] = citeTest[6] +1
								elif(dataList[train_indices[q]][1] == 8):
									citeTest[7] = citeTest[7] +1
								elif(dataList[train_indices[q]][1] == 9):
									citeTest[8] = citeTest[8] +1
								elif(dataList[train_indices[q]][1] == 10):
									citeTest[9] = citeTest[9] +1
								elif(dataList[train_indices[q]][1] == 11):
									citeTest[10] = citeTest[10] +1
								elif(dataList[train_indices[q]][1] == 12):
									citeTest[11] = citeTest[11] +1
								elif(dataList[train_indices[q]][1] == 13):
									citeTest[12] = citeTest[12] +1
								elif(dataList[train_indices[q]][1] == 14):
									citeTest[13] = citeTest[13] +1
								elif(dataList[train_indices[q]][1] == 15):
									citeTest[14] = citeTest[14] +1
								elif(dataList[train_indices[q]][1] == 16):
									citeTest[15] = citeTest[15] +1
								elif(dataList[train_indices[q]][1] == 17):
									citeTest[16] = citeTest[16] +1
								elif(dataList[train_indices[q]][1] == 18):
									citeTest[17] = citeTest[17] +1
								elif(dataList[train_indices[q]][1] == 19):
									citeTest[18] = citeTest[18] +1
								break
						for q in range(len(test_indices)):
							if(dataList[test_indices[i]][3][1][p] == dataList[test_indices[q]][0]):
								if(testPreStart[q] == 1):
									citeTest[0] = citeTest[0] + 1
								elif(testPreStart[q] == 2):
									citeTest[1] = citeTest[1] + 1
								elif(testPreStart[q] == 3):
									citeTest[2] = citeTest[2] + 1
								elif(testPreStart[q] == 4):
									citeTest[3] = citeTest[3] + 1
								elif(testPreStart[q] == 5):
									citeTest[4] = citeTest[4] + 1
								elif(testPreStart[q] == 6):
									citeTest[5] = citeTest[5] + 1
								elif(testPreStart[q] == 7):
									citeTest[6] = citeTest[6] + 1
								elif(testPreStart[q] == 8):
									citeTest[7] = citeTest[7] + 1
								elif(testPreStart[q] == 9):
									citeTest[8] = citeTest[8] + 1
								elif(testPreStart[q] == 10):
									citeTest[9] = citeTest[9] + 1
								elif(testPreStart[q] == 11):
									citeTest[10] = citeTest[10] + 1
								elif(testPreStart[q] == 12):
									citeTest[11] = citeTest[11] + 1
								elif(testPreStart[q] == 13):
									citeTest[12] = citeTest[12] + 1
								elif(testPreStart[q] == 14):
									citeTest[13] = citeTest[13] + 1
								elif(testPreStart[q] == 15):
									citeTest[14] = citeTest[14] + 1
								elif(testPreStart[q] == 16):
									citeTest[15] = citeTest[15] + 1
								elif(testPreStart[q] == 17):
									citeTest[16] = citeTest[16] + 1
								elif(testPreStart[q] == 18):
									citeTest[17] = citeTest[17] + 1
								elif(testPreStart[q] == 19):
									citeTest[18] = citeTest[18] + 1
								break
					for u in range(19):
#						totalTest = totalTest + (citeTest[u]+citingTest[u])
						totalCitedTest = totalCitedTest + citeTest[u]
						totalCitingTest = totalCitingTest + citingTest[u]
					for u in range(19):
#						if(index_test == 0):
#							countTestAttUnd[i].append(float(citeTest[u]+citingTest[u]))	
#						else:
#							countTestAttUnd[i][3703+u] = float(citeTest[u]+citingTest[u])
#						countTestAttDir[i].append(citeTest[u])
#						if(index_test == 0):
#							if(totalTest == 0):
#								proporTestAttUnd[i].append(0)
#							else:
#								proporTestAttUnd[i].append(round((citeTest[u]+citingTest[u])/totalTest,3))
#						else:
#							if(totalTest == 0):
#								proporTestAttUnd[i][3703+u] = 0
#							else:
#								proporTestAttUnd[i][3703+u] = round((citeTest[u]+citingTest[u])/totalTest,3)
						if(index_test == 0):
							if(totalCitedTest == 0):
								proporTestAttDir[i].append(0)
							else:
								proporTestAttDir[i].append(round(citeTest[u]/totalCitedTest,3))
						else:
							if(totalCitedTest == 0):
								proporTestAttDir[i][4973+u] = 0
							else:
								proporTestAttDir[i][4973+u] = round(citeTest[u]/totalCitedTest,3)

#						if(citeTest[u]+citingTest[u] == 0):
#							existTestAttUnd[i].append(0)
#						else:
#							existTestAttUnd[i].append(1)
#						if(citeTest[u] == 0):
#							existTestAttDir[i].append(0)
#						else:
#							existTestAttDir[i].append(1)
#						if((citeTest[u]+citingTest[u]) > maxTotalTest):
#							indexTest = u	
#							maxTotalTest = citeTest[u]+citingTest[u]
#						if(citeTest[u] > maxCitedTest):
#							indexCitedTest = u
#							maxCitedTest = citeTest[u]
#					modeAttUnd[i].append(indexTest)
#					modeAttDir[i].append(indexCitedTest)
					for u in range(19):
#						countTestAttDir[i].append(citingTest[u])
						if(index_test == 0):
							if(totalCitingTest == 0):
								proporTestAttDir[i].append(0)
							else:
								proporTestAttDir[i].append(round(citingTest[u]/totalCitingTest,3))
						else:
							if(totalCitingTest == 0):
								proporTestAttDir[i][4992+u] = 0
							else:
								proporTestAttDir[i][4992+u] = round(citingTest[u]/totalCitingTest,3)
#						if(citingTest[u] == 0):
#								existTestAttDir[i].append(0)
#						else:
#								existTestAttDir[i].append(1)
#						if(citingTest[u] > maxCitingTest):
#							indexCitingTest = u
#							maxCitingTest = citingTest[u]
#					modeTestAttDir[i].append(indexCitingTest)
			index_test = index_test + 1
			m_test = np.array(proporTestAttDir,dtype='f')	
			testPreNew = clf_proDir.predict(m_test)		
			if(np.allclose(testPreNew,testPreStart)):
				break
			else:
				testPreStart = testPreNew
		scoreProDir = scoreProDir + clf_proDir.score(m_test,n_test)
		print(scoreProDir)




		index_test = 0
		testPreStart = testPre
		testPreNew =  testPre
		while( 0 <= index_test <10):
			for i in range(len(test_indices)):
					citeTest = []
					citingTest = []
					for m in range(19):
						citeTest.append(0)
						citingTest.append(0)
#					totalTest = totalCitedTest = totalCitingTest = 0.0
#					maxTotalTest = maxCitedTest = maxCitingTest = 0
#					indexTest = indexCitedTest = indexCitingTest = 0
					for m in range(len(dataList[test_indices[i]][3][0])):
						for n in range(len(train_indices)):
							if(dataList[test_indices[i]][3][0][m] == dataList[train_indices[n]][0]):
								if(dataList[train_indices[n]][1] == 1):
									citingTest[0] = citingTest[0] + 1
								elif(dataList[train_indices[n]][1] == 2):
									citingTest[1] = citingTest[1] + 1
								elif(dataList[train_indices[n]][1] == 3):
									citingTest[2] = citingTest[2] + 1
								elif(dataList[train_indices[n]][1] == 4):
									citingTest[3] = citingTest[3] + 1
								elif(dataList[train_indices[n]][1] == 5):
									citingTest[4] = citingTest[4] + 1
								elif(dataList[train_indices[n]][1] == 6):
									citingTest[5] = citingTest[5] + 1
								elif(dataList[train_indices[n]][1] == 7):
									citingTest[6] = citingTest[6] + 1
								elif(dataList[train_indices[n]][1] == 8):
									citingTest[7] = citingTest[7] + 1
								elif(dataList[train_indices[n]][1] == 9):
									citingTest[8] = citingTest[8] + 1
								elif(dataList[train_indices[n]][1] == 10):
									citingTest[9] = citingTest[9] + 1
								elif(dataList[train_indices[n]][1] == 11):
									citingTest[10] = citingTest[10] + 1
								elif(dataList[train_indices[n]][1] == 12):
									citingTest[11] = citingTest[11] + 1
								elif(dataList[train_indices[n]][1] == 13):
									citingTest[12] = citingTest[12] + 1
								elif(dataList[train_indices[n]][1] == 14):
									citingTest[13] = citingTest[13] + 1
								elif(dataList[train_indices[n]][1] == 15):
									citingTest[14] = citingTest[14] + 1
								elif(dataList[train_indices[n]][1] == 16):
									citingTest[15] = citingTest[15] + 1
								elif(dataList[train_indices[n]][1] == 17):
									citingTest[16] = citingTest[16] + 1
								elif(dataList[train_indices[n]][1] == 18):
									citingTest[17] = citingTest[17] + 1
								elif(dataList[train_indices[n]][1] == 19):
									citingTest[18] = citingTest[18] + 1
								break
						for n in range(len(test_indices)):
							if(dataList[test_indices[i]][3][0][m] == dataList[test_indices[n]][0]):
								if(testPreStart[n] == 1):
									citingTest[0] = citingTest[0] + 1
								elif(testPreStart[n] == 2):
									citingTest[1] = citingTest[1] + 1
								elif(testPreStart[n] == 3):
									citingTest[2] = citingTest[2] + 1
								elif(testPreStart[n] == 4):
									citingTest[3] = citingTest[3] + 1
								elif(testPreStart[n] == 5):
									citingTest[4] = citingTest[4] + 1
								elif(testPreStart[n] == 6):
									citingTest[5] = citingTest[5] + 1
								elif(testPreStart[n] == 7):
									citingTest[6] = citingTest[6] + 1
								elif(testPreStart[n] == 8):
									citingTest[7] = citingTest[7] + 1
								elif(testPreStart[n] == 9):
									citingTest[8] = citingTest[8] + 1
								elif(testPreStart[n] == 10):
									citingTest[9] = citingTest[9] + 1
								elif(testPreStart[n] == 11):
									citingTest[10] = citingTest[10] + 1
								elif(testPreStart[n] == 12):
									citingTest[11] = citingTest[11] + 1
								elif(testPreStart[n] == 13):
									citingTest[12] = citingTest[12] + 1
								elif(testPreStart[n] == 14):
									citingTest[13] = citingTest[13] + 1
								elif(testPreStart[n] == 15):
									citingTest[14] = citingTest[14] + 1
								elif(testPreStart[n] == 16):
									citingTest[15] = citingTest[15] + 1
								elif(testPreStart[n] == 17):
									citingTest[16] = citingTest[16] + 1
								elif(testPreStart[n] == 18):
									citingTest[17] = citingTest[17] + 1
								elif(testPreStart[n] == 19):
									citingTest[18] = citingTest[18] + 1
								break
					for p in range(len(dataList[test_indices[i]][3][1])):
						for q in range(len(train_indices)):
							if(dataList[test_indices[i]][3][1][p] == dataList[train_indices[q]][0]):
								if(dataList[train_indices[q]][1] == 1):
									citeTest[0] = citeTest[0] +1
								elif(dataList[train_indices[q]][1] == 2):
									citeTest[1] = citeTest[1] +1
								elif(dataList[train_indices[q]][1] == 3):
									citeTest[2] = citeTest[2] +1
								elif(dataList[train_indices[q]][1] == 4):
									citeTest[3] = citeTest[3] +1
								elif(dataList[train_indices[q]][1] == 5):
									citeTest[4] = citeTest[4] +1
								elif(dataList[train_indices[q]][1] == 6):
									citeTest[5] = citeTest[5] +1
								elif(dataList[train_indices[q]][1] == 7):
									citeTest[6] = citeTest[6] +1
								elif(dataList[train_indices[q]][1] == 8):
									citeTest[7] = citeTest[7] +1
								elif(dataList[train_indices[q]][1] == 9):
									citeTest[8] = citeTest[8] +1
								elif(dataList[train_indices[q]][1] == 10):
									citeTest[9] = citeTest[9] +1
								elif(dataList[train_indices[q]][1] == 11):
									citeTest[10] = citeTest[10] +1
								elif(dataList[train_indices[q]][1] == 12):
									citeTest[11] = citeTest[11] +1
								elif(dataList[train_indices[q]][1] == 13):
									citeTest[12] = citeTest[12] +1
								elif(dataList[train_indices[q]][1] == 14):
									citeTest[13] = citeTest[13] +1
								elif(dataList[train_indices[q]][1] == 15):
									citeTest[14] = citeTest[14] +1
								elif(dataList[train_indices[q]][1] == 16):
									citeTest[15] = citeTest[15] +1
								elif(dataList[train_indices[q]][1] == 17):
									citeTest[16] = citeTest[16] +1
								elif(dataList[train_indices[q]][1] == 18):
									citeTest[17] = citeTest[17] +1
								elif(dataList[train_indices[q]][1] == 19):
									citeTest[18] = citeTest[18] +1
								break
						for q in range(len(test_indices)):
							if(dataList[test_indices[i]][3][1][p] == dataList[test_indices[q]][0]):
								if(testPreStart[q] == 1):
									citeTest[0] = citeTest[0] + 1
								elif(testPreStart[q] == 2):
									citeTest[1] = citeTest[1] + 1
								elif(testPreStart[q] == 3):
									citeTest[2] = citeTest[2] + 1
								elif(testPreStart[q] == 4):
									citeTest[3] = citeTest[3] + 1
								elif(testPreStart[q] == 5):
									citeTest[4] = citeTest[4] + 1
								elif(testPreStart[q] == 6):
									citeTest[5] = citeTest[5] + 1
								elif(testPreStart[q] == 7):
									citeTest[6] = citeTest[6] + 1
								elif(testPreStart[q] == 8):
									citeTest[7] = citeTest[7] + 1
								elif(testPreStart[q] == 9):
									citeTest[8] = citeTest[8] + 1
								elif(testPreStart[q] == 10):
									citeTest[9] = citeTest[9] + 1
								elif(testPreStart[q] == 11):
									citeTest[10] = citeTest[10] + 1
								elif(testPreStart[q] == 12):
									citeTest[11] = citeTest[11] + 1
								elif(testPreStart[q] == 13):
									citeTest[12] = citeTest[12] + 1
								elif(testPreStart[q] == 14):
									citeTest[13] = citeTest[13] + 1
								elif(testPreStart[q] == 15):
									citeTest[14] = citeTest[14] + 1
								elif(testPreStart[q] == 16):
									citeTest[15] = citeTest[15] + 1
								elif(testPreStart[q] == 17):
									citeTest[16] = citeTest[16] + 1
								elif(testPreStart[q] == 18):
									citeTest[17] = citeTest[17] + 1
								elif(testPreStart[q] == 19):
									citeTest[18] = citeTest[18] + 1
								break
#					for u in range(6):
#						totalTest = totalTest + (citeTest[u]+citingTest[u])
#						totalCitedTest = totalCitedTest + citeTest[u]
#						totalCitingTest = totalCitingTest + citingTest[u]]
					for u in range(19):
#						if(index_test == 0):
#							countTestAttUnd[i].append(float(citeTest[u]+citingTest[u]))	
#						else:
#							countTestAttUnd[i][3703+u] = float(citeTest[u]+citingTest[u])
#						countTestAttDir[i].append(citeTest[u])
#						if(index_test == 0):
#							if(totalTest == 0):
#								proporTestAttUnd[i].append(0)
#							else:
#								proporTestAttUnd[i].append(round((citeTest[u]+citingTest[u])/totalTest,3))
#						else:
#							if(totalTest == 0):
#								proporTestAttUnd[i][3703+u] = 0
#							else:
#								proporTestAttUnd[i][3703+u] = round((citeTest[u]+citingTest[u])/totalTest,3)
#						if(index_test == 0):
#							if(totalCitedTest == 0):
#								proporTestAttDir[i].append(0)
#							else:
#								proporTestAttDir[i].append(round(citeTest[u]/totalCitedTest,3))
#						else:
#							if(totalCitedTest == 0):
#								proporTestAttDir[i][3703+u] = 0
#							else:
#								proporTestAttDir[i][3703+u] = round(citeTest[u]/totalCitedTest,3)
						if(index_test == 0):
							if(citeTest[u]+citingTest[u] == 0):
								existTestAttUnd[i].append(0)
							else:
								existTestAttUnd[i].append(1)
						else:
							if(citeTest[u]+citingTest[u] == 0):
								existTestAttUnd[i][4973+u] = 0
							else:
								existTestAttUnd[i][4973+u] = 1
#						if(citeTest[u] == 0):
#							existTestAttDir[i].append(0)
#						else:
#							existTestAttDir[i].append(1)
#						if((citeTest[u]+citingTest[u]) > maxTotalTest):
#							indexTest = u	
#							maxTotalTest = citeTest[u]+citingTest[u]
#						if(citeTest[u] > maxCitedTest):
#							indexCitedTest = u
#							maxCitedTest = citeTest[u]
#					modeAttUnd[i].append(indexTest)
#					modeAttDir[i].append(indexCitedTest)
#					for u in range(6):
#						countTestAttDir[i].append(citingTest[u])
#						if(index_test == 0):
#							if(totalCitingTest == 0):
#								proporTestAttDir[i].append(0)
#							else:
#								proporTestAttDir[i].append(round(citingTest[u]/totalCitingTest,3))
#						else:
#							if(totalCitedTest == 0):
#								proporTestAttDir[i][3709+u] = 0
#							else:
#								proporTestAttDir[i][3703+u] = round(citingTest[u]/totalCitingTest,3)
#						if(citingTest[u] == 0):
#								existTestAttDir[i].append(0)
#						else:
#								existTestAttDir[i].append(1)
#						if(citingTest[u] > maxCitingTest):
#							indexCitingTest = u
#							maxCitingTest = citingTest[u]
#					modeTestAttDir[i].append(indexCitingTest)
			index_test = index_test + 1
			m_test = np.array(existTestAttUnd,dtype='f')	
			testPreNew = clf_existUnd.predict(m_test)		
			if(np.allclose(testPreNew,testPreStart)):
				break
			else:
				testPreStart = testPreNew
		scoreExistUnd = scoreExistUnd + clf_existUnd.score(m_test,n_test)
		print(scoreExistUnd)



		index_test = 0
		testPreStart = testPre
		testPreNew =  testPre
		while( 0 <= index_test <10):
			for i in range(len(test_indices)):
					citeTest = []
					citingTest = []
					for m in range(19):
						citeTest.append(0)
						citingTest.append(0)
#					totalTest = totalCitedTest = totalCitingTest = 0.0
#					maxTotalTest = maxCitedTest = maxCitingTest = 0
#					indexTest = indexCitedTest = indexCitingTest = 0
					for m in range(len(dataList[test_indices[i]][3][0])):
						for n in range(len(train_indices)):
							if(dataList[test_indices[i]][3][0][m] == dataList[train_indices[n]][0]):
								if(dataList[train_indices[n]][1] == 1):
									citingTest[0] = citingTest[0] + 1
								elif(dataList[train_indices[n]][1] == 2):
									citingTest[1] = citingTest[1] + 1
								elif(dataList[train_indices[n]][1] == 3):
									citingTest[2] = citingTest[2] + 1
								elif(dataList[train_indices[n]][1] == 4):
									citingTest[3] = citingTest[3] + 1
								elif(dataList[train_indices[n]][1] == 5):
									citingTest[4] = citingTest[4] + 1
								elif(dataList[train_indices[n]][1] == 6):
									citingTest[5] = citingTest[5] + 1
								elif(dataList[train_indices[n]][1] == 7):
									citingTest[6] = citingTest[6] + 1
								elif(dataList[train_indices[n]][1] == 8):
									citingTest[7] = citingTest[7] + 1
								elif(dataList[train_indices[n]][1] == 9):
									citingTest[8] = citingTest[8] + 1
								elif(dataList[train_indices[n]][1] == 10):
									citingTest[9] = citingTest[9] + 1
								elif(dataList[train_indices[n]][1] == 11):
									citingTest[10] = citingTest[10] + 1
								elif(dataList[train_indices[n]][1] == 12):
									citingTest[11] = citingTest[11] + 1
								elif(dataList[train_indices[n]][1] == 13):
									citingTest[12] = citingTest[12] + 1
								elif(dataList[train_indices[n]][1] == 14):
									citingTest[13] = citingTest[13] + 1
								elif(dataList[train_indices[n]][1] == 15):
									citingTest[14] = citingTest[14] + 1
								elif(dataList[train_indices[n]][1] == 16):
									citingTest[15] = citingTest[15] + 1
								elif(dataList[train_indices[n]][1] == 17):
									citingTest[16] = citingTest[16] + 1
								elif(dataList[train_indices[n]][1] == 18):
									citingTest[17] = citingTest[17] + 1
								elif(dataList[train_indices[n]][1] == 19):
									citingTest[18] = citingTest[18] + 1
								break
						for n in range(len(test_indices)):
							if(dataList[test_indices[i]][3][0][m] == dataList[test_indices[n]][0]):
								if(testPreStart[n] == 1):
									citingTest[0] = citingTest[0] + 1
								elif(testPreStart[n] == 2):
									citingTest[1] = citingTest[1] + 1
								elif(testPreStart[n] == 3):
									citingTest[2] = citingTest[2] + 1
								elif(testPreStart[n] == 4):
									citingTest[3] = citingTest[3] + 1
								elif(testPreStart[n] == 5):
									citingTest[4] = citingTest[4] + 1
								elif(testPreStart[n] == 6):
									citingTest[5] = citingTest[5] + 1
								elif(testPreStart[n] == 7):
									citingTest[6] = citingTest[6] + 1
								elif(testPreStart[n] == 8):
									citingTest[7] = citingTest[7] + 1
								elif(testPreStart[n] == 9):
									citingTest[8] = citingTest[8] + 1
								elif(testPreStart[n] == 10):
									citingTest[9] = citingTest[9] + 1
								elif(testPreStart[n] == 11):
									citingTest[10] = citingTest[10] + 1
								elif(testPreStart[n] == 12):
									citingTest[11] = citingTest[11] + 1
								elif(testPreStart[n] == 13):
									citingTest[12] = citingTest[12] + 1
								elif(testPreStart[n] == 14):
									citingTest[13] = citingTest[13] + 1
								elif(testPreStart[n] == 15):
									citingTest[14] = citingTest[14] + 1
								elif(testPreStart[n] == 16):
									citingTest[15] = citingTest[15] + 1
								elif(testPreStart[n] == 17):
									citingTest[16] = citingTest[16] + 1
								elif(testPreStart[n] == 18):
									citingTest[17] = citingTest[17] + 1
								elif(testPreStart[n] == 19):
									citingTest[18] = citingTest[18] + 1
								break
					for p in range(len(dataList[test_indices[i]][3][1])):
						for q in range(len(train_indices)):
							if(dataList[test_indices[i]][3][1][p] == dataList[train_indices[q]][0]):
								if(dataList[train_indices[q]][1] == 1):
									citeTest[0] = citeTest[0] +1
								elif(dataList[train_indices[q]][1] == 2):
									citeTest[1] = citeTest[1] +1
								elif(dataList[train_indices[q]][1] == 3):
									citeTest[2] = citeTest[2] +1
								elif(dataList[train_indices[q]][1] == 4):
									citeTest[3] = citeTest[3] +1
								elif(dataList[train_indices[q]][1] == 5):
									citeTest[4] = citeTest[4] +1
								elif(dataList[train_indices[q]][1] == 6):
									citeTest[5] = citeTest[5] +1
								elif(dataList[train_indices[q]][1] == 7):
									citeTest[6] = citeTest[6] +1
								elif(dataList[train_indices[q]][1] == 8):
									citeTest[7] = citeTest[7] +1
								elif(dataList[train_indices[q]][1] == 9):
									citeTest[8] = citeTest[8] +1
								elif(dataList[train_indices[q]][1] == 10):
									citeTest[9] = citeTest[9] +1
								elif(dataList[train_indices[q]][1] == 11):
									citeTest[10] = citeTest[10] +1
								elif(dataList[train_indices[q]][1] == 12):
									citeTest[11] = citeTest[11] +1
								elif(dataList[train_indices[q]][1] == 13):
									citeTest[12] = citeTest[12] +1
								elif(dataList[train_indices[q]][1] == 14):
									citeTest[13] = citeTest[13] +1
								elif(dataList[train_indices[q]][1] == 15):
									citeTest[14] = citeTest[14] +1
								elif(dataList[train_indices[q]][1] == 16):
									citeTest[15] = citeTest[15] +1
								elif(dataList[train_indices[q]][1] == 17):
									citeTest[16] = citeTest[16] +1
								elif(dataList[train_indices[q]][1] == 18):
									citeTest[17] = citeTest[17] +1
								elif(dataList[train_indices[q]][1] == 19):
									citeTest[18] = citeTest[18] +1
								break
						for q in range(len(test_indices)):
							if(dataList[test_indices[i]][3][1][p] == dataList[test_indices[q]][0]):
								if(testPreStart[q] == 1):
									citeTest[0] = citeTest[0] + 1
								elif(testPreStart[q] == 2):
									citeTest[1] = citeTest[1] + 1
								elif(testPreStart[q] == 3):
									citeTest[2] = citeTest[2] + 1
								elif(testPreStart[q] == 4):
									citeTest[3] = citeTest[3] + 1
								elif(testPreStart[q] == 5):
									citeTest[4] = citeTest[4] + 1
								elif(testPreStart[q] == 6):
									citeTest[5] = citeTest[5] + 1
								elif(testPreStart[q] == 7):
									citeTest[6] = citeTest[6] + 1
								elif(testPreStart[q] == 8):
									citeTest[7] = citeTest[7] + 1
								elif(testPreStart[q] == 9):
									citeTest[8] = citeTest[8] + 1
								elif(testPreStart[q] == 10):
									citeTest[9] = citeTest[9] + 1
								elif(testPreStart[q] == 11):
									citeTest[10] = citeTest[10] + 1
								elif(testPreStart[q] == 12):
									citeTest[11] = citeTest[11] + 1
								elif(testPreStart[q] == 13):
									citeTest[12] = citeTest[12] + 1
								elif(testPreStart[q] == 14):
									citeTest[13] = citeTest[13] + 1
								elif(testPreStart[q] == 15):
									citeTest[14] = citeTest[14] + 1
								elif(testPreStart[q] == 16):
									citeTest[15] = citeTest[15] + 1
								elif(testPreStart[q] == 17):
									citeTest[16] = citeTest[16] + 1
								elif(testPreStart[q] == 18):
									citeTest[17] = citeTest[17] + 1
								elif(testPreStart[q] == 19):
									citeTest[18] = citeTest[18] + 1
								break
#					for u in range(6):
#						totalTest = totalTest + (citeTest[u]+citingTest[u])
#						totalCitedTest = totalCitedTest + citeTest[u]
#						totalCitingTest = totalCitingTest + citingTest[u]]
					for u in range(19):
#						if(index_test == 0):
#							countTestAttUnd[i].append(float(citeTest[u]+citingTest[u]))	
#						else:
#							countTestAttUnd[i][3703+u] = float(citeTest[u]+citingTest[u])
#						countTestAttDir[i].append(citeTest[u])
#						if(index_test == 0):
#							if(totalTest == 0):
#								proporTestAttUnd[i].append(0)
#							else:
#								proporTestAttUnd[i].append(round((citeTest[u]+citingTest[u])/totalTest,3))
#						else:
#							if(totalTest == 0):
#								proporTestAttUnd[i][3703+u] = 0
#							else:
#								proporTestAttUnd[i][3703+u] = round((citeTest[u]+citingTest[u])/totalTest,3)
#						if(index_test == 0):
#							if(totalCitedTest == 0):
#								proporTestAttDir[i].append(0)
#							else:
#								proporTestAttDir[i].append(round(citeTest[u]/totalCitedTest,3))
#						else:
#							if(totalCitedTest == 0):
#								proporTestAttDir[i][3703+u] = 0
#							else:
#								proporTestAttDir[i][3703+u] = round(citeTest[u]/totalCitedTest,3)
#						if(test_index == 0):
#							if(citeTest[u]+citingTest[u] == 0):
#								existTestAttUnd[i].append(0)
#							else:
#								existTestAttUnd[i].append(1)
#						else:
#							if(citeTest[u]+citingTest[u] == 0):
#								existTestAttUnd[i][3703+u] = 0
#							else:
#								existTestAttUnd[i][3703+u] = 1
						if(index_test == 0):
							if(citeTest[u] == 0):
								existTestAttDir[i].append(0)
							else:
								existTestAttDir[i].append(1)
						else:
							if(citeTest[u] == 0):
								existTestAttDir[i][4973+u] = 0
							else:
							 	existTestAttDir[i][4973+u] = 1
#						if((citeTest[u]+citingTest[u]) > maxTotalTest):
#							indexTest = u	
#							maxTotalTest = citeTest[u]+citingTest[u]
#						if(citeTest[u] > maxCitedTest):
#							indexCitedTest = u
#							maxCitedTest = citeTest[u]
#					modeAttUnd[i].append(indexTest)
#					modeAttDir[i].append(indexCitedTest)
					for u in range(19):
#						countTestAttDir[i].append(citingTest[u])
#						if(index_test == 0):
#							if(totalCitingTest == 0):
#								proporTestAttDir[i].append(0)
#							else:
#								proporTestAttDir[i].append(round(citingTest[u]/totalCitingTest,3))
#						else:
#							if(totalCitedTest == 0):
#								proporTestAttDir[i][3709+u] = 0
#							else:
#								proporTestAttDir[i][3703+u] = round(citingTest[u]/totalCitingTest,3)
						if(index_test == 0):
							if(citingTest[u] == 0):
								existTestAttDir[i].append(0)
							else:
								existTestAttDir[i].append(1)
						else:
							if(citingTest[u] == 0):
								existTestAttDir[i][4992+u] = 0
							else:
							 	existTestAttDir[i][4992+u] = 1
#						if(citingTest[u] > maxCitingTest):
#							indexCitingTest = u
#							maxCitingTest = citingTest[u]
#					modeTestAttDir[i].append(indexCitingTest)
			index_test = index_test + 1
			m_test = np.array(existTestAttDir,dtype='f')	
			testPreNew = clf_existDir.predict(m_test)		
			if(np.allclose(testPreNew,testPreStart)):
				break
			else:
				testPreStart = testPreNew
		scoreExistDir = scoreExistDir + clf_existDir.score(m_test,n_test)
		print(scoreExistDir)



		index_test = 0
		testPreStart = testPre
		testPreNew =  testPre
		while( 0 <= index_test <10):
			for i in range(len(test_indices)):
					citeTest = []
					citingTest = []
					for m in range(19):
						citeTest.append(0)
						citingTest.append(0)
#					totalTest = totalCitedTest = totalCitingTest = 0.0
					maxTotalTest = maxCitedTest = maxCitingTest = 0
					indexTest = indexCitedTest = indexCitingTest = 0
					for m in range(len(dataList[test_indices[i]][3][0])):
						for n in range(len(train_indices)):
							if(dataList[test_indices[i]][3][0][m] == dataList[train_indices[n]][0]):
								if(dataList[train_indices[n]][1] == 1):
									citingTest[0] = citingTest[0] + 1
								elif(dataList[train_indices[n]][1] == 2):
									citingTest[1] = citingTest[1] + 1
								elif(dataList[train_indices[n]][1] == 3):
									citingTest[2] = citingTest[2] + 1
								elif(dataList[train_indices[n]][1] == 4):
									citingTest[3] = citingTest[3] + 1
								elif(dataList[train_indices[n]][1] == 5):
									citingTest[4] = citingTest[4] + 1
								elif(dataList[train_indices[n]][1] == 6):
									citingTest[5] = citingTest[5] + 1
								elif(dataList[train_indices[n]][1] == 7):
									citingTest[6] = citingTest[6] + 1
								elif(dataList[train_indices[n]][1] == 8):
									citingTest[7] = citingTest[7] + 1
								elif(dataList[train_indices[n]][1] == 9):
									citingTest[8] = citingTest[8] + 1
								elif(dataList[train_indices[n]][1] == 10):
									citingTest[9] = citingTest[9] + 1
								elif(dataList[train_indices[n]][1] == 11):
									citingTest[10] = citingTest[10] + 1
								elif(dataList[train_indices[n]][1] == 12):
									citingTest[11] = citingTest[11] + 1
								elif(dataList[train_indices[n]][1] == 13):
									citingTest[12] = citingTest[12] + 1
								elif(dataList[train_indices[n]][1] == 14):
									citingTest[13] = citingTest[13] + 1
								elif(dataList[train_indices[n]][1] == 15):
									citingTest[14] = citingTest[14] + 1
								elif(dataList[train_indices[n]][1] == 16):
									citingTest[15] = citingTest[15] + 1
								elif(dataList[train_indices[n]][1] == 17):
									citingTest[16] = citingTest[16] + 1
								elif(dataList[train_indices[n]][1] == 18):
									citingTest[17] = citingTest[17] + 1
								elif(dataList[train_indices[n]][1] == 19):
									citingTest[18] = citingTest[18] + 1
								break
						for n in range(len(test_indices)):
							if(dataList[test_indices[i]][3][0][m] == dataList[test_indices[n]][0]):
								if(testPreStart[n] == 1):
									citingTest[0] = citingTest[0] + 1
								elif(testPreStart[n] == 2):
									citingTest[1] = citingTest[1] + 1
								elif(testPreStart[n] == 3):
									citingTest[2] = citingTest[2] + 1
								elif(testPreStart[n] == 4):
									citingTest[3] = citingTest[3] + 1
								elif(testPreStart[n] == 5):
									citingTest[4] = citingTest[4] + 1
								elif(testPreStart[n] == 6):
									citingTest[5] = citingTest[5] + 1
								elif(testPreStart[n] == 7):
									citingTest[6] = citingTest[6] + 1
								elif(testPreStart[n] == 8):
									citingTest[7] = citingTest[7] + 1
								elif(testPreStart[n] == 9):
									citingTest[8] = citingTest[8] + 1
								elif(testPreStart[n] == 10):
									citingTest[9] = citingTest[9] + 1
								elif(testPreStart[n] == 11):
									citingTest[10] = citingTest[10] + 1
								elif(testPreStart[n] == 12):
									citingTest[11] = citingTest[11] + 1
								elif(testPreStart[n] == 13):
									citingTest[12] = citingTest[12] + 1
								elif(testPreStart[n] == 14):
									citingTest[13] = citingTest[13] + 1
								elif(testPreStart[n] == 15):
									citingTest[14] = citingTest[14] + 1
								elif(testPreStart[n] == 16):
									citingTest[15] = citingTest[15] + 1
								elif(testPreStart[n] == 17):
									citingTest[16] = citingTest[16] + 1
								elif(testPreStart[n] == 18):
									citingTest[17] = citingTest[17] + 1
								elif(testPreStart[n] == 19):
									citingTest[18] = citingTest[18] + 1
								break
					for p in range(len(dataList[test_indices[i]][3][1])):
						for q in range(len(train_indices)):
							if(dataList[test_indices[i]][3][1][p] == dataList[train_indices[q]][0]):
								if(dataList[train_indices[q]][1] == 1):
									citeTest[0] = citeTest[0] +1
								elif(dataList[train_indices[q]][1] == 2):
									citeTest[1] = citeTest[1] +1
								elif(dataList[train_indices[q]][1] == 3):
									citeTest[2] = citeTest[2] +1
								elif(dataList[train_indices[q]][1] == 4):
									citeTest[3] = citeTest[3] +1
								elif(dataList[train_indices[q]][1] == 5):
									citeTest[4] = citeTest[4] +1
								elif(dataList[train_indices[q]][1] == 6):
									citeTest[5] = citeTest[5] +1
								elif(dataList[train_indices[q]][1] == 7):
									citeTest[6] = citeTest[6] +1
								elif(dataList[train_indices[q]][1] == 8):
									citeTest[7] = citeTest[7] +1
								elif(dataList[train_indices[q]][1] == 9):
									citeTest[8] = citeTest[8] +1
								elif(dataList[train_indices[q]][1] == 10):
									citeTest[9] = citeTest[9] +1
								elif(dataList[train_indices[q]][1] == 11):
									citeTest[10] = citeTest[10] +1
								elif(dataList[train_indices[q]][1] == 12):
									citeTest[11] = citeTest[11] +1
								elif(dataList[train_indices[q]][1] == 13):
									citeTest[12] = citeTest[12] +1
								elif(dataList[train_indices[q]][1] == 14):
									citeTest[13] = citeTest[13] +1
								elif(dataList[train_indices[q]][1] == 15):
									citeTest[14] = citeTest[14] +1
								elif(dataList[train_indices[q]][1] == 16):
									citeTest[15] = citeTest[15] +1
								elif(dataList[train_indices[q]][1] == 17):
									citeTest[16] = citeTest[16] +1
								elif(dataList[train_indices[q]][1] == 18):
									citeTest[17] = citeTest[17] +1
								elif(dataList[train_indices[q]][1] == 19):
									citeTest[18] = citeTest[18] +1
								break
						for q in range(len(test_indices)):
							if(dataList[test_indices[i]][3][1][p] == dataList[test_indices[q]][0]):
								if(testPreStart[q] == 1):
									citeTest[0] = citeTest[0] + 1
								elif(testPreStart[q] == 2):
									citeTest[1] = citeTest[1] + 1
								elif(testPreStart[q] == 3):
									citeTest[2] = citeTest[2] + 1
								elif(testPreStart[q] == 4):
									citeTest[3] = citeTest[3] + 1
								elif(testPreStart[q] == 5):
									citeTest[4] = citeTest[4] + 1
								elif(testPreStart[q] == 6):
									citeTest[5] = citeTest[5] + 1
								elif(testPreStart[q] == 7):
									citeTest[6] = citeTest[6] + 1
								elif(testPreStart[q] == 8):
									citeTest[7] = citeTest[7] + 1
								elif(testPreStart[q] == 9):
									citeTest[8] = citeTest[8] + 1
								elif(testPreStart[q] == 10):
									citeTest[9] = citeTest[9] + 1
								elif(testPreStart[q] == 11):
									citeTest[10] = citeTest[10] + 1
								elif(testPreStart[q] == 12):
									citeTest[11] = citeTest[11] + 1
								elif(testPreStart[q] == 13):
									citeTest[12] = citeTest[12] + 1
								elif(testPreStart[q] == 14):
									citeTest[13] = citeTest[13] + 1
								elif(testPreStart[q] == 15):
									citeTest[14] = citeTest[14] + 1
								elif(testPreStart[q] == 16):
									citeTest[15] = citeTest[15] + 1
								elif(testPreStart[q] == 17):
									citeTest[16] = citeTest[16] + 1
								elif(testPreStart[q] == 18):
									citeTest[17] = citeTest[17] + 1
								elif(testPreStart[q] == 19):
									citeTest[18] = citeTest[18] + 1
								break
#					for u in range(6):
#						totalTest = totalTest + (citeTest[u]+citingTest[u])
#						totalCitedTest = totalCitedTest + citeTest[u]
#						totalCitingTest = totalCitingTest + citingTest[u]]
					for u in range(19):
#						if(index_test == 0):
#							countTestAttUnd[i].append(float(citeTest[u]+citingTest[u]))	
#						else:
#							countTestAttUnd[i][3703+u] = float(citeTest[u]+citingTest[u])
#						countTestAttDir[i].append(citeTest[u])
#						if(index_test == 0):
#							if(totalTest == 0):
#								proporTestAttUnd[i].append(0)
#							else:
#								proporTestAttUnd[i].append(round((citeTest[u]+citingTest[u])/totalTest,3))
#						else:
#							if(totalTest == 0):
#								proporTestAttUnd[i][3703+u] = 0
#							else:
#								proporTestAttUnd[i][3703+u] = round((citeTest[u]+citingTest[u])/totalTest,3)
#						if(index_test == 0):
#							if(totalCitedTest == 0):
#								proporTestAttDir[i].append(0)
#							else:
#								proporTestAttDir[i].append(round(citeTest[u]/totalCitedTest,3))
#						else:
#							if(totalCitedTest == 0):
#								proporTestAttDir[i][3703+u] = 0
#							else:
#								proporTestAttDir[i][3703+u] = round(citeTest[u]/totalCitedTest,3)
#						if(test_index == 0):
#							if(citeTest[u]+citingTest[u] == 0):
#								existTestAttUnd[i].append(0)
#							else:
#								existTestAttUnd[i].append(1)
#						else:
#							if(citeTest[u]+citingTest[u] == 0):
#								existTestAttUnd[i][3703+u] = 0
#							else:
#								existTestAttUnd[i][3703+u] = 1
#						if(test_index == 0):
#							if(citeTest[u] == 0):
#								existTestAttDir[i].append(0)
#							else:
#								existTestAttDir[i].append(1)
#						else:
#							if(citeTest[u] == 0):
#								existTestAttDir[i][3703+u] = 0
#							else:
#							 	existTestAttDir[i][3703+u] = 1
						if((citeTest[u]+citingTest[u]) > maxTotalTest):
							indexTest = u	
							maxTotalTest = citeTest[u]+citingTest[u]
#						if(citeTest[u] > maxCitedTest):
#							indexCitedTest = u
#							maxCitedTest = citeTest[u]
					if(index_test == 0):
						modeTestAttUnd[i].append(indexTest)
					else:
						modeTestAttUnd[i][4973] = indexTest
#					modeAttDir[i].append(indexCitedTest)
#					for u in range(6):
#						countTestAttDir[i].append(citingTest[u])
#						if(index_test == 0):
#							if(totalCitingTest == 0):
#								proporTestAttDir[i].append(0)
#							else:
#								proporTestAttDir[i].append(round(citingTest[u]/totalCitingTest,3))
#						else:
#							if(totalCitedTest == 0):
#								proporTestAttDir[i][3709+u] = 0
#							else:
#								proporTestAttDir[i][3703+u] = round(citingTest[u]/totalCitingTest,3)
#						if(test_index == 0):
#							if(citingTest[u] == 0):
#								existTestAttDir[i].append(0)
#							else:
#								existTestAttDir[i].append(1)
#						else:
#							if(citingTest[u] == 0):
#								existTestAttDir[i][3709+u] = 0
#							else:
#							 	existTestAttDir[i][3709+u] = 1
#						if(citingTest[u] > maxCitingTest):
#							indexCitingTest = u
#							maxCitingTest = citingTest[u]
#					modeTestAttDir[i].append(indexCitingTest)
			index_test = index_test + 1
			m_test = np.array(modeTestAttUnd,dtype='f')	
			testPreNew = clf_modeUnd.predict(m_test)		
			if(np.allclose(testPreNew,testPreStart)):
				break
			else:
				testPreStart = testPreNew
		scoreModeUnd = scoreModeUnd + clf_modeUnd.score(m_test,n_test)
		print(scoreModeUnd)




		index_test = 0
		testPreStart = testPre
		testPreNew =  testPre
		while( 0 <= index_test <10):
			for i in range(len(test_indices)):
					citeTest = []
					citingTest = []
					for m in range(19):
						citeTest.append(0)
						citingTest.append(0)
#					totalTest = totalCitedTest = totalCitingTest = 0.0
					maxTotalTest = maxCitedTest = maxCitingTest = 0
					indexTest = indexCitedTest = indexCitingTest = 0
					for m in range(len(dataList[test_indices[i]][3][0])):
						for n in range(len(train_indices)):
							if(dataList[test_indices[i]][3][0][m] == dataList[train_indices[n]][0]):
								if(dataList[train_indices[n]][1] == 1):
									citingTest[0] = citingTest[0] + 1
								elif(dataList[train_indices[n]][1] == 2):
									citingTest[1] = citingTest[1] + 1
								elif(dataList[train_indices[n]][1] == 3):
									citingTest[2] = citingTest[2] + 1
								elif(dataList[train_indices[n]][1] == 4):
									citingTest[3] = citingTest[3] + 1
								elif(dataList[train_indices[n]][1] == 5):
									citingTest[4] = citingTest[4] + 1
								elif(dataList[train_indices[n]][1] == 6):
									citingTest[5] = citingTest[5] + 1
								elif(dataList[train_indices[n]][1] == 7):
									citingTest[6] = citingTest[6] + 1
								elif(dataList[train_indices[n]][1] == 8):
									citingTest[7] = citingTest[7] + 1
								elif(dataList[train_indices[n]][1] == 9):
									citingTest[8] = citingTest[8] + 1
								elif(dataList[train_indices[n]][1] == 10):
									citingTest[9] = citingTest[9] + 1
								elif(dataList[train_indices[n]][1] == 11):
									citingTest[10] = citingTest[10] + 1
								elif(dataList[train_indices[n]][1] == 12):
									citingTest[11] = citingTest[11] + 1
								elif(dataList[train_indices[n]][1] == 13):
									citingTest[12] = citingTest[12] + 1
								elif(dataList[train_indices[n]][1] == 14):
									citingTest[13] = citingTest[13] + 1
								elif(dataList[train_indices[n]][1] == 15):
									citingTest[14] = citingTest[14] + 1
								elif(dataList[train_indices[n]][1] == 16):
									citingTest[15] = citingTest[15] + 1
								elif(dataList[train_indices[n]][1] == 17):
									citingTest[16] = citingTest[16] + 1
								elif(dataList[train_indices[n]][1] == 18):
									citingTest[17] = citingTest[17] + 1
								elif(dataList[train_indices[n]][1] == 19):
									citingTest[18] = citingTest[18] + 1
								break
						for n in range(len(test_indices)):
							if(dataList[test_indices[i]][3][0][m] == dataList[test_indices[n]][0]):
								if(testPreStart[n] == 1):
									citingTest[0] = citingTest[0] + 1
								elif(testPreStart[n] == 2):
									citingTest[1] = citingTest[1] + 1
								elif(testPreStart[n] == 3):
									citingTest[2] = citingTest[2] + 1
								elif(testPreStart[n] == 4):
									citingTest[3] = citingTest[3] + 1
								elif(testPreStart[n] == 5):
									citingTest[4] = citingTest[4] + 1
								elif(testPreStart[n] == 6):
									citingTest[5] = citingTest[5] + 1
								elif(testPreStart[n] == 7):
									citingTest[6] = citingTest[6] + 1
								elif(testPreStart[n] == 8):
									citingTest[7] = citingTest[7] + 1
								elif(testPreStart[n] == 9):
									citingTest[8] = citingTest[8] + 1
								elif(testPreStart[n] == 10):
									citingTest[9] = citingTest[9] + 1
								elif(testPreStart[n] == 11):
									citingTest[10] = citingTest[10] + 1
								elif(testPreStart[n] == 12):
									citingTest[11] = citingTest[11] + 1
								elif(testPreStart[n] == 13):
									citingTest[12] = citingTest[12] + 1
								elif(testPreStart[n] == 14):
									citingTest[13] = citingTest[13] + 1
								elif(testPreStart[n] == 15):
									citingTest[14] = citingTest[14] + 1
								elif(testPreStart[n] == 16):
									citingTest[15] = citingTest[15] + 1
								elif(testPreStart[n] == 17):
									citingTest[16] = citingTest[16] + 1
								elif(testPreStart[n] == 18):
									citingTest[17] = citingTest[17] + 1
								elif(testPreStart[n] == 19):
									citingTest[18] = citingTest[18] + 1
								break
					for p in range(len(dataList[test_indices[i]][3][1])):
						for q in range(len(train_indices)):
							if(dataList[test_indices[i]][3][1][p] == dataList[train_indices[q]][0]):
								if(dataList[train_indices[q]][1] == 1):
									citeTest[0] = citeTest[0] +1
								elif(dataList[train_indices[q]][1] == 2):
									citeTest[1] = citeTest[1] +1
								elif(dataList[train_indices[q]][1] == 3):
									citeTest[2] = citeTest[2] +1
								elif(dataList[train_indices[q]][1] == 4):
									citeTest[3] = citeTest[3] +1
								elif(dataList[train_indices[q]][1] == 5):
									citeTest[4] = citeTest[4] +1
								elif(dataList[train_indices[q]][1] == 6):
									citeTest[5] = citeTest[5] +1
								elif(dataList[train_indices[q]][1] == 7):
									citeTest[6] = citeTest[6] +1
								elif(dataList[train_indices[q]][1] == 8):
									citeTest[7] = citeTest[7] +1
								elif(dataList[train_indices[q]][1] == 9):
									citeTest[8] = citeTest[8] +1
								elif(dataList[train_indices[q]][1] == 10):
									citeTest[9] = citeTest[9] +1
								elif(dataList[train_indices[q]][1] == 11):
									citeTest[10] = citeTest[10] +1
								elif(dataList[train_indices[q]][1] == 12):
									citeTest[11] = citeTest[11] +1
								elif(dataList[train_indices[q]][1] == 13):
									citeTest[12] = citeTest[12] +1
								elif(dataList[train_indices[q]][1] == 14):
									citeTest[13] = citeTest[13] +1
								elif(dataList[train_indices[q]][1] == 15):
									citeTest[14] = citeTest[14] +1
								elif(dataList[train_indices[q]][1] == 16):
									citeTest[15] = citeTest[15] +1
								elif(dataList[train_indices[q]][1] == 17):
									citeTest[16] = citeTest[16] +1
								elif(dataList[train_indices[q]][1] == 18):
									citeTest[17] = citeTest[17] +1
								elif(dataList[train_indices[q]][1] == 19):
									citeTest[18] = citeTest[18] +1
								break
						for q in range(len(test_indices)):
							if(dataList[test_indices[i]][3][1][p] == dataList[test_indices[q]][0]):
								if(testPreStart[q] == 1):
									citeTest[0] = citeTest[0] + 1
								elif(testPreStart[q] == 2):
									citeTest[1] = citeTest[1] + 1
								elif(testPreStart[q] == 3):
									citeTest[2] = citeTest[2] + 1
								elif(testPreStart[q] == 4):
									citeTest[3] = citeTest[3] + 1
								elif(testPreStart[q] == 5):
									citeTest[4] = citeTest[4] + 1
								elif(testPreStart[q] == 6):
									citeTest[5] = citeTest[5] + 1
								elif(testPreStart[q] == 7):
									citeTest[6] = citeTest[6] + 1
								elif(testPreStart[q] == 8):
									citeTest[7] = citeTest[7] + 1
								elif(testPreStart[q] == 9):
									citeTest[8] = citeTest[8] + 1
								elif(testPreStart[q] == 10):
									citeTest[9] = citeTest[9] + 1
								elif(testPreStart[q] == 11):
									citeTest[10] = citeTest[10] + 1
								elif(testPreStart[q] == 12):
									citeTest[11] = citeTest[11] + 1
								elif(testPreStart[q] == 13):
									citeTest[12] = citeTest[12] + 1
								elif(testPreStart[q] == 14):
									citeTest[13] = citeTest[13] + 1
								elif(testPreStart[q] == 15):
									citeTest[14] = citeTest[14] + 1
								elif(testPreStart[q] == 16):
									citeTest[15] = citeTest[15] + 1
								elif(testPreStart[q] == 17):
									citeTest[16] = citeTest[16] + 1
								elif(testPreStart[q] == 18):
									citeTest[17] = citeTest[17] + 1
								elif(testPreStart[q] == 19):
									citeTest[18] = citeTest[18] + 1
								break
#					for u in range(6):
#						totalTest = totalTest + (citeTest[u]+citingTest[u])
#						totalCitedTest = totalCitedTest + citeTest[u]
#						totalCitingTest = totalCitingTest + citingTest[u]]
					for u in range(19):
#						if(index_test == 0):
#							countTestAttUnd[i].append(float(citeTest[u]+citingTest[u]))	
#						else:
#							countTestAttUnd[i][3703+u] = float(citeTest[u]+citingTest[u])
#						countTestAttDir[i].append(citeTest[u])
#						if(index_test == 0):
#							if(totalTest == 0):
#								proporTestAttUnd[i].append(0)
#							else:
#								proporTestAttUnd[i].append(round((citeTest[u]+citingTest[u])/totalTest,3))
#						else:
#							if(totalTest == 0):
#								proporTestAttUnd[i][3703+u] = 0
#							else:
#								proporTestAttUnd[i][3703+u] = round((citeTest[u]+citingTest[u])/totalTest,3)
#						if(index_test == 0):
#							if(totalCitedTest == 0):
#								proporTestAttDir[i].append(0)
#							else:
#								proporTestAttDir[i].append(round(citeTest[u]/totalCitedTest,3))
#						else:
#							if(totalCitedTest == 0):
#								proporTestAttDir[i][3703+u] = 0
#							else:
#								proporTestAttDir[i][3703+u] = round(citeTest[u]/totalCitedTest,3)
#						if(test_index == 0):
#							if(citeTest[u]+citingTest[u] == 0):
#								existTestAttUnd[i].append(0)
#							else:
#								existTestAttUnd[i].append(1)
#						else:
#							if(citeTest[u]+citingTest[u] == 0):
#								existTestAttUnd[i][3703+u] = 0
#							else:
#								existTestAttUnd[i][3703+u] = 1
#						if(test_index == 0):
#							if(citeTest[u] == 0):
#								existTestAttDir[i].append(0)
#							else:
#								existTestAttDir[i].append(1)
#						else:
#							if(citeTest[u] == 0):
#								existTestAttDir[i][3703+u] = 0
#							else:
#							 	existTestAttDir[i][3703+u] = 1
#						if((citeTest[u]+citingTest[u]) > maxTotalTest):
#							indexLabel = u	
#							maxTotalTest = citeTest[u]+citingTest[u]
						if(citeTest[u] > maxCitedTest):
							indexCitedTest = u
							maxCitedTest = citeTest[u]
#					if(index_test == 0)
#						modeAttUnd[i].append(indexLabel)
#					else:
#						modeAttUnd[i][3703] = indexLabel
					if(index_test == 0):
						modeTestAttDir[i].append(indexCitedTest)
					else:
						modeTestAttDir[i][4973] = indexCitedTest
					for u in range(19):
#						countTestAttDir[i].append(citingTest[u])
#						if(index_test == 0):
#							if(totalCitingTest == 0):
#								proporTestAttDir[i].append(0)
#							else:
#								proporTestAttDir[i].append(round(citingTest[u]/totalCitingTest,3))
#						else:
#							if(totalCitedTest == 0):
#								proporTestAttDir[i][3709+u] = 0
#							else:
#								proporTestAttDir[i][3703+u] = round(citingTest[u]/totalCitingTest,3)
#						if(test_index == 0):
#							if(citingTest[u] == 0):
#								existTestAttDir[i].append(0)
#							else:
#								existTestAttDir[i].append(1)
#						else:
#							if(citingTest[u] == 0):
#								existTestAttDir[i][3709+u] = 0
#							else:
#							 	existTestAttDir[i][3709+u] = 1
						if(citingTest[u] > maxCitingTest):
							indexCitingTest = u
							maxCitingTest = citingTest[u]
					if(index_test == 0):
						modeTestAttDir[i].append(indexCitingTest)
					else:
						modeTestAttDir[i][4974] = indexCitingTest
			index_test = index_test + 1
			m_test = np.array(modeTestAttDir,dtype='f')	
			testPreNew = clf_modeDir.predict(m_test)		
			if(np.allclose(testPreNew,testPreStart)):
				break
			else:
				testPreStart = testPreNew
		scoreModeDir = scoreModeDir + clf_modeDir.score(m_test,n_test)
		print(scoreModeDir)
	print(scoreContent/5.0)
	print(scoreCountUnd/5.0)
	print(scoreCountDir/5.0)
	print(scoreProUnd/5.0)
	print(scoreProDir/5.0)
	print(scoreExistUnd/5.0)
	print(scoreExistDir/5.0)
	print(scoreModeUnd/5.0)
	print(scoreModeDir/5.0)

			



				
struct()
