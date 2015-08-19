#------->Project-CS 583<----------
#------Author: Randy Wang---------
#---Mail:randywhisper@gmail.com---

#execute the structed python file and get the dataList
execfile("structed_cite.py")
import copy

#exectue 5-fold cross-validation
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()

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
		X_test = np.array(testAtt,dtype ='f')
		y_test = np.array(testLab,dtype ='f')
		clf.fit(X,y)
		testPre = clf.predict(X_test)		
		scoreContent = scoreContent + clf.score(X_test,y_test)
		print(scoreContent)

#creating relational features of the training instances	


		
		countAttUnd = []
		countAttDir = []
		proporAttUnd = []
		proporAttDir = []
		existAttUnd = []
		existAttDir = []
		modeAttUnd = []
		modeAttDir = []
		for m in range(len(train_indices)):
			countAttUnd.append([])
			countAttDir.append([])
			proporAttUnd.append([])
			proporAttDir.append([])
			existAttUnd.append([])
			existAttDir.append([])
			modeAttUnd.append([])
			modeAttDir.append([])
		countTestAttUnd = []
		countTestAttDir = []
		proporTestAttUnd = []
		proporTestAttDir = []
		existTestAttUnd = []
		existTestAttDir = []
		modeTestAttUnd = []
		modeTestAttDir = []
		for m in range(len(test_indices)):
			countTestAttUnd.append([])
			countTestAttDir.append([])
			proporTestAttUnd.append([])
			proporTestAttDir.append([])
			existTestAttUnd.append([])
			existTestAttDir.append([])
			modeTestAttUnd.append([])
			modeTestAttDir.append([])
#execute ICA
		for i in range(len(train_indices)):
			citeTrain = [0,0,0,0,0,0]
			citingTrain = [0,0,0,0,0,0]
			total = totalCited = totalCiting = 0.0
			maxTotal = maxCited = maxCiting = 0
			index = indexCited = indexCiting = 0
			for m in range(len(dataList[train_indices[i]][3][0])):
				for n in range(len(train_indices)):
					if(dataList[train_indices[i]][3][0][m] == dataList[train_indices[n]][0]):
						if(dataList[train_indices[n]][1] == 0):
							citingTrain[0] = citingTrain[0] + 1
						elif(dataList[train_indices[n]][1] == 1):
							citingTrain[1] = citingTrain[1] + 1
						elif(dataList[train_indices[n]][1] == 2):
							citingTrain[2] = citingTrain[2] + 1
						elif(dataList[train_indices[n]][1] == 3):
							citingTrain[3] = citingTrain[3] + 1
						elif(dataList[train_indices[n]][1] == 4):
							citingTrain[4] = citingTrain[4] + 1
						elif(dataList[train_indices[n]][1] == 5):
							citingTrain[5] = citingTrain[5] + 1
						break
			for p in range(len(dataList[train_indices[i]][3][1])):
				for q in range(len(train_indices)):
					if(dataList[train_indices[i]][3][1][p] == dataList[train_indices[q]][0]):
						if(dataList[train_indices[q]][1] == 0):
							citeTrain[0] = citeTrain[0] +1
						elif(dataList[train_indices[q]][1] == 1):
							citeTrain[1] = citeTrain[1] +1
						elif(dataList[train_indices[q]][1] == 2):
							citeTrain[2] = citeTrain[2] +1
						elif(dataList[train_indices[q]][1] == 3):
							citeTrain[3] = citeTrain[3] +1
						elif(dataList[train_indices[q]][1] == 4):
							citeTrain[4] = citeTrain[4] +1
						elif(dataList[train_indices[q]][1] == 5):
							citeTrain[5] = citeTrain[5] +1
						break
			for u in range(6):
				total = total + (citeTrain[u]+citingTrain[u])
				totalCited = totalCited + citeTrain[u]
				totalCiting = totalCiting + citingTrain[u]
			for u in range(6):
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
			for u in range(6):
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

		X_countUnd = np.hstack((X,np.array(countAttUnd,dtype='f')))
		clf.fit(X_countUnd,y)
        
#creating the relational features of testing instances
		index_test = 0
		testPreStart = testPre
		testPreNew =  testPre
		while( 0 <= index_test <10):
			for i in range(len(test_indices)):
					citeTest = [0,0,0,0,0,0]
					citingTest = [0,0,0,0,0,0]
					for m in range(len(dataList[test_indices[i]][3][0])):
						for n in range(len(train_indices)):
							if(dataList[test_indices[i]][3][0][m] == dataList[train_indices[n]][0]):
								if(dataList[train_indices[n]][1] == 0):
									citingTest[0] = citingTest[0] + 1
								elif(dataList[train_indices[n]][1] == 1):
									citingTest[1] = citingTest[1] + 1
								elif(dataList[train_indices[n]][1] == 2):
									citingTest[2] = citingTest[2] + 1
								elif(dataList[train_indices[n]][1] == 3):
									citingTest[3] = citingTest[3] + 1
								elif(dataList[train_indices[n]][1] == 4):
									citingTest[4] = citingTest[4] + 1
								elif(dataList[train_indices[n]][1] == 5):
									citingTest[5] = citingTest[5] + 1
								break
						for n in range(len(test_indices)):
							if(dataList[test_indices[i]][3][0][m] == dataList[test_indices[n]][0]):
								if(testPreStart[n] == 0):
									citingTest[0] = citingTest[0] + 1
								elif(testPreStart[n] == 1):
									citingTest[1] = citingTest[1] + 1
								elif(testPreStart[n] == 2):
									citingTest[2] = citingTest[2] + 1
								elif(testPreStart[n] == 3):
									citingTest[3] = citingTest[3] + 1
								elif(testPreStart[n] == 4):
									citingTest[4] = citingTest[4] + 1
								elif(testPreStart[n] == 5):
									citingTest[5] = citingTest[5] + 1
								break
					for p in range(len(dataList[test_indices[i]][3][1])):
						for q in range(len(train_indices)):
							if(dataList[test_indices[i]][3][1][p] == dataList[train_indices[q]][0]):
								if(dataList[train_indices[q]][1] == 0):
									citeTest[0] = citeTest[0] +1
								elif(dataList[train_indices[q]][1] == 1):
									citeTest[1] = citeTest[1] +1
								elif(dataList[train_indices[q]][1] == 2):
									citeTest[2] = citeTest[2] +1
								elif(dataList[train_indices[q]][1] == 3):
									citeTest[3] = citeTest[3] +1
								elif(dataList[train_indices[q]][1] == 4):
									citeTest[4] = citeTest[4] +1
								elif(dataList[train_indices[q]][1] == 5):
									citeTest[5] = citeTest[5] +1
								break
						for q in range(len(test_indices)):
							if(dataList[test_indices[i]][3][1][p] == dataList[test_indices[q]][0]):
								if(testPreStart[q] == 0):
									citeTest[0] = citeTest[0] + 1
								elif(testPreStart[q] == 1):
									citeTest[1] = citeTest[1] + 1
								elif(testPreStart[q] == 2):
									citeTest[2] = citeTest[2] + 1
								elif(testPreStart[q] == 3):
									citeTest[3] = citeTest[3] + 1
								elif(testPreStart[q] == 4):
									citeTest[4] = citeTest[4] + 1
								elif(testPreStart[q] == 5):
									citeTest[5] = citeTest[5] + 1
								break
					for u in range(6):
						if(index_test == 0):
							countTestAttUnd[i].append(float(citeTest[u]+citingTest[u]))	
						else:
							countTestAttUnd[i][u] = float(citeTest[u]+citingTest[u])

			index_test = index_test + 1
			m_test = np.hstack((X_test,np.array(countTestAttUnd,dtype='f')))
			testPreNew = clf.predict(m_test)		
			if(np.allclose(testPreNew,testPreStart)):
				break
			else:
				testPreStart = testPreNew
		scoreCountUnd = scoreCountUnd + clf.score(m_test,y_test)
		print(scoreCountUnd)


		X_countDir = np.hstack((X,np.array(countAttDir,dtype='f')))
		clf.fit(X_countDir,y)

		index_test = 0
		testPreStart = testPre
		testPreNew =  testPre
		while( 0 <= index_test <10):
			for i in range(len(test_indices)):
					citeTest = [0,0,0,0,0,0]
					citingTest = [0,0,0,0,0,0]
					for m in range(len(dataList[test_indices[i]][3][0])):
						for n in range(len(train_indices)):
							if(dataList[test_indices[i]][3][0][m] == dataList[train_indices[n]][0]):
								if(dataList[train_indices[n]][1] == 0):
									citingTest[0] = citingTest[0] + 1
								elif(dataList[train_indices[n]][1] == 1):
									citingTest[1] = citingTest[1] + 1
								elif(dataList[train_indices[n]][1] == 2):
									citingTest[2] = citingTest[2] + 1
								elif(dataList[train_indices[n]][1] == 3):
									citingTest[3] = citingTest[3] + 1
								elif(dataList[train_indices[n]][1] == 4):
									citingTest[4] = citingTest[4] + 1
								elif(dataList[train_indices[n]][1] == 5):
									citingTest[5] = citingTest[5] + 1
								break
						for n in range(len(test_indices)):
							if(dataList[test_indices[i]][3][0][m] == dataList[test_indices[n]][0]):
								if(testPreStart[n] == 0):
									citingTest[0] = citingTest[0] + 1
								elif(testPreStart[n] == 1):
									citingTest[1] = citingTest[1] + 1
								elif(testPreStart[n] == 2):
									citingTest[2] = citingTest[2] + 1
								elif(testPreStart[n] == 3):
									citingTest[3] = citingTest[3] + 1
								elif(testPreStart[n] == 4):
									citingTest[4] = citingTest[4] + 1
								elif(testPreStart[n] == 5):
									citingTest[5] = citingTest[5] + 1
								break
					for p in range(len(dataList[test_indices[i]][3][1])):
						for q in range(len(train_indices)):
							if(dataList[test_indices[i]][3][1][p] == dataList[train_indices[q]][0]):
								if(dataList[train_indices[q]][1] == 0):
									citeTest[0] = citeTest[0] +1
								elif(dataList[train_indices[q]][1] == 1):
									citeTest[1] = citeTest[1] +1
								elif(dataList[train_indices[q]][1] == 2):
									citeTest[2] = citeTest[2] +1
								elif(dataList[train_indices[q]][1] == 3):
									citeTest[3] = citeTest[3] +1
								elif(dataList[train_indices[q]][1] == 4):
									citeTest[4] = citeTest[4] +1
								elif(dataList[train_indices[q]][1] == 5):
									citeTest[5] = citeTest[5] +1
								break
						for q in range(len(test_indices)):
							if(dataList[test_indices[i]][3][1][p] == dataList[test_indices[q]][0]):
								if(testPreStart[q] == 0):
									citeTest[0] = citeTest[0] + 1
								elif(testPreStart[q] == 1):
									citeTest[1] = citeTest[1] + 1
								elif(testPreStart[q] == 2):
									citeTest[2] = citeTest[2] + 1
								elif(testPreStart[q] == 3):
									citeTest[3] = citeTest[3] + 1
								elif(testPreStart[q] == 4):
									citeTest[4] = citeTest[4] + 1
								elif(testPreStart[q] == 5):
									citeTest[5] = citeTest[5] + 1
								break
					for u in range(6):
						if(index_test == 0):
							countTestAttDir[i].append(float(citeTest[u]))	
						else:
							countTestAttDir[i][u] = float(citeTest[u])
					for u in range(6):
						if(index_test == 0):
							countTestAttDir[i].append(citingTest[u])
						else:
							countTestAttDir[i][6+u] = float(citingTest[u])

			index_test = index_test + 1
			m_test = np.hstack((X_test,np.array(countTestAttDir,dtype='f')))
			testPreNew = clf.predict(m_test)		
			if(np.allclose(testPreNew,testPreStart)):
				break
			else:
				testPreStart = testPreNew
		scoreCountDir = scoreCountDir + clf.score(m_test,y_test)
		print(scoreCountDir)

		X_proporUnd = np.hstack((X,np.array(proporAttUnd,dtype='f')))
		clf.fit(X_proporUnd,y)

		index_test = 0
		testPreStart = testPre
		testPreNew =  testPre
		while( 0 <= index_test <10):
			for i in range(len(test_indices)):
					citeTest = [0,0,0,0,0,0]
					citingTest = [0,0,0,0,0,0]
					totalTest = totalCitedTest = totalCitingTest = 0.0
					maxTotalTest = maxCitedTest = maxCitingTest = 0
					indexTest = indexCitedTest = indexCitingTest = 0
					for m in range(len(dataList[test_indices[i]][3][0])):
						for n in range(len(train_indices)):
							if(dataList[test_indices[i]][3][0][m] == dataList[train_indices[n]][0]):
								if(dataList[train_indices[n]][1] == 0):
									citingTest[0] = citingTest[0] + 1
								elif(dataList[train_indices[n]][1] == 1):
									citingTest[1] = citingTest[1] + 1
								elif(dataList[train_indices[n]][1] == 2):
									citingTest[2] = citingTest[2] + 1
								elif(dataList[train_indices[n]][1] == 3):
									citingTest[3] = citingTest[3] + 1
								elif(dataList[train_indices[n]][1] == 4):
									citingTest[4] = citingTest[4] + 1
								elif(dataList[train_indices[n]][1] == 5):
									citingTest[5] = citingTest[5] + 1
								break
						for n in range(len(test_indices)):
							if(dataList[test_indices[i]][3][0][m] == dataList[test_indices[n]][0]):
								if(testPreStart[n] == 0):
									citingTest[0] = citingTest[0] + 1
								elif(testPreStart[n] == 1):
									citingTest[1] = citingTest[1] + 1
								elif(testPreStart[n] == 2):
									citingTest[2] = citingTest[2] + 1
								elif(testPreStart[n] == 3):
									citingTest[3] = citingTest[3] + 1
								elif(testPreStart[n] == 4):
									citingTest[4] = citingTest[4] + 1
								elif(testPreStart[n] == 5):
									citingTest[5] = citingTest[5] + 1
								break
					for p in range(len(dataList[test_indices[i]][3][1])):
						for q in range(len(train_indices)):
							if(dataList[test_indices[i]][3][1][p] == dataList[train_indices[q]][0]):
								if(dataList[train_indices[q]][1] == 0):
									citeTest[0] = citeTest[0] +1
								elif(dataList[train_indices[q]][1] == 1):
									citeTest[1] = citeTest[1] +1
								elif(dataList[train_indices[q]][1] == 2):
									citeTest[2] = citeTest[2] +1
								elif(dataList[train_indices[q]][1] == 3):
									citeTest[3] = citeTest[3] +1
								elif(dataList[train_indices[q]][1] == 4):
									citeTest[4] = citeTest[4] +1
								elif(dataList[train_indices[q]][1] == 5):
									citeTest[5] = citeTest[5] +1
								break
						for q in range(len(test_indices)):
							if(dataList[test_indices[i]][3][1][p] == dataList[test_indices[q]][0]):
								if(testPreStart[q] == 0):
									citeTest[0] = citeTest[0] + 1
								elif(testPreStart[q] == 1):
									citeTest[1] = citeTest[1] + 1
								elif(testPreStart[q] == 2):
									citeTest[2] = citeTest[2] + 1
								elif(testPreStart[q] == 3):
									citeTest[3] = citeTest[3] + 1
								elif(testPreStart[q] == 4):
									citeTest[4] = citeTest[4] + 1
								elif(testPreStart[q] == 5):
									citeTest[5] = citeTest[5] + 1
								break
					for u in range(6):
						totalTest = totalTest + (citeTest[u]+citingTest[u])
					for u in range(6):
						if(index_test == 0):
							if(totalTest == 0):
								proporTestAttUnd[i].append(0)
							else:
								proporTestAttUnd[i].append(round((citeTest[u]+citingTest[u])/totalTest,3))
						else:
							if(totalTest == 0):
								proporTestAttUnd[i][u] = 0
							else:
								proporTestAttUnd[i][u] = round((citeTest[u]+citingTest[u])/totalTest,3)

			index_test = index_test + 1
			m_test = np.hstack((X_test,np.array(proporTestAttUnd,dtype='f')))
			testPreNew = clf.predict(m_test)		
			if(np.allclose(testPreNew,testPreStart)):
				break
			else:
				testPreStart = testPreNew
		scoreProUnd = scoreProUnd + clf.score(m_test,y_test)
		print(scoreProUnd)


		X_proporDir = np.hstack((X,np.array(proporAttDir,dtype='f')))
		clf.fit(X_proporDir,y)

		index_test = 0
		testPreStart = testPre
		testPreNew =  testPre
		while( 0 <= index_test <10):
			for i in range(len(test_indices)):
					citeTest = [0,0,0,0,0,0]
					citingTest = [0,0,0,0,0,0]
					totalTest = totalCitedTest = totalCitingTest = 0.0
					maxTotalTest = maxCitedTest = maxCitingTest = 0
					indexTest = indexCitedTest = indexCitingTest = 0
					for m in range(len(dataList[test_indices[i]][3][0])):
						for n in range(len(train_indices)):
							if(dataList[test_indices[i]][3][0][m] == dataList[train_indices[n]][0]):
								if(dataList[train_indices[n]][1] == 0):
									citingTest[0] = citingTest[0] + 1
								elif(dataList[train_indices[n]][1] == 1):
									citingTest[1] = citingTest[1] + 1
								elif(dataList[train_indices[n]][1] == 2):
									citingTest[2] = citingTest[2] + 1
								elif(dataList[train_indices[n]][1] == 3):
									citingTest[3] = citingTest[3] + 1
								elif(dataList[train_indices[n]][1] == 4):
									citingTest[4] = citingTest[4] + 1
								elif(dataList[train_indices[n]][1] == 5):
									citingTest[5] = citingTest[5] + 1
								break
						for n in range(len(test_indices)):
							if(dataList[test_indices[i]][3][0][m] == dataList[test_indices[n]][0]):
								if(testPreStart[n] == 0):
									citingTest[0] = citingTest[0] + 1
								elif(testPreStart[n] == 1):
									citingTest[1] = citingTest[1] + 1
								elif(testPreStart[n] == 2):
									citingTest[2] = citingTest[2] + 1
								elif(testPreStart[n] == 3):
									citingTest[3] = citingTest[3] + 1
								elif(testPreStart[n] == 4):
									citingTest[4] = citingTest[4] + 1
								elif(testPreStart[n] == 5):
									citingTest[5] = citingTest[5] + 1
								break
					for p in range(len(dataList[test_indices[i]][3][1])):
						for q in range(len(train_indices)):
							if(dataList[test_indices[i]][3][1][p] == dataList[train_indices[q]][0]):
								if(dataList[train_indices[q]][1] == 0):
									citeTest[0] = citeTest[0] +1
								elif(dataList[train_indices[q]][1] == 1):
									citeTest[1] = citeTest[1] +1
								elif(dataList[train_indices[q]][1] == 2):
									citeTest[2] = citeTest[2] +1
								elif(dataList[train_indices[q]][1] == 3):
									citeTest[3] = citeTest[3] +1
								elif(dataList[train_indices[q]][1] == 4):
									citeTest[4] = citeTest[4] +1
								elif(dataList[train_indices[q]][1] == 5):
									citeTest[5] = citeTest[5] +1
								break
						for q in range(len(test_indices)):
							if(dataList[test_indices[i]][3][1][p] == dataList[test_indices[q]][0]):
								if(testPreStart[q] == 0):
									citeTest[0] = citeTest[0] + 1
								elif(testPreStart[q] == 1):
									citeTest[1] = citeTest[1] + 1
								elif(testPreStart[q] == 2):
									citeTest[2] = citeTest[2] + 1
								elif(testPreStart[q] == 3):
									citeTest[3] = citeTest[3] + 1
								elif(testPreStart[q] == 4):
									citeTest[4] = citeTest[4] + 1
								elif(testPreStart[q] == 5):
									citeTest[5] = citeTest[5] + 1
								break
					for u in range(6):
						totalCitedTest = totalCitedTest + citeTest[u]
						totalCitingTest = totalCitingTest + citingTest[u]
					for u in range(6):
						if(index_test == 0):
							if(totalCitedTest == 0):
								proporTestAttDir[i].append(0)
							else:
								proporTestAttDir[i].append(round(citeTest[u]/totalCitedTest,3))
						else:
							if(totalCitedTest == 0):
								proporTestAttDir[i][u] = 0
							else:
								proporTestAttDir[i][u] = round(citeTest[u]/totalCitedTest,3)

					for u in range(6):
						if(index_test == 0):
							if(totalCitingTest == 0):
								proporTestAttDir[i].append(0)
							else:
								proporTestAttDir[i].append(round(citingTest[u]/totalCitingTest,3))
						else:
							if(totalCitingTest == 0):
								proporTestAttDir[i][6+u] = 0
							else:
								proporTestAttDir[i][6+u] = round(citingTest[u]/totalCitingTest,3)

			index_test = index_test + 1
			m_test = np.hstack((X_test,np.array(proporTestAttDir,dtype='f')))
			testPreNew = clf.predict(m_test)		
			if(np.allclose(testPreNew,testPreStart)):
				break
			else:
				testPreStart = testPreNew
		scoreProDir = scoreProDir + clf.score(m_test,y_test)
		print(scoreProDir)


		X_existUnd = np.hstack((X,np.array(existAttUnd,dtype='f')))
		clf.fit(X_existUnd,y)


		index_test = 0
		testPreStart = testPre
		testPreNew =  testPre
		while( 0 <= index_test <10):
			for i in range(len(test_indices)):
					citeTest = [0,0,0,0,0,0]
					citingTest = [0,0,0,0,0,0]
					for m in range(len(dataList[test_indices[i]][3][0])):
						for n in range(len(train_indices)):
							if(dataList[test_indices[i]][3][0][m] == dataList[train_indices[n]][0]):
								if(dataList[train_indices[n]][1] == 0):
									citingTest[0] = citingTest[0] + 1
								elif(dataList[train_indices[n]][1] == 1):
									citingTest[1] = citingTest[1] + 1
								elif(dataList[train_indices[n]][1] == 2):
									citingTest[2] = citingTest[2] + 1
								elif(dataList[train_indices[n]][1] == 3):
									citingTest[3] = citingTest[3] + 1
								elif(dataList[train_indices[n]][1] == 4):
									citingTest[4] = citingTest[4] + 1
								elif(dataList[train_indices[n]][1] == 5):
									citingTest[5] = citingTest[5] + 1
								break
						for n in range(len(test_indices)):
							if(dataList[test_indices[i]][3][0][m] == dataList[test_indices[n]][0]):
								if(testPreStart[n] == 0):
									citingTest[0] = citingTest[0] + 1
								elif(testPreStart[n] == 1):
									citingTest[1] = citingTest[1] + 1
								elif(testPreStart[n] == 2):
									citingTest[2] = citingTest[2] + 1
								elif(testPreStart[n] == 3):
									citingTest[3] = citingTest[3] + 1
								elif(testPreStart[n] == 4):
									citingTest[4] = citingTest[4] + 1
								elif(testPreStart[n] == 5):
									citingTest[5] = citingTest[5] + 1
								break
					for p in range(len(dataList[test_indices[i]][3][1])):
						for q in range(len(train_indices)):
							if(dataList[test_indices[i]][3][1][p] == dataList[train_indices[q]][0]):
								if(dataList[train_indices[q]][1] == 0):
									citeTest[0] = citeTest[0] +1
								elif(dataList[train_indices[q]][1] == 1):
									citeTest[1] = citeTest[1] +1
								elif(dataList[train_indices[q]][1] == 2):
									citeTest[2] = citeTest[2] +1
								elif(dataList[train_indices[q]][1] == 3):
									citeTest[3] = citeTest[3] +1
								elif(dataList[train_indices[q]][1] == 4):
									citeTest[4] = citeTest[4] +1
								elif(dataList[train_indices[q]][1] == 5):
									citeTest[5] = citeTest[5] +1
								break
						for q in range(len(test_indices)):
							if(dataList[test_indices[i]][3][1][p] == dataList[test_indices[q]][0]):
								if(testPreStart[q] == 0):
									citeTest[0] = citeTest[0] + 1
								elif(testPreStart[q] == 1):
									citeTest[1] = citeTest[1] + 1
								elif(testPreStart[q] == 2):
									citeTest[2] = citeTest[2] + 1
								elif(testPreStart[q] == 3):
									citeTest[3] = citeTest[3] + 1
								elif(testPreStart[q] == 4):
									citeTest[4] = citeTest[4] + 1
								elif(testPreStart[q] == 5):
									citeTest[5] = citeTest[5] + 1
								break
					for u in range(6):
						if(index_test == 0):
							if(citeTest[u]+citingTest[u] == 0):
								existTestAttUnd[i].append(0)
							else:
								existTestAttUnd[i].append(1)
						else:
							if(citeTest[u]+citingTest[u] == 0):
								existTestAttUnd[i][u] = 0
							else:
								existTestAttUnd[i][u] = 1
			index_test = index_test + 1
			m_test = np.hstack((X_test,np.array(existTestAttUnd,dtype='f')))
			testPreNew = clf.predict(m_test)		
			if(np.allclose(testPreNew,testPreStart)):
				break
			else:
				testPreStart = testPreNew
		scoreExistUnd = scoreExistUnd + clf.score(m_test,y_test)
		print(scoreExistUnd)


		X_existDir = np.hstack((X,np.array(existAttDir,dtype='f')))
		clf.fit(X_existDir,y)

		index_test = 0
		testPreStart = testPre
		testPreNew =  testPre
		while( 0 <= index_test <10):
			for i in range(len(test_indices)):
					citeTest = [0,0,0,0,0,0]
					citingTest = [0,0,0,0,0,0]
					for m in range(len(dataList[test_indices[i]][3][0])):
						for n in range(len(train_indices)):
							if(dataList[test_indices[i]][3][0][m] == dataList[train_indices[n]][0]):
								if(dataList[train_indices[n]][1] == 0):
									citingTest[0] = citingTest[0] + 1
								elif(dataList[train_indices[n]][1] == 1):
									citingTest[1] = citingTest[1] + 1
								elif(dataList[train_indices[n]][1] == 2):
									citingTest[2] = citingTest[2] + 1
								elif(dataList[train_indices[n]][1] == 3):
									citingTest[3] = citingTest[3] + 1
								elif(dataList[train_indices[n]][1] == 4):
									citingTest[4] = citingTest[4] + 1
								elif(dataList[train_indices[n]][1] == 5):
									citingTest[5] = citingTest[5] + 1
								break
						for n in range(len(test_indices)):
							if(dataList[test_indices[i]][3][0][m] == dataList[test_indices[n]][0]):
								if(testPreStart[n] == 0):
									citingTest[0] = citingTest[0] + 1
								elif(testPreStart[n] == 1):
									citingTest[1] = citingTest[1] + 1
								elif(testPreStart[n] == 2):
									citingTest[2] = citingTest[2] + 1
								elif(testPreStart[n] == 3):
									citingTest[3] = citingTest[3] + 1
								elif(testPreStart[n] == 4):
									citingTest[4] = citingTest[4] + 1
								elif(testPreStart[n] == 5):
									citingTest[5] = citingTest[5] + 1
								break
					for p in range(len(dataList[test_indices[i]][3][1])):
						for q in range(len(train_indices)):
							if(dataList[test_indices[i]][3][1][p] == dataList[train_indices[q]][0]):
								if(dataList[train_indices[q]][1] == 0):
									citeTest[0] = citeTest[0] +1
								elif(dataList[train_indices[q]][1] == 1):
									citeTest[1] = citeTest[1] +1
								elif(dataList[train_indices[q]][1] == 2):
									citeTest[2] = citeTest[2] +1
								elif(dataList[train_indices[q]][1] == 3):
									citeTest[3] = citeTest[3] +1
								elif(dataList[train_indices[q]][1] == 4):
									citeTest[4] = citeTest[4] +1
								elif(dataList[train_indices[q]][1] == 5):
									citeTest[5] = citeTest[5] +1
								break
						for q in range(len(test_indices)):
							if(dataList[test_indices[i]][3][1][p] == dataList[test_indices[q]][0]):
								if(testPreStart[q] == 0):
									citeTest[0] = citeTest[0] + 1
								elif(testPreStart[q] == 1):
									citeTest[1] = citeTest[1] + 1
								elif(testPreStart[q] == 2):
									citeTest[2] = citeTest[2] + 1
								elif(testPreStart[q] == 3):
									citeTest[3] = citeTest[3] + 1
								elif(testPreStart[q] == 4):
									citeTest[4] = citeTest[4] + 1
								elif(testPreStart[q] == 5):
									citeTest[5] = citeTest[5] + 1
								break
					for u in range(6):
						if(index_test == 0):
							if(citeTest[u] == 0):
								existTestAttDir[i].append(0)
							else:
								existTestAttDir[i].append(1)
						else:
							if(citeTest[u] == 0):
								existTestAttDir[i][u] = 0
							else:
							 	existTestAttDir[i][u] = 1
					for u in range(6):
						if(index_test == 0):
							if(citingTest[u] == 0):
								existTestAttDir[i].append(0)
							else:
								existTestAttDir[i].append(1)
						else:
							if(citingTest[u] == 0):
								existTestAttDir[i][6+u] = 0
							else:
							 	existTestAttDir[i][6+u] = 1

			index_test = index_test + 1
			m_test = np.hstack((X_test,np.array(existTestAttDir,dtype='f')))
			testPreNew = clf.predict(m_test)		
			if(np.allclose(testPreNew,testPreStart)):
				break
			else:
				testPreStart = testPreNew
		scoreExistDir = scoreExistDir + clf.score(m_test,y_test)
		print(scoreExistDir)


		X_modeUnd = np.hstack((X,np.array(modeAttUnd,dtype='f')))
		clf.fit(X_modeUnd,y)

		index_test = 0
		testPreStart = testPre
		testPreNew =  testPre
		while( 0 <= index_test <10):
			for i in range(len(test_indices)):
					citeTest = [0,0,0,0,0,0]
					citingTest = [0,0,0,0,0,0]
					maxTotalTest = maxCitedTest = maxCitingTest = 0
					indexTest = indexCitedTest = indexCitingTest = 0
					for m in range(len(dataList[test_indices[i]][3][0])):
						for n in range(len(train_indices)):
							if(dataList[test_indices[i]][3][0][m] == dataList[train_indices[n]][0]):
								if(dataList[train_indices[n]][1] == 0):
									citingTest[0] = citingTest[0] + 1
								elif(dataList[train_indices[n]][1] == 1):
									citingTest[1] = citingTest[1] + 1
								elif(dataList[train_indices[n]][1] == 2):
									citingTest[2] = citingTest[2] + 1
								elif(dataList[train_indices[n]][1] == 3):
									citingTest[3] = citingTest[3] + 1
								elif(dataList[train_indices[n]][1] == 4):
									citingTest[4] = citingTest[4] + 1
								elif(dataList[train_indices[n]][1] == 5):
									citingTest[5] = citingTest[5] + 1
								break
						for n in range(len(test_indices)):
							if(dataList[test_indices[i]][3][0][m] == dataList[test_indices[n]][0]):
								if(testPreStart[n] == 0):
									citingTest[0] = citingTest[0] + 1
								elif(testPreStart[n] == 1):
									citingTest[1] = citingTest[1] + 1
								elif(testPreStart[n] == 2):
									citingTest[2] = citingTest[2] + 1
								elif(testPreStart[n] == 3):
									citingTest[3] = citingTest[3] + 1
								elif(testPreStart[n] == 4):
									citingTest[4] = citingTest[4] + 1
								elif(testPreStart[n] == 5):
									citingTest[5] = citingTest[5] + 1
								break
					for p in range(len(dataList[test_indices[i]][3][1])):
						for q in range(len(train_indices)):
							if(dataList[test_indices[i]][3][1][p] == dataList[train_indices[q]][0]):
								if(dataList[train_indices[q]][1] == 0):
									citeTest[0] = citeTest[0] +1
								elif(dataList[train_indices[q]][1] == 1):
									citeTest[1] = citeTest[1] +1
								elif(dataList[train_indices[q]][1] == 2):
									citeTest[2] = citeTest[2] +1
								elif(dataList[train_indices[q]][1] == 3):
									citeTest[3] = citeTest[3] +1
								elif(dataList[train_indices[q]][1] == 4):
									citeTest[4] = citeTest[4] +1
								elif(dataList[train_indices[q]][1] == 5):
									citeTest[5] = citeTest[5] +1
								break
						for q in range(len(test_indices)):
							if(dataList[test_indices[i]][3][1][p] == dataList[test_indices[q]][0]):
								if(testPreStart[q] == 0):
									citeTest[0] = citeTest[0] + 1
								elif(testPreStart[q] == 1):
									citeTest[1] = citeTest[1] + 1
								elif(testPreStart[q] == 2):
									citeTest[2] = citeTest[2] + 1
								elif(testPreStart[q] == 3):
									citeTest[3] = citeTest[3] + 1
								elif(testPreStart[q] == 4):
									citeTest[4] = citeTest[4] + 1
								elif(testPreStart[q] == 5):
									citeTest[5] = citeTest[5] + 1
								break
					for u in range(6):
						if((citeTest[u]+citingTest[u]) > maxTotalTest):
							indexTest = u	
							maxTotalTest = citeTest[u]+citingTest[u]
					if(index_test == 0):
						modeTestAttUnd[i].append(indexTest)
					else:
						modeTestAttUnd[i][0] = indexTest
			index_test = index_test + 1
			m_test = np.hstack((X_test,np.array(modeTestAttUnd,dtype='f')))
			testPreNew = clf.predict(m_test)		
			if(np.allclose(testPreNew,testPreStart)):
				break
			else:
				testPreStart = testPreNew
		scoreModeUnd = scoreModeUnd + clf.score(m_test,y_test)
		print(scoreModeUnd)



		X_modeDir = np.hstack((X,np.array(modeAttDir,dtype='f')))
		clf.fit(X_modeDir,y)

		index_test = 0
		testPreStart = testPre
		testPreNew =  testPre
		while( 0 <= index_test <10):
			for i in range(len(test_indices)):
					citeTest = [0,0,0,0,0,0]
					citingTest = [0,0,0,0,0,0]
					maxTotalTest = maxCitedTest = maxCitingTest = 0
					indexTest = indexCitedTest = indexCitingTest = 0
					for m in range(len(dataList[test_indices[i]][3][0])):
						for n in range(len(train_indices)):
							if(dataList[test_indices[i]][3][0][m] == dataList[train_indices[n]][0]):
								if(dataList[train_indices[n]][1] == 0):
									citingTest[0] = citingTest[0] + 1
								elif(dataList[train_indices[n]][1] == 1):
									citingTest[1] = citingTest[1] + 1
								elif(dataList[train_indices[n]][1] == 2):
									citingTest[2] = citingTest[2] + 1
								elif(dataList[train_indices[n]][1] == 3):
									citingTest[3] = citingTest[3] + 1
								elif(dataList[train_indices[n]][1] == 4):
									citingTest[4] = citingTest[4] + 1
								elif(dataList[train_indices[n]][1] == 5):
									citingTest[5] = citingTest[5] + 1
								break
						for n in range(len(test_indices)):
							if(dataList[test_indices[i]][3][0][m] == dataList[test_indices[n]][0]):
								if(testPreStart[n] == 0):
									citingTest[0] = citingTest[0] + 1
								elif(testPreStart[n] == 1):
									citingTest[1] = citingTest[1] + 1
								elif(testPreStart[n] == 2):
									citingTest[2] = citingTest[2] + 1
								elif(testPreStart[n] == 3):
									citingTest[3] = citingTest[3] + 1
								elif(testPreStart[n] == 4):
									citingTest[4] = citingTest[4] + 1
								elif(testPreStart[n] == 5):
									citingTest[5] = citingTest[5] + 1
								break
					for p in range(len(dataList[test_indices[i]][3][1])):
						for q in range(len(train_indices)):
							if(dataList[test_indices[i]][3][1][p] == dataList[train_indices[q]][0]):
								if(dataList[train_indices[q]][1] == 0):
									citeTest[0] = citeTest[0] +1
								elif(dataList[train_indices[q]][1] == 1):
									citeTest[1] = citeTest[1] +1
								elif(dataList[train_indices[q]][1] == 2):
									citeTest[2] = citeTest[2] +1
								elif(dataList[train_indices[q]][1] == 3):
									citeTest[3] = citeTest[3] +1
								elif(dataList[train_indices[q]][1] == 4):
									citeTest[4] = citeTest[4] +1
								elif(dataList[train_indices[q]][1] == 5):
									citeTest[5] = citeTest[5] +1
								break
						for q in range(len(test_indices)):
							if(dataList[test_indices[i]][3][1][p] == dataList[test_indices[q]][0]):
								if(testPreStart[q] == 0):
									citeTest[0] = citeTest[0] + 1
								elif(testPreStart[q] == 1):
									citeTest[1] = citeTest[1] + 1
								elif(testPreStart[q] == 2):
									citeTest[2] = citeTest[2] + 1
								elif(testPreStart[q] == 3):
									citeTest[3] = citeTest[3] + 1
								elif(testPreStart[q] == 4):
									citeTest[4] = citeTest[4] + 1
								elif(testPreStart[q] == 5):
									citeTest[5] = citeTest[5] + 1
								break
					for u in range(6):
						if(citeTest[u] > maxCitedTest):
							indexCitedTest = u
							maxCitedTest = citeTest[u]
					if(index_test == 0):
						modeTestAttDir[i].append(indexCitedTest)
					else:
						modeTestAttDir[i][0] = indexCitedTest
					for u in range(6):
						if(citingTest[u] > maxCitingTest):
							indexCitingTest = u
							maxCitingTest = citingTest[u]
					if(index_test == 0):
						modeTestAttDir[i].append(indexCitingTest)
					else:
						modeTestAttDir[i][1] = indexCitingTest
			index_test = index_test + 1
			m_test = np.hstack((X_test,np.array(modeTestAttDir,dtype='f')))
			testPreNew = clf.predict(m_test)		
			if(np.allclose(testPreNew,testPreStart)):
				break
			else:
				testPreStart = testPreNew
		scoreModeDir = scoreModeDir + clf.score(m_test,y_test)
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
