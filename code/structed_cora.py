from sklearn import cross_validation
f = open("cora/cora.cites")
m = open("cora/cora.content")

#define the dataset
dataList = []

#	dataList[i] = [paper_id + class_label + word_attributed + [citing paper + cited paper]]
#construct the data from the content file:

def buildCon():
	for line in m:
		sonList = []
		c = line.split()
		sonList.append(c[0])
		n = len(c)

#transfer the label to the number
		if(c[n-1] == "Case_Based"):
			q = 0
		elif(c[n-1] == "Genetic_Algorithms"):
			q = 1
		elif(c[n-1] == "Neural_Networks"):
			q = 2
		elif(c[n-1] == "Probabilistic_Methods"):
			q = 3
		elif(c[n-1] == "Reinforcement_Learning"):
			q = 4
		elif(c[n-1] == "Rule_Learning"):
			q = 5
		elif(c[n-1] == "Theory"):
			q = 6
		sonList.append(q)
		sonList.append([])
		for i in range(n-2):
			sonList[2].append(float(c[i+1]))
		dataList.append(sonList)

#construct the data from the cite file
def buildCite():
	for i in range(len(dataList)):
		dataList[i].append([])
		dataList[i][3].append([])
		dataList[i][3].append([])
	for line in f:
		p = line.split()[0]
		q = line.split()[1]
		for i in range(len(dataList)):
			if(dataList[i][0] == p and p != q):
					dataList[i][3][0].append(q)
			if(dataList[i][0] == q and p != q):
					dataList[i][3][1].append(p)

buildCon()
buildCite()
