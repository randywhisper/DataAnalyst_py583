from sklearn import cross_validation
f = open("nips/WithinWithinLinks.txt")
m = open("nips/pruned-documentFilteredBy1.txt")
n = open("nips/newCategoryBelonging.txt")

#define the dataset
dataList = []

#	dataList[i] = [paper_id + class_label + word_attributed + [citing paper + cited paper]]
#construct the data from the content file:

def buildCon():
	for line in m:
		sonList = []
		c = line.split()
		sonList.append(c[0])
		sonList.append(0)
		lenUnit = len(c)
		sonList.append([])
		for i in range(4973):
			sonList[2].append(0.0)
		if(lenUnit > 2):
			for i in range(lenUnit-2):
				index = int(c[i+1].split(":")[0])
				sonList[2][index] = c[i+1].split(":")[1]
		sonList.append([])
		sonList[3].append([])
		sonList[3].append([])
		dataList.append(sonList)	
def buildLabel():
	for line in n:
		for i in range(len(dataList)):
			if(dataList[i][0] == line.split()[0]): 
				dataList[i][1] = int(line.split()[1])-1
				break

#construct the data from the cite file
def buildCite():
	for line in f:
		p = line.split()[0]
		q = line.split()[1]
		for i in range(len(dataList)):
			if(dataList[i][0] == p and p != q):
					dataList[i][3][0].append(q)
			if(dataList[i][0] == q and p != q):
					dataList[i][3][1].append(p)

buildCon()
buildLabel()
buildCite()
