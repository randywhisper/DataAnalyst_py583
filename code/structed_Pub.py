from sklearn import cross_validation
f = open("Pubmed-Diabetes/data/Pubmed-Diabetes.DIRECTED.cites.tab")
m = open("Pubmed-Diabetes/data/Pubmed-diabetes.NODE.paper.tab")

#define the dataset
dataList = []

#	dataList[i] = [paper_id + class_label + word_attributed + [citing paper + cited paper]]
#construct the data from the content file:

def buildCon():
	m.readline()
	basic = []
	firstLine = m.readline().split()
	for i in range(len(firstLine)-2):
		basic.append(firstLine[i+1].split(":")[1])	
	for line in m:
		sonList = []
		c = line.split()
		sonList.append(c[0])
		sonList.append(int(c[1].split("=")[1])-1)
		lenUnit = len(c)
		sonList.append([])
		for i in range(500):
			sonList[2].append(0.0)
		for i in range(lenUnit-3):
			test = c[i+2].split("=")[0]
			for j in range(500):
				if(basic[j] == test):
					sonList[2][j] = c[i+2].split("=")[1]
					break
		sonList.append([])
		sonList[3].append([])
		sonList[3].append([])
		dataList.append(sonList)	

##transfer the label to the number
#		if(c[n-1] == "Agents"):
#			q = 0
#		elif(c[n-1] == "AI"):
#			q = 1
#		elif(c[n-1] == "DB"):
#			q = 2
#		elif(c[n-1] == "IR"):
#			q = 3
#		elif(c[n-1] == "ML"):
#			q = 4
#		elif(c[n-1] == "HCI"):
#			q = 5
#		sonList.append(q)
#		sonList.append([])
#		for i in range(n-2):
#			sonList[2].append(float(c[i+1]))
#		dataList.append(sonList)
#
#construct the data from the cite file
def buildCite():
	f.readline()
	f.readline()
	for line in f:
		p = line.split()[1].split(":")[1]
		q = line.split()[3].split(":")[1]
		for i in range(len(dataList)):
			if(dataList[i][0] == p and p != q):
					dataList[i][3][0].append(q)
			if(dataList[i][0] == q and p != q):
					dataList[i][3][1].append(p)

buildCon()
buildCite()
print(1)
