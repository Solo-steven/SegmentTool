import math

class NetEdge():
    def __init__(self, weight):
        self.weight = weight 

class NetVertex():
    def __init__(self,parent=None):
        self.child   = set()
        self.parent  = set()
        if parent is not None :
            self.parent.add(parent)   
  
class WordNet : 
    def __init__(self ,dic , word=None):
        self.__dic = dic 
        self.__word = word 
        self.__vertexTable = {}
        self.__edgeTable   = {}
        self.__DPTable     = {}
        self.__sentence    = []
        if word is not None:
            self.build_Word_Net(word)

    def print_Word_Data(self):
        for key ,value in self.__vertexTable.items():
            print(key, value.parent, value.child);
        for key ,value in self.__edgeTable.items():
            print(key,value.weight)
        for key , value in self.__DPTable.items():
            print(key, value['cost'], value["predecessor"]) 
        return

    def print_Sentence(self):
        print(self.__sentence[1:len(self.__sentence)-1])                

    def build_Word_Net(self, word):
        self.__word = word
        self.__sentence = []
        self.__vertexTable.clear()
        self.__vertexTable.setdefault("BOS", NetVertex())
        self.__vertexTable.setdefault("EOS", NetVertex())
        self.__build_Vertex_Table(0, "BOS")
        self.__edgeTable.clear()
        self.__build_Edge_Table("BOS")
        
        self.__DPTable.clear()
        for key in self.__vertexTable.keys():
            self.__DPTable.setdefault(key,{'cost':math.inf, "predecessor":None})
        visited = set() 
        visited.add("BOS")
        self.__DPTable['BOS']['cost']=0 
        self.__build_DP_Table("BOS", visited)  
        self.__find_Correct_Sentence()
        
    def __build_Vertex_Table(self,startIndex, father, oov=None):
        if startIndex == len(self.__word) :
            if oov is not None :
                self.__vertexTable[father].child.add(oov)
                self.__vertexTable.setdefault(oov, NetVertex(father))
                father = oov 
            self.__vertexTable[father].child.add("EOS")
            return
        mark = False 
        for i in range(startIndex+1, len(self.__word)+1):
            word=self.__word[startIndex:i]
            if word in self.__dic :
                mark = True
                if oov is not None :
                    self.__vertexTable[father].child.add(oov)
                    self.__vertexTable.setdefault(oov, NetVertex(father))
                    father = oov 
                    oov = None
                if self.__vertexTable.get(word) is None :
                    self.__vertexTable.setdefault(word,NetVertex())
                self.__vertexTable[word].parent.add(father)   
                self.__vertexTable[father].child.add(word)
                self.__build_Vertex_Table(i, word)
        if mark == False :
            if oov is not None :
                self.__build_Vertex_Table(startIndex+1, father, oov+self.__word[startIndex:startIndex+1])
            else :    
                self.__build_Vertex_Table(startIndex+1, father, self.__word[startIndex:startIndex+1])   

    def __build_Edge_Table(self, word):
        vertex = self.__vertexTable[word]
        for child in vertex.child : 
            totalTime  =  self.__dic.get_Corpus_TotalTime()
            frequency  =  1 if (self.__dic.get_Word_Frequency(word) == 0)  else  self.__dic.get_Word_Frequency(word)
            twoGramFrequency = self.__dic.get_TwoGram_Frequency(word, child)
            sConst  = 0.9
            dConst  = 1-(1/ totalTime + 0.00001)
            weight = - math.log( sConst*(dConst*(twoGramFrequency/frequency)+(1-dConst)) + (1-sConst)*((frequency+1)/totalTime) )
            self.__edgeTable.setdefault(word+"#"+child, NetEdge(weight))
            self.__build_Edge_Table(child)

    def __build_DP_Table(self, word, visited):
        vertex = self.__vertexTable[word]
        if len(visited) == len(self.__DPTable):
            return
        for child in vertex.child : 
            weight = self.__edgeTable[word+"#"+child].weight
            if  self.__DPTable[child]['cost']  > weight + self.__DPTable[word]['cost'] :
                self.__DPTable[child]['cost']  = weight + self.__DPTable[word]['cost']
                self.__DPTable[child]["predecessor"] = word
        while  len(visited) < len(self.__DPTable) and len(vertex.child) !=0 :   
            minChild = ""
            minWeight = math.inf        
            for child in vertex.child :
                if self.__edgeTable[word+"#"+child].weight < minWeight and  (child not in visited) :
                    minWeight = self.__edgeTable[word+"#"+child].weight
                    minChild = child 
            if minChild == "" :   
                return    
            visited.add(minChild)           
            self.__build_DP_Table(minChild,visited)    
    def __find_Correct_Sentence(self):
        word = "EOS"
        while word != None :
            self.__sentence.insert(0,word)
            word = self.__DPTable[word]["predecessor"]    
                    