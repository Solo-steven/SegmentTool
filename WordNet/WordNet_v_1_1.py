"""
=====   實作二元語法分析   =====
"""

import math

"""   這個class處理分詞網路的節點資訊
     1. word : 這個節點的詞。
     2. index : 這個詞在原本句子的開始下標。
     3. cost : 最短路徑演算法時紀錄節點的最小花費。
     4. predecessor : 最短路徑演算法時紀錄節點的前繼者。
"""    
class Vertex :
    def __init__(self, word, index):
        self.word  = word 
        self.index = index
        self.cost  = math.inf 
        self.predecessor = None;
        
"""   這個class處理分詞網路的邊資訊
     1. word : 不是單詞，是two gram。例如要代表『商品』『和』兩個節點的邊時，
               就用『商品#和』儲存。
     2. index： 這個two gram的下標開始位置。                                                                                                      
"""
class Edge :
    def __init__(self ,word, index, weight):
        self.word  = word
        self.index  = index
        self.weight = weight


"""      這個class處理分詞網路的分析。
        1. dic      : 代表需要用到的字典。(後記零)
        2. word     : 代表需要分詞的句子。
        3. sentence : 代表分析完後的分詞結構。
        4. vertexTable : 節點資訊。為Vertex object。
        5. edgeTable   : 邊的資訊。為Edge object。                                                                                         
"""    
class WordNet:
    def __init__(self, dic,word=None):
        self.__dic  = dic
        self.__word = None
        self.__sentence    = None
        self.__vertexTable = None
        self.__edgeTable   = None
        if word is not None:
            self.build_Word_Net(word)
    """   印出經過圖行計算之後，最有可能的分詞組成   """
    def print_Sentence(self):
        for i in range(1,len(self.__sentence)):
            print(self.__sentence[len(self.__sentence)-1-i],end=" ")
        print("")

    """   印出經過圖行計算中，所運用到的任何紀錄    """
    def print_Net_Data(self):
        print("Vertex Table : ")
        for  i in range(len(self.__vertexTable)):
            print("Start at Index",i," : ",end="")
            for vertex in self.__vertexTable[i]:
                print(vertex.word, vertex.index ,end=",")
            print("")
        print("Edge Table :")    
        for i in range(len(self.__edgeTable)):
            print("Start at Index",i, " : ",end="")
            for edge in self.__edgeTable[i]:
                print(edge.word, edge.index, edge.weight, end=",")
            print("") 
        print("Shortest Path Info :")
        for  i in range(len(self.__vertexTable)):
            print("Start at Index",i," : ",end="")
            for vertex in self.__vertexTable[i]:
                print(vertex.predecessor, vertex.cost ,end=",")
            print("")   

    """   建構一個新的詞網並分析 
         1. 目的 ： 根據一個新輸入的詞，重新建構WordNet的相關資訊。
         2. 方法 ： 建構節點table -> 建構邊table -> 計算最短路徑 -> 找出最短路徑
    """        
    def build_Word_Net(self, word):
        ## reset the reference word
        self.__word = word 
        ## clear the Vertex Table and Init it 
        self.__vertexTable = [ []for i in range(len(word)+2)]
        self.__vertexTable[0].append(Vertex("BOS",0))
        self.__vertexTable[-1].append(Vertex("EOS",len(word)+1))
        self.__build_Vertex_Table()
        # clear the Edge Table and Init it 
        self.__edgeTable = [ [] for i in range(len(word)+2)]
        self.__build_Edge_Table()
        # Init BOS Vertex for dijkstra algorithm to find shortest path. 
        self.__vertexTable[0][0].cost =0 ;
        self.__build_DP_Info(self.__vertexTable[0][0])
        self.__structure_Correct_Sentence();

    """   處理字串長度
         1.目的 ： 字首字尾的節點中的word是用BOS或EOS，在用len()求長度的時明顯不合預期。
         2.方法 ： 當為字首的時候，回傳1，讓index+長度可以移動到下一個state，而字尾時，
                  回傳0，不讓存取範圍超過。           
    """
    def __Word_Length(self,word):
        if word == "BOS" :
            return 1
        if word == "EOS" : 
            return 0
        return len(word)   

    """   原子分詞
         1. 目的 ： 保證句子的連通性，即保證每一個vertex都存在一個path到EOS的節點。
         2. 方法 ： 遞迴表示狀態轉移，當此index開始的字串為0個的時候，將當前index的字加入
                   WordNet中，並移動到下一個字的狀態。    
    """
    def __atomic_Segment(self, index=0):
        if len(self.__vertexTable[index]) == 0:
            word = self.__word[index-1:index]
            self.__vertexTable[index].append(Vertex(word, index))
        if self.__vertexTable[index][0].word == "EOS":
            return 
        for vertex in self.__vertexTable[index] :
            toIndex = index + self.__Word_Length(vertex.word)
            self.__atomic_Segment(toIndex)

    """   建構Vertex Table (後記一)
         1. 目的 ： 建構一個長度為n的list，每一個index代表由此下標開始的詞。
         2. 方法 ： 類似全切分，由當前下標為首，在字典找是否存在，若存在，則
                   加入對應下標的list。
    """                
    def __build_Vertex_Table(self):
        for index in range(1, len(self.__word)+1):
            start = index-1 
            for end in range(start+1, len(self.__word)+1):
                word = self.__word[start:end]
                if word in self.__dic:
                    self.__vertexTable[index].append(Vertex(word,index))
        self.__atomic_Segment()  

    """   計算兩個節點的邊權重
         1. 目的 ： 由公式和字典中的二元詞頻，計算兩個節點相連的權重。
         2. 方法 ： 使用一元詞頻平滑二元詞頻的公式。  
    """
    def __calcul_Weight(self, beginWord, endWord):
        totalTime = self.__dic.get_Corpus_TotalTime()
        frequency =  self.__dic.get_Word_Frequency(beginWord)
        frequency = frequency if frequency >0 else 1
        twoGramFrequency = self.__dic.get_TwoGram_Frequency(beginWord, endWord)
        sConst  = 0.9
        uConst  = 1-(1/ totalTime + 0.00001)
        return - math.log( sConst*(uConst*(twoGramFrequency/frequency)+(1-uConst)) + (1-uConst)*((frequency+1)/totalTime) )

    """   建構Edge Table 
         1. 目的 ： 類似Vertex Table的方法。建構一個長度為n的list，每一個index代表由此下標開始的二元連詞。
         2. 方法 ： DFS。每次拜訪一個節點時，任意選一個子點，計算邊，再拜訪那個子點。直到子點都被拜訪。
    """
    def __build_Edge_Table(self, index=0, record=set()):       
        for beginVertex in self.__vertexTable[index]:
            if beginVertex not in record :  
                record.add(beginVertex) 
                toIndex = index+self.__Word_Length(beginVertex.word)  
                for toVertex in self.__vertexTable[toIndex] :
                    weight = self.__calcul_Weight(beginVertex.word, toVertex.word)
                    self.__edgeTable[beginVertex.index].append(Edge(beginVertex.word+"#"+toVertex.word, beginVertex.index, weight))
                    self.__build_Edge_Table(toVertex.index, record)

    def __find_Edge(self, beginWord, toWord, index):
        twoGram = beginWord+"#"+toWord;
        for edge in self.__edgeTable[index]:
            if edge.word == twoGram :
                return edge.weight
    """   建構最短路徑的資訓
         1. 目的 ： 根據建構好的節點和邊的權重，建構行程最短路徑的資訊。
         2. 方法 ： 使用dijkstra建構最短路徑的DP info。 
    """
    def __build_DP_Info(self, vertex, record=set()):
        record.add(vertex);
        toIndex = vertex.index + self.__Word_Length(vertex.word)
        for toVertex in self.__vertexTable[toIndex]:
            weight = self.__find_Edge(vertex.word, toVertex.word, vertex.index)
            if(toVertex.cost > vertex.cost + weight):
                toVertex.cost = vertex.cost + weight;
                toVertex.predecessor = vertex.word
        while 1 : 
            minWeight = math.inf
            minVertex = None ;
            for toVertex in self.__vertexTable[toIndex]:
                weight = self.__find_Edge(vertex.word, toVertex.word, vertex.index)
                if( (toVertex not in record) and (weight < minWeight)):
                    minWeight = weight
                    minVertex = toVertex 
            if(minVertex is not None):
                self.__build_DP_Info(minVertex, record)
            else: 
                return 

    """   根據dijkstra的資訊找出最短路徑
         1. 目的 ： 找出最短路徑，即代表最有可能的分詞組成。
         2. 方法 ： 由EOS向前尋找predecessor。
    """
    def __structure_Correct_Sentence(self):
        self.__sentence = []
        vertex = self.__vertexTable[-1][0]
        while vertex.predecessor is not None :
            self.__sentence.append(vertex.predecessor)
            predecessorIndex = vertex.index-self.__Word_Length(vertex.predecessor)
            for predecessor in self.__vertexTable[predecessorIndex]:
                if(predecessor.word == vertex.predecessor):
                    vertex = predecessor 
                    break

"""  後記零： 字典物件需要滿足的方法
     1. __contains__(self, word:str) -> True or False
       => 功用 ： 字是否存在於字典中。
     2. get_Corpus_TotalTime(self) -> number 
       => 功用 ： 回傳字典『詞』(或一元詞)的總詞頻。
     3. get_Word_Frequency(self, word:str) ->number 
       => 功用 ： 回傳字典中指定的詞頻(一元詞頻)。
     4. get_TwoGram_Frequency(self, beginWord:str, endWord:str) -> number
       => 功用 ： 回傳字典中指定的二元詞頻
"""

"""  後記一 ： 節點的資料結構
    ㄧ. 紀錄節點的資料結構 ： 當長度為n的字串輸入時，建構一個長度為n+2的list，每一個index存放一個list。
        1. 每個list的功用 ： 儲存以當前index開始的詞。
        2. 加入字首字尾 ： 多加入兩個空list，代表字首和字尾。     
    二. 例： BOS 測試的工作 EOS 
          [0] -> 以『BOS』開頭的詞： BOS(begin of sentence)是自訂的抽象節點，因此沒有。
          [1] -> 以『測』開頭的詞 ： 測試、測
          [2] -> 以『試』開頭的詞 ： 試
          [3] -> 以『的』開頭的詞 ： 的
          [4] -> 以『工』開頭的詞 ： 工作、工
          [5] -> 以『作』開頭的詞 ： 作
          [6] -> 以『EOS』開頭的字： EOS(end of sentence)是自訂的抽象節點，因此沒有。
        1.好處：減少資料結構的複雜度，也可以清楚的知道父or子的存在。
            (1) 求子節點: 當前index直接加上字串的長度，例如『測試』的子節點，就是index + len(測試) = 1 + 2 = 3，
                         就是Table中為index=3的所有節點，即『的』字。
            (2) 求父節點: 同理，只是用減的而已。 
"""