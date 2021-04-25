""" 
======   實作『基礎』隱性馬可夫模型(HMM)   ======
    註：這個模型的train method需要輸入顯隱兩個狀態的
        序列，不符合常見的只有顯狀態的資料。
"""
import math

"""   這個class 負責一階馬可夫模型的實現 
    1. Hidden State  :
    2. Visible State :
    3. Transition  : 
    4. Emission    :
    5. PIVector    : 
"""

class BaseFirstOrderHMM: 
    def __init__(self, hiddenState=None, visibleState=None, piVector=None, transition=None, emission=None):
        self.__hiddenState  = hiddenState 
        self.__visibleState = visibleState 
        self.__transition = transition 
        self.__emission   = emission 
        self.__piVector   = piVector  

    """   樣本生成
        1. 目的  : 給定兩個向量，一個代表狀態，一個代表狀態出現的機率，選出狀態。
        2. 方法  : 使用numpy的生成函數。
    """
    def __drawFrom(self, state, p):
        return  np.random.choice(len(state), 1, replace=False,p= p)[0]  

    """   樣本生成
        1. 目的  : 給定模型參數，和一個長度，生成指定長度的模型樣本。
        2. 方法  : 使用HMM的基本定義，即按照顯隱狀態間的自然轉移。
    """
    def generateSample(self, size):
        stateSeqence = [[0]*size,[0]*size]; 
        stateSeqence[0][0] =  self.__drawFrom(self.__hiddenState , self.__piVector)
        stateSeqence[1][0] =  self.__drawFrom(self.__visibleState, self.__emission[stateSeqence[0][0]])
        pre_hiddenState = stateSeqence[0][0];
        for t in range(1,size):
            stateSeqence[0][t] = self.__drawFrom(self.__hiddenState, self.__transition[pre_hiddenState])
            stateSeqence[1][t] = self.__drawFrom(self.__visibleState, self.__emission[stateSeqence[0][t]])
            pre_hiddenState = stateSeqence[0][t]
        ##for i in range(size):
        ##    stateSeqence[0][i] = self.__hiddenState[stateSeqence[0][i]];
        ##    stateSeqence[1][i] = self.__visibleState[stateSeqence[1][i]];   
        return stateSeqence;  

    """   Learning Problem solver 
        1. 目的 ： 給予樣本，輸出建構出可能的 Transition Array 參數。
        2. 方法 ： 基礎紀錄資料量的方法(應該是頻率學派的觀點)，需要同時輸入兩個state的資料。
    """
    def __establishTransition(self, collection):
        for i in range(len(collection)):
            sample = collection[i]
            pre_hiddenState = sample[0][0];
            for t in range(1,len(sample[0])):
                self.__transition[pre_hiddenState][sample[0][t]] += 1;
                pre_hiddenState =  sample[0][t]
        for i in range(len(self.__transition)):
            sumNumber = 0;
            for j in range(len(self.__transition[i])):
                sumNumber += self.__transition[i][j]
            for j in range(len(self.__transition[i])):
                self.__transition[i][j] /= sumNumber 
                             
    """   Learning Problem solver 
        1. 目的 ： 給予樣本，輸出建構出可能的 Emission Array 參數。
        2. 方法 ： 基礎紀錄資料量的方法(應該是頻率學派的觀點)，需要同時輸入兩個state的資料。
    """
    def __establishEmission(self, collection):
        for i in range(len(collection)):
            sample = collection[i]
            for t in range(len(sample[1])):
                hiddenState = sample[0][t]
                visibleState = sample[1][t]
                self.__emission[hiddenState][visibleState]+=1
        for i in range(len(self.__emission)):
            sumNumber =0 
            for j in range(len(self.__emission[i])):
                sumNumber += self.__emission[i][j]
            for j in range(len(self.__emission[i])):
                self.__emission[i][j] /=  sumNumber  

    """   Learning Problem solver 
        1. 目的 ： 給予樣本，輸出建構出可能的 Pi Vector 參數。
        2. 方法 ： 基礎紀錄資料量的方法(應該是頻率學派的觀點)，需要同時輸入兩個state的資料。
    """
    def __establishPiVector(self,collection):
        for i in range(len(collection)):
            sample = collection[i]
            self.__piVector[sample[0][0]] += 1    
        sumNumber =0 ;
        for i in range(len(self.__piVector)):
            sumNumber += self.__piVector[i]
        for i in range(len(self.__piVector)):
            self.__piVector[i] /= sumNumber  

    """   Learning Problem solver 
        1. 目的 ： 給予樣本，輸出建構出可能的模型參數。
        2. 方法 ： 基礎紀錄資料量的方法(應該是頻率學派的觀點)，需要同時輸入兩個state的資料。
    """
    def establishModel(self, collection, hiddenState, visibleState):
        self.__hiddenState = hiddenState;
        self.__transition = [[0]*len(hiddenState) for i in range(len(hiddenState)) ];
        self.__establishTransition(collection);
        self.__visibleState = visibleState;
        self.__emission = [ [0]*len(visibleState) for i in range(len(hiddenState)) ] ;
        self.__establishEmission(collection);
        self.__piVector =  [0] * len(hiddenState) 
        self.__establishPiVector(collection)
        print(self.__transition, self.__emission, self.__piVector)

    """   Decoding Problem solver
        1. 目的 : 給予模型參數、觀察的時間序列，回傳最有可能的顯狀態和機率。
        2. 方法 : 用維特比演算法。其中機率有取對數，避免underflow，動態規劃使用滾動陣列優化。
    """
    def predictSeqencesProbility(self, observateState):
        pre_probability = [0] * len(self.__hiddenState)
        cur_probability = [0] * len(self.__hiddenState)
        path = [ [0] * len(self.__hiddenState) for i in range(len(observateState))]
        for i in range(len(self.__hiddenState)):
            cur_probability[i] =  -(math.log(self.__piVector[i]) + math.log(self.__emission[i][observateState[0]]))

        for t in range(1, len(observateState)):
            pre_probability = cur_probability 
            cur_probability = [0] * len(self.__hiddenState)
            for i in range(len(self.__hiddenState)):
              cur_probability[i] = math.inf
              for x in range(len(pre_probability)):
                  probability = pre_probability[x]+ -(math.log(self.__transition[x][i])+ math.log(self.__emission[i][observateState[t]]))
                  if(probability < cur_probability[i]):
                      cur_probability[i] = probability
                      path[t][i] = x

        max_probability = math.inf 
        end_point = 0
        for i in range(len(cur_probability)):
            if max_probability > cur_probability[i]:
                max_probability = cur_probability[i]
                end_point = i
        max_probability = math.exp(-max_probability)

        shortest_path = []
        for t in range(len(path)): 
            shortest_path.append(end_point)
            end_point = path[t][end_point]
        shortest_path.reverse()    
        return {"maxProbability" : max_probability , "hiddenState": shortest_path}       


""" 後記零 ： numpy 的評測
   1. 馬可夫模型會用到大量的矩陣(array、martix)的運算，似乎使用
      numpy是一個效率比較好的方法。
   2. numpy在『運算』上，是比較好。但在隱性馬可夫模型的樣本生成和
      統計生成參數時，運算的比例比較低，更多的是下標存取。而實測發現
      numpy在下標存取的速度，明顯比原生list低。因此，在樣本生成，和
      參數訓練的時候，我不用numpy進行運算。     
"""        