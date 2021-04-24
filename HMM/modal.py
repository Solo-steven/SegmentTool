""" 
======   實作隱性馬可夫模型(HMM)   ======
"""

"""   這個class 負責一階馬可夫模型的實現
    1. Hidden State  :
    2. Visible State :
    3. Transition  : 
    4. Emission    :
    5. PIVector    : 
"""

class FirstOrderHMM: 
    def __init__(self, hiddenState=None, visibleState=None, piVector=None, transition=None, emission=None):
        self.__hiddenState =hiddenState ;
        self.__visibleState =visibleState ;
        self.__transition = transition ;
        self.__emission  = emission ;
        self.__piVector  = piVector ;

    def __drawFrom(self, state, p):
        return  np.random.choice(len(state), 1, replace=False,p= p)[0]  

    def generateSample(self, size):
        stateSeqence = [[0]*size,[0]*size]; 
        stateSeqence[0][0] =  self.__drawFrom(self.__hiddenState , self.__piVector)
        stateSeqence[1][0] =  self.__drawFrom(self.__visibleState, self.__emission[stateSeqence[0][0]])
        pre_hiddenState = stateSeqence[0][0];
        for t in range(1,size):
            stateSeqence[0][t] = self.__drawFrom(self.__hiddenState, self.__transition[pre_hiddenState])
            stateSeqence[1][t] = self.__drawFrom(self.__visibleState, self.__emission[stateSeqence[0][t]]);
            pre_hiddenState = stateSeqence[0][t]
        ##for i in range(size):
        ##    stateSeqence[0][i] = self.__hiddenState[stateSeqence[0][i]];
        ##    stateSeqence[1][i] = self.__visibleState[stateSeqence[1][i]];   
        return stateSeqence;  

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
                self.__transition[i][j] /= sumNumber;              

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
                sumNumber += self.__emission[i][j];
            for j in range(len(self.__emission[i])):
                self.__emission[i][j] /=  sumNumber     

    def __establishPiVector(self,collection):
        for i in range(len(collection)):
            sample = collection[i]
            self.__piVector[sample[0][0]] += 1    
        sumNumber =0 ;
        for i in range(len(self.__piVector)):
            sumNumber += self.__piVector[i]
        for i in range(len(self.__piVector)):
            self.__piVector[i] /= sumNumber   
  
    def trainModal(self, collection, hiddenState, visibleState):
        self.__hiddenState = hiddenState;
        self.__transition = [[0]*len(hiddenState) for i in range(len(hiddenState)) ];
        self.__establishTransition(collection);
        self.__visibleState = visibleState;
        self.__emission = [ [0]*len(visibleState) for i in range(len(hiddenState)) ] ;
        self.__establishEmission(collection);
        self.__piVector =  [0] * len(hiddenState) 
        self.__establishPiVector(collection)