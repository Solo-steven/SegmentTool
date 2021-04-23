""" 
======   實作隱性馬可夫模型(HMM)   ======
"""

class FirstOrderHMM: 
    def __init__(self, piVector=None, trasitionA=None, trasitionB=None):
        self.__piVector= piVector;
        self.__trasitionA = trasitionA ;
        self.__trasitionB = trasitionB ;
    def generateSamlp(self, size):

