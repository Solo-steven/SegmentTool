from  Dictionary.Dictionary import Dictionary
from WordNet.WordNet_v_1_1 import WordNet

from HMM.modal import FirstOrderHMM
import time 

hiddenState= ["Healthy", "Fever"]
visibleState = ["normal", "cold", "dizzy"];


dic = Dictionary("./Dictionary/TestSet.utf8");
wordNet = WordNet (dic);
wordNet.build_Word_Net("李廷偉服務")
wordNet.print_Sentence()

"""
piVector = [0.57, 0.43];
trasition = [
    [.7, .3],
    [.4, .6]];
emission = [
    [0.5, 0.4, 0.1],
    [0.1, 0.3, 0.6]];
modal = FirstOrderHMM(hiddenState, visibleState,piVector, trasition, emission)

seqence = []
start = time.time()
for i in range(100):
    seqence.append(modal.generateSample(100))
end = time.time()
print(end - start)    
start = time.time()
modal.trainModal(seqence ,hiddenState, visibleState);
end = time.time()
print(end - start)


test = [0]*10000 ;
numTest = np.zeros(10000)
newNumTest = numTest.reshape((100, 100))

sumNumber = 0 ;
start = time.time()
for t in range(100):
    sumNumber =0 ;
    for i in range(10000):
        sumNumber += test[i]
end = time.time()
print(end-start)


sumNumber = 0 ;
start = time.time()
for t in range(100):
    sumNumber =0 ;
    for i in range(10000):
        sumNumber += numTest[i]
end = time.time()
print(end-start)


"""







