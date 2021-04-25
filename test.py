from HMM.BaseModel import BaseFirstOrderHMM
from HMM.Model import FirstOrderHMM
import time , math

hiddenState= ["Healthy", "Fever"]
visibleState = ["normal", "cold", "dizzy"];
piVector = [0.6, 0.4];
trasition = [
    [.7, .3],
    [.4, .6]];
emission = [
    [0.5, 0.4, 0.1],
    [0.1, 0.3, 0.6]];
modal = FirstOrderHMM(hiddenState, visibleState,piVector, trasition, emission)

M = FirstOrderHMM(hiddenState, visibleState,piVector, trasition, emission)

"""
seqence = []
start = time.time()
for i in range(200):
    seqence.append(modal.generateSample(150))
end = time.time()
print(end - start)    
start = time.time()
modal.establishModel(seqence ,hiddenState, visibleState);
end = time.time()
print(end - start)
"""
data = modal.predictSeqencesProbility([0,1,2])

for key, value in data.items():
    print(key, value)