from hebbian_hn import HebbianHopfieldNet

patternX = [1, 0, 0, 0, 1,
            0, 1, 0, 1, 0,
            0, 0, 1, 0, 0,
            0, 1, 0, 1, 0,
            1, 0, 0, 0, 1]

patternA = [0, 1, 1, 1, 0,
            0, 1, 0, 1, 0,
            0, 1, 1, 1, 0,
            0, 1, 0, 1, 0,
            0, 1, 0, 1, 0]

patternC = [1, 1, 1, 1, 0,
            1, 0, 0, 0, 0,
            1, 0, 0, 0, 0,
            1, 0, 0, 0, 0,
            1, 1, 1, 1, 0]

patternE = [1, 1, 1, 1, 0,
            1, 0, 0, 0, 0,
            1, 1, 1, 1, 0,
            1, 0, 0, 0, 0,
            1, 1, 1, 1, 0]

patternPI = [1, 1, 1, 1, 1,
             0, 1, 0, 1, 0,
             0, 1, 0, 1, 0,
             0, 1, 0, 1, 0,
             0, 1, 0, 1, 0]

input = [1, 1, 1, 1, 1,
         0, 0, 0, 1, 0,
         0, 1, 0, 1, 0,
         0, 0, 0, 1, 1,
         0, 1, 1, 1, 0]

patterns = [patternX, patternA,  patternC, patternE, patternPI]

hnn = HebbianHopfieldNet(patterns)
print(hnn.play(input).reshape(5,5))
