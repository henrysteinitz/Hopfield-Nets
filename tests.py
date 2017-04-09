from hebbian_hn import HebbianHopfieldNet

pattern1 = [0, 0, 0, 0, 1,
            0, 0, 0, 1, 0,
            0, 0, 1, 0, 0,
            0, 1, 0, 0, 0,
            1, 0, 0, 0, 0]

pattern2 = [0, 0, 0, 0, 1,
            0, 0, 0, 0, 1,
            0, 0, 0, 0, 1,
            0, 0, 0, 0, 1,
            1, 1, 1, 1, 1]

input = [1, 0, 0, 0, 1,
         0, 0, 0, 0, 0,
         0, 0, 0, 0, 0,
         0, 1, 0, 1, 1,
         1, 1, 1, 1, 1]

patterns = [pattern1, pattern2]

hnn = HebbianHopfieldNet(patterns)
print(hnn.play(input).reshape(5,5))
