"""
From this directory call:

python setup.py build
mv /Users/admin/Documents/QMUL/MSc\ Project/Code/HiddiniPractice/Hiddini/build/lib.macosx-10.9-x86_64-3.7/hiddini.cpython-37m-darwin.so /Users/admin/Documents/QMUL/MSc\ Project/Code/HiddiniPractice/Hiddini_2/Hiddini/
python Testing_Hiddini.py
"""
from hiddini import HMMRaw, HMMDiscrete, ObservationsDiscrete
import numpy as np

print("Testing normal Viterbi..")

#Test Viterbi Working using example from Hernando et al (2005)
init_probs = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
trans_probs = np.matrix([[0.0, 0.5, 0.0, 0.5, 0.0],
                       [0.0, 0.5, 0.5, 0.0, 0.0],
                       [0.0, 0.5, 0.5, 0.0, 0.0],
                       [0.0, 0.0, 0.0, 0.5, 0.5],
                       [0.0, 0.0, 0.0, 0.0, 1.0]])
chord_probs = np.matrix([[1.0, 0.0, 0.0, 0.0, 0.0],
                       [0.5, 0.5, 0.0, 0.0, 0.0],
                       [0.0, 0.5, 0.5, 0.0, 0.0],
                       [0.0, 0.5, 0.0, 0.5, 0.0],
                       [0.0, 0.0, 0.0, 0.0, 1.0]])
obs = np.array([0,1,1,1,4])
hmm = HMMDiscrete(chord_probs, trans_probs, init_probs)
outputSequence = hmm.decodeMAP(obs)[0]
np.testing.assert_array_equal(outputSequence, np.array([0,3,3,3,4]))
print('Output Sequence:', outputSequence)


print("Testing Entropy-Viterbi..")
#Test entropy working using the same example from the 2005 paper
outputSequence = hmm.decodeMAP_with_entropy(obs)
#Check the viterbi algorithm still works
np.testing.assert_array_equal(outputSequence[0], np.array([0,3,3,3,4]))
#Check that entropy has been correctly calculated as 0
assert outputSequence[2] == 0
print("C++ has returned:", outputSequence)

#Now check that the sequential entropy termination has worked as expected.
print("Testing Sequential Entropy..")
outputSequence = hmm.decodeMAP_with_sequential_entropy(obs);
print(outputSequence)

print("Testing Framewise Entropy..")
outputSequence = hmm.decodeMAP_with_framewise_entropy(obs);
print(outputSequence)
