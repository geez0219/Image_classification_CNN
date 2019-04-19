import pickle
import numpy as np

if __name__ == "__main__":

    model_weight = pickle.load(open("model_weight.pkl","rb"))
    model_weight2 = pickle.load(open("model_weight2.pkl","rb"))

    for i in range(len(model_weight)):
        assert np.array_equal(model_weight[i], model_weight2[i])

    print('they are the same')