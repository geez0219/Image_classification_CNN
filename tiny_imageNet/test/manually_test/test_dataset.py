import pickle
import numpy as np
import cv2


def test_dataset():
    """
    this test require manually check whether the image objects are consistent with their title
    """
    (x_train, y_train), (x_test, y_test) = pickle.load(open("dataset.pkl", "rb"))
    id_name_list = pickle.load(open("id_name_list.pkl","rb"))

    # test training data
    for i in np.random.permutation(x_train.shape[0])[:10]:
        cv2.imshow(id_name_list[y_train[i]], x_train[i])
        cv2.waitKey(0)

    # test training data
    for i in np.random.permutation(x_test.shape[0])[:10]:
        cv2.imshow(id_name_list[y_test[i]], x_test[i])
        cv2.waitKey(0)


if __name__ == "__main__":
    test_dataset()
