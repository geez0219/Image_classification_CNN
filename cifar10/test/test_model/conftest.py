import pytest
import pickle


@pytest.fixture(scope="module")
def load_data():
    yield pickle.load(open("dataset.pkl", "rb"))
    print("terminate load_data fixture")