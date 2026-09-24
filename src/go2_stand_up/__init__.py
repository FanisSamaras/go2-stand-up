from .go2_stand_up import train,load_test
from .go2_stand_up_PID import test

def train_dog() -> None:
    train()

def load_dog() -> None:
    load_test(num_of_steps=700_000,steps=5000)

def test_dog_PID() -> None:
    test()
