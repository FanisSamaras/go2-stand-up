from .go2_stand_up import train, load_test,env
from .go2_stand_up_PID import test

def train_dog() -> None:
    train()
    env.close()

def load_dog() -> None:
    load_test(250000)
    env.close()

def test_dog_PID() -> None:
    test()