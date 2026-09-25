from .go2_stand_up import train,load_test, resume_training
from .go2_stand_up_PID import test

def train_dog() -> None:
    train()

def load_dog() -> None:
    load_test(num_of_steps=2_000_000,steps=5000)

def resume_train_dog():
    resume_training()

def test_dog_PID() -> None:
    test()
