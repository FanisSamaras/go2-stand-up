from .go2_stand_up import train, load_test,env

def main() -> None:
    train()
    env.close()
