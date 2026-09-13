from .go2_stand_up import train, env

def main() -> None:
    train()
    env.close()
