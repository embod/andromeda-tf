import argparse
from embodsdk import Client

def state_callback(state):
    print(state)


if __name__ == "__main__":
    client = Client("ne6KsGiUDffndKrlXwU63tQO6UA", state_callback)

    client.add_agent("b4f1e94f-5e3c-483e-b2d2-b6ea9f2ad74a")

    client.run_loop()