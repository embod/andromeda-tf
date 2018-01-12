import argparse

from Controller import Controller
from config import config

if __name__ == "__main__":

    controller = Controller(config["apikey"], config["agent_id"])

    controller.train(100000)
