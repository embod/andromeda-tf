import argparse
from logger import setup_custom_logger
from Controller import Controller
from config import config

if __name__ == "__main__":

    setup_custom_logger("root")

    controller = Controller(config["apikey"], config["agent_id"])

    controller.train(1000000)
