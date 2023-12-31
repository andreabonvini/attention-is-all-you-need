import argparse
from typing import List
import os


def get_requirements_list() -> List[str]:
    parser = argparse.ArgumentParser(description='Script to train a Transformer model.')
    parser.add_argument('-d', '--device', choices=["CPU", "GPU"], required=True)
    args = parser.parse_args()
    requirements_list = [
        "spacy==3.7.2",
        "torchtext==0.16.0"
    ]

    if args.device.lower() == "cpu":
        requirements_list.append("torch==2.1.0")
    elif args.device.lower() == "gpu":
        requirements_list.append("torch==2.1.0+cu121")
    return requirements_list


if __name__ == "__main__":
    requirements = get_requirements_list()
    os.system("pip install --upgrade pip")
    for r in requirements:
        os.system(f"pip install {r} --force-reinstall")
