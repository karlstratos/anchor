# Author: Karl Stratos (stratos@cs.columbia.edu)
"""
This module is used to display possible labels of an observation sorted in
decreasing frequency. E.g.,

>> python3 display_label.py [labeled_sequences]
"""
import argparse
from collections import Counter

def display_label(args):
    """Interactively displays possible labels of an observation."""
    label_count = {}
    with open(args.data_path, "r") as infile:
        for line in infile:
            toks = line.split()
            for tok in toks:
                x, h = tok.split(args.sep)
                if not x in label_count:
                    label_count[x] = Counter()
                label_count[x][h] += 1

    for x in label_count:
        label_count[x] = sorted(label_count[x].items(), key=lambda x: x[1],
                              reverse=True)

    while True:
        try:
            x = input("Type an observation string (or just quit the program): ")
            if not x in label_count:
                print("There is no \"{0}\"".format(x))
            else:
                for h, frequency in label_count[x]:
                    print("\t\t{0}\t\t{1}".format(frequency, h))
        except (KeyboardInterrupt, EOFError):
            print()
            exit(0)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("data_path", type=str, help="labeled sequences")
    argparser.add_argument("--sep", type=str, default="__<label>__",
                           help="observation-label separator")
    parsed_args = argparser.parse_args()
    display_label(parsed_args)
