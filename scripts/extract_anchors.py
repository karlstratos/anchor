# Author: Karl Stratos (stratos@cs.columbia.edu)
"""
This module is used to extract anchor words from labeled sequences like

Influential[sep]ADJ members[sep]NOUN of[sep]ADP ...

where [sep] is a special separating symbol. E.g.,

>> python3 extract-anchors.py [labeled_sequences] --output /tmp/anchors.txt
"""
import argparse
from collections import Counter

def extract_anchors(args):
    """Extracts anchor words."""
    word_to_tag_map = {}
    anchors = {}
    tag_count = Counter()
    with open(args.input_path, "r") as infile:
        for line in infile:
            tokens = line.split()
            if not tokens: continue
            for token in tokens:
                word_tag = token.split(args.sep)
                assert(len(word_tag) == 2)
                (word, tag) =  word_tag
                if not word in word_to_tag_map:
                    word_to_tag_map[word] = Counter()
                word_to_tag_map[word][tag] += 1
                anchors[tag] = []
                tag_count[tag] += 1

    for word in word_to_tag_map:
        if len(word_to_tag_map[word]) == 1:
            tag = list(word_to_tag_map[word].keys())[0]
            count = list(word_to_tag_map[word].values())[0]
            anchors[tag].append((word, count))

    for tag in anchors:
        anchors[tag] = sorted(anchors[tag], key=lambda x:x[1], reverse=True)
        anchors[tag] = [pair for pair in anchors[tag] if pair[1] > args.cutoff]

    sorted_tags = sorted(tag_count.items(), key=lambda x:x[1], reverse=True)
    num_distinct_tags = len(tag_count)
    final_anchor_list = []
    tracker = Counter()

    while len(final_anchor_list) < num_distinct_tags:
        print("----------------ROUND--------------")
        for tag, _ in sorted_tags:
            if tracker[tag] < len(anchors[tag]):
                ind = tracker[tag]
                final_anchor_list.append(anchors[tag][ind][0])
                tracker[tag] += 1
                print(tag, anchors[tag][ind][0])
                if len(final_anchor_list) >= num_distinct_tags: break
            else:
                print(">>>>>>>>>>Skipping ", tag)

    output_path = args.output_path if args.output_path else \
        args.input_path + ".anchors-min" + str(args.cutoff)
    with open(output_path, "w") as outfile:
        for anchor in final_anchor_list:
            outfile.write(anchor + "\n")

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("input_path", type=str, help="path to input data")
    argparser.add_argument("--output_path", type=str, default="",
                           help="path to output (auto if not specified)")
    argparser.add_argument("--sep", type=str, default="__<label>__",
                           help="word-tag separator")
    argparser.add_argument("--cutoff", type=int, default=0,
                           help="frequency <= this not considered")
    parsed_args = argparser.parse_args()
    extract_anchors(parsed_args)
