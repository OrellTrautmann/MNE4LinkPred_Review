import pandas as pd
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Multiplex network embedding tester pipeline')
    parser.add_argument('-r', '--retweet', type=str, default='Datasets/Twitter/retweet_edges.csv',
    					help='path to the retweet dataset')
    parser.add_argument('-s', '--social', type=str, default='Datasets/Twitter/social_edges.csv',
    					help='path to the social dataset')
    parser.add_argument('-o', '--outfile', type=str, default='Datasets/Twitter/twitter.edges.txt',
    					help='file with path to save the network')
    return parser.parse_args()

args = parse_args()

if not os.path.exists(args.retweet):
    raise FileNotFoundError("retweet file does not exist!")
if not os.path.exists(args.social):
    raise FileNotFoundError("social file does not exist!")

dir_path = os.path.dirname(args.outfile)
os.makedirs(dir_path, exist_ok=True)

retweet_df = pd.read_csv(args.retweet, sep=",")
retweet_df = retweet_df.drop(columns=[" weight"])
retweet_df.insert(loc=0, column='layer', value=[1]*retweet_df.shape[0])

social_df = pd.read_csv(args.social, sep=",")
social_df.insert(loc=0, column='layer', value=[2]*social_df.shape[0])

twitter_df = pd.concat([retweet_df, social_df], ignore_index=True)
twitter_df.insert(loc=len(twitter_df.columns), column='weight', value=[1]*twitter_df.shape[0])
twitter_df.to_csv(args.outfile, header=False, index=False, sep=" ")

print("done!")


