import pandas as pd
import argparse

"""WARNING: Not to be used on big datasets like Twitter!!!"""

def parse_args():
    parser = argparse.ArgumentParser(description='Multiplex network embedding tester pipeline')
    parser.add_argument('-p', '--path', type=str, default='Datasets/Vickers/Vickers-Chan-7thGraders_multiplex.edges.txt',
    					help='path to the dataset')
    return parser.parse_args()

def read_csv(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path, sep=" ", header=None)
    df.rename(columns={0:"layer", 1:"source", 2:"target", 3:"weight"}, inplace=True)
    return df

def reindex_nodes(file_path: str) -> None:
    df = read_csv(file_path)
    node_list = list(set(df["source"]).union(set(df["target"])))
    df.loc[:,["source", "target"]] = df.loc[:,["source", "target"]].map(lambda x: node_list.index(x))
    df.to_csv(file_path, sep=" ", index=False, header=False)
    

if __name__ == "__main__":
    args = parse_args()
    reindex_nodes(args.path)
    

