import argparse
import os
from dotenv import load_dotenv

load_dotenv()

from .retriever import build_vectorstore, load_vectorstore, get_qa_chain

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--build', action='store_true', help='Build the vector store from data/')
    parser.add_argument('--data', default='../data', help='Data folder path')
    parser.add_argument('--query', help='Run a demo query')
    args = parser.parse_args()

    if args.build:
        print('Building vector store...')
        vs = build_vectorstore(args.data)
        print('Built and persisted to default directory')
        return

    vs = load_vectorstore()
    qa = get_qa_chain(vs)

    if args.query:
        res = qa.run(args.query)
        print('Answer:\n', res)
    else:
        print('Run with --build to build or --query "..." to ask a question')

if __name__ == '__main__':
    main()
