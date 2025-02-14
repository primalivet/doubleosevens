import os
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str)
    args = parser.parse_args()
    for i in range(5):
        os.system(f"python extraction.py -m gpt-4o-mini -p openai \"{args.input}\"")

if __name__ == '__main__':
    main()
