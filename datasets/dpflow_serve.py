import argparse
from dplink import serve_remote

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--link_config", type=str)
    args = parser.parse_args()

    serve_remote(args.link_config)