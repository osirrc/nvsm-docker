import argparse

from manager import Manager

if __name__ == "__main__":
    manager = Manager()

    parser = argparse.ArgumentParser()
    parser_sub = parser.add_subparsers()

    # Specific to prepare
    parser_prepare = parser_sub.add_parser("prepare")
    parser_prepare.set_defaults(run=manager.prepare)
    parser_prepare.add_argument("--repo", required=True, type=str, help="the image repo (i.e., osirrc2019/anserini)")
    parser_prepare.add_argument("--tag", default="latest", type=str, help="the image tag (i.e., latest)")
    parser_prepare.add_argument("--save_id", default="save", type=str,
                                help="the ID of the saved image (to search from)")
    parser_prepare.add_argument("--collections", required=True, nargs="+", help="the name of the collection")
    parser_prepare.add_argument("--opts", nargs="+", default="", type=str, help="the args passed to the index script")

    parser_prepare = parser_sub.add_parser("train")
    parser_prepare.set_defaults(run=manager.train)
    parser_prepare.add_argument("--repo", required=True, type=str, help="the image repo (i.e., osirrc2019/anserini)")
    parser_prepare.add_argument("--tag", default="latest", type=str, help="the image tag (i.e., latest)")
    parser_prepare.add_argument("--save_id", default="save", type=str,
                                help="the ID of the saved image (to search from)")
    parser_prepare.add_argument("--opts", nargs="+", default="", type=str, help="the args passed to the index script")

    # Specific to search
    parser_search = parser_sub.add_parser("search")
    parser_search.set_defaults(run=manager.search)
    parser_search.add_argument("--repo", required=True, type=str, help="the image repo (i.e., osirrc2019/anserini)")
    parser_search.add_argument("--tag", default="latest", type=str, help="the image tag (i.e., latest)")
    parser_search.add_argument("--save_id", default="save", type=str, help="the ID of the saved image (to search from)")
    parser_search.add_argument("--opts", nargs="+", default="", type=str, help="the args passed to the search script")

    # Parse the args
    args = parser.parse_args()

    args.run(args)
