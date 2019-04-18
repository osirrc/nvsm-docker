import argparse
import docker
import hashlib
import datetime

if __name__ == "__main__":
    client = docker.from_env(timeout=86.400)
    generate_save_tag = lambda tag, save_id: hashlib.sha256((tag + save_id).encode()).hexdigest()
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, type=str, help="the image repo (i.e., rclancy/anserini-test)")
    parser.add_argument("--tag", required=True, type=str, help="the image tag (i.e., latest)")
    parser.add_argument("--save_id", default="ultimo", type=str, help="the ID of the saved image (to search from)")
    parser.add_argument("--collection_path", required=True, help="the name of the collection")
    parser.add_argument("--test_split", required=True, type=str, help="test_split")
    parser.add_argument("--val_split", required=True, type=str, help="val_split")
    parser.add_argument("--stopwords", required=True, type=str, help="stopwords")
    parser.add_argument("--qrels", required=True, type=str, help="qrels")
    parser.add_argument("--topics", required=True, type=str, help="topics")

    parser.add_argument("--output_folder", required=True, type=str, help="output folder")

    args = parser.parse_args()

    print("Running...")

    # Mapping from collection name to path on host
    # path_host = args.collection_path

    # Mapping from collection name to path in container
    volumes = {}
    # mount volumes
    volumes[args.collection_path] = {"bind": "/collection", "mode": "rw"}
    volumes[args.test_split] = {"bind": "/test_split.txt", "mode": "ro"}
    volumes[args.val_split] = {"bind": "/val_split.txt", "mode": "ro"}
    volumes[args.stopwords] = {"bind": "/stopwords.txt", "mode": "ro"}
    volumes[args.qrels] = {"bind": "/qrels.txt", "mode": "ro"}
    volumes[args.topics] = {"bind": "/topics.txt", "mode": "ro"}
    volumes[args.output_folder] = {"bind": "/output", "mode": "rw"}

    # The first step is to pull an image from an OSIRRC participant,
    # start up a container, run its `init` and `index` hooks, and then
    # use `docker commit` to save the image after the index has been
    # built. The rationale for doing this is that indexing may take a
    # while, but only needs to be done once, so in essence we are
    # "snapshotting" the system with the indexes.
    base = client.containers.run("{}:{}".format(args.repo, args.tag),
                                 command="sh -c 'sh launch /collection /test_split.txt /val_split.txt /stopwords.txt /qrels.txt /topics.txt /output'",
                                 volumes=volumes, detach=True)

    # print("Waiting for stuff to finish...")
    # base.wait()
    for line in base.logs(stream=True):
        s = str(line.decode('utf-8'))
        print(s)

    print("Committing image...")
    base.commit(repository=args.repo, tag=generate_save_tag(args.tag, args.save_id))
