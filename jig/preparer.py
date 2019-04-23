class Preparer:

    def __init__(self, preparer_config=None):
        self.config = preparer_config

    def set_config(self, preparer_config):
        self.config = preparer_config
        params = {item.split('=')[0]: item.split('=')[1] for item in preparer_config.opts[0].split('+')}
        self.output_folder = params['output_folder']
        self.coll_folder = params['coll_folder']
        self.stopwords = params['stopwords']

    def prepare(self, client, collection_path_guest, generate_save_tag):
        print("Indexing...")
        volumes = {}
        volumes[self.coll_folder] = {"bind": "/collection", "mode": "ro"}
        volumes[self.output_folder] = {"bind": "/output", "mode": "rw"}
        volumes[self.stopwords] = {"bind": "/stopwords.txt", "mode": "ro"}
        container = client.containers.run("{}:{}".format(self.config.repo, self.config.tag),
                                          command="sh -c 'sh index /collection /output /stopwords.txt'",
                                          volumes=volumes, detach=True)

        print("Logs for init and index in container '{}'...".format(container.name))
        for line in container.logs(stream=True):
            print(str(line.decode('utf-8')), end="")

        print("Committing image...")
        container.commit(repository=self.config.repo, tag=generate_save_tag(self.config.tag, self.config.save_id))
