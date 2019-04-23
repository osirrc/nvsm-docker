class Searcher:

    def __init__(self, searcher_config=None):
        self.config = searcher_config

    def set_config(self, searcher_config):
        self.config = searcher_config
        params = {item.split('=')[0]: item.split('=')[1] for item in searcher_config.opts[0].split('+')}
        self.test_split = params['test_split']
        self.qrels = params['qrels']
        self.topics = params['topics']
        self.output_folder_host = params['output_folder_host']

    def search(self, client, output_folder_host, generate_save_tag):
        print("Searching...")
        volumes = {}
        volumes[self.test_split] = {"bind": "/test_split.txt", "mode": "ro"}
        volumes[self.qrels] = {"bind": "/qrels.txt", "mode": "ro"}
        volumes[self.topics] = {"bind": "/topics.txt", "mode": "ro"}
        volumes[self.output_folder_host] = {"bind": "/output", "mode": "rw"}
        # --queries_file $1 --qrels_file $2 --splits_test $3 --output_folder $4
        container = client.containers.run("{}:{}".format(self.config.repo, self.config.tag),
                                          command="sh -c 'sh search /topics.txt /qrels.txt /test_split.txt /output'",
                                          detach=True, volumes=volumes)

        print("Logs for init and index in container '{}'...".format(container.name))
        for line in container.logs(stream=True):
            print(str(line.decode('utf-8')), end="")

        print("Committing image...")
        container.commit(repository=self.config.repo, tag=generate_save_tag(self.config.tag, self.config.save_id))
