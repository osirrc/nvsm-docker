class Trainer:

    def __init__(self, trainer_config=None):
        self.config = trainer_config

    def set_config(self, trainer_config):
        self.config = trainer_config
        params = {item.split('=')[0]: item.split('=')[1] for item in trainer_config.opts[0].split('+')}
        self.test_split = params['test_split']
        self.val_split = params['val_split']
        self.qrels = params['qrels']
        self.topics = params['topics']
        self.output_folder_host = params['output_folder_host']

    def train(self, client, output_folder_host, generate_save_tag):
        print("Training...")
        volumes = {}
        volumes[self.test_split] = {"bind": "/test_split.txt", "mode": "ro"}
        volumes[self.val_split] = {"bind": "/val_split.txt", "mode": "ro"}
        volumes[self.qrels] = {"bind": "/qrels.txt", "mode": "ro"}
        volumes[self.topics] = {"bind": "/topics.txt", "mode": "ro"}
        volumes[self.output_folder_host] = {"bind": "/output", "mode": "rw"}
        container = client.containers.run("{}:{}".format(self.config.repo, self.config.tag),
                                          command="sh -c 'sh train /topics.txt /test_split.txt /val_split.txt /qrels.txt /output'",
                                          detach=True, volumes=volumes)

        print("Logs for init and index in container '{}'...".format(container.name))
        for line in container.logs(stream=True):
            print(str(line.decode('utf-8')), end="")

        print("Committing image...")
        container.commit(repository=self.config.repo, tag=generate_save_tag(self.config.tag, self.config.save_id))
