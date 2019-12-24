import random


class SubTask:
    def __init__(self, id, namespace, name, version, runtime, input, output):
        self.id = id
        self.namespace = namespace
        self.name = name
        self.version = version
        self.runtime = runtime
        self.input = input
        self.output = output

        self.type = random.randint(0, 2)

    def show(self):
        print('task info:\nid:%d\ntype:%d\nnamespace:%s\nname:%s\nversion:%s\nruntime:%s\ninput:%s\noutput:%s' % (
            self.id, self.type, self.namespace, self.name, self.version, self.runtime, self.input, self.output))
        print('----------')
