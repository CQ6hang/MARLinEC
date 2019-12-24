import xml.dom.minidom

import numpy as np

from entity.subtask import SubTask

path = './workflow-dataset/'


class XMLProcess:
    def __init__(self, file):
        self.xmlFile = file

        self.task_num = 0
        self.tasks = None
        self.dag = None

        self.__get_task()
        self.__get_dag()

    def __get_task(self):
        domtree = xml.dom.minidom.parse(self.xmlFile)
        collection = domtree.documentElement
        jobs = collection.getElementsByTagName('job')

        num = 0
        tasks = []
        for job in jobs:
            job_id = int(job.getAttribute('id')[2:])
            if job_id > num:
                num = job_id

            files = job.getElementsByTagName('uses')
            input = []
            output = []
            for file in files:
                file_name = file.getAttribute('file')
                file_type = file.getAttribute('link')
                if file_type == 'input':
                    input.append(file_name)
                else:
                    output.append(file_name)

            tasks.append(
                SubTask(job_id, job.getAttribute('namespace'), job.getAttribute('name'),
                        job.getAttribute('version'), float(job.getAttribute('runtime')), input, output))

        self.task_num = num + 1
        self.tasks = tasks

    def __get_dag(self):
        domtree = xml.dom.minidom.parse(self.xmlFile)
        collection = domtree.documentElement
        childrens = collection.getElementsByTagName("child")

        self.dag = np.zeros((self.task_num, self.task_num), dtype=int)
        for child in childrens:
            child_id = child.getAttribute('ref')
            child_id = int(child_id[2:])
            parents = child.getElementsByTagName('parent')
            for parent in parents:
                parent_id = parent.getAttribute('ref')
                parent_id = int(parent_id[2:])
                self.dag[parent_id, child_id] = 1

    def get_precursor(self):
        precursor = []
        for i in range(self.task_num):
            temp = self.dag[:, i]
            if np.sum(temp) == 0:
                precursor.append(i)
        return precursor

    def print_dag(self):
        print(self.dag)


# test func
if __name__ == '__main__':
    temps = [XMLProcess(path + 'Sipht_29.xml'), XMLProcess(path + 'Montage_25.xml'),
             XMLProcess(path + 'Inspiral_30.xml'),
             XMLProcess(path + 'Epigenomics_24.xml'), XMLProcess(path + 'CyberShake_30.xml')]
    for graph in temps:
        graph.print_dag()
        print(graph.get_precursor())
