import random
from data.XMLProcess import XMLProcess

path = '../data/workflow-dataset/'
test = ['CyberShake_30', 'Epigenomics_24', 'Inspiral_30', 'Montage_25', 'Sipht_29']


class UserRequest:
    def __init__(self, user_id):
        self.user = user_id
        self.r_type = test[random.randint(0, 4)]

        processor = XMLProcess(path + self.r_type + '.xml')

        self.task_num = processor.task_num
        self.subTask = processor.tasks
        self.structure = processor.dag
        self.precursor = processor.get_precursor()

    def show(self):
        print('user request info:\nuser:%d\nrequest type:%s\ntotal subtask number:%d' % (
            self.user, self.r_type, self.task_num))
        print('----------')
        for task in self.subTask:
            task.show()


# test func
if __name__ == '__main__':
    workflow = [UserRequest(i) for i in range(5)]
    for w in workflow:
        w.show()
    # print(numpy.random.random((10,7)))
