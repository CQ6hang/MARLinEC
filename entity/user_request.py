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

        self.max_layer = self.layered()

    def show(self):
        print('user request info:\nuser:%d\nrequest type:%s\ntotal subtask number:%d\nmax layer:%d' % (
            self.user, self.r_type, self.task_num, self.max_layer))
        print('----------')
        for task in self.subTask:
            task.show()

    def layered(self):
        released = []
        parents = self.precursor
        layer = 1
        while len(released) < self.task_num:
            child = []
            new_parents = []
            for i in parents:
                self.subTask[i].layers = layer
                released.append(i)
                for j, k in enumerate(self.structure[i]):
                    if k == 1 and j not in child:
                        child.append(j)
            for j in child:
                for k in range(self.task_num):
                    if self.structure[k][j] == 1 and k not in released:
                        break
                    elif k == self.task_num - 1:
                        new_parents.append(j)
            parents = new_parents
            layer += 1

        return layer - 1


# test func


if __name__ == '__main__':
    request = UserRequest(0)
    print(request.precursor)
