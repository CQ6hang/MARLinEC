import math
from data.csv_processor import CSVProcessor
from data.plotter import generate_data
from entity.user_request import UserRequest

bs_path = '../data/heqiang-dataset/edge-servers/site-optus-melbCBD.csv'
user_path = '../data/heqiang-dataset/users/users-melbcbd-generated.csv'
data_path = '../data/server-performance-dataset/'
SERVER = ['AMAZON/amazon.csv', 'HUAWEI/huawei.csv', 'TENCENT/tencent.csv']


class Env:
    def __init__(self):
        self.n_bs = None
        self.bss = None

        self.n_user = None
        self.users = None

        self.n_request = self.n_user
        self.request = None
        self.idle_pool = None
        self.curr_task = None
        self.curr_index = None
        self.released = None

        self.obs_space = None
        self.action_space = None

        self.done = None

        self.global_constraint = None

        self.amz_data = generate_data(CSVProcessor(data_path + SERVER[0], 1).get_raw_data())
        self.hw_data = generate_data(CSVProcessor(data_path + SERVER[1], 1).get_raw_data())
        self.tec_data = generate_data(CSVProcessor(data_path + SERVER[2], 1).get_raw_data())

        self.create()

    def create(self):
        # generate base station entities
        bs_processor = CSVProcessor(bs_path, 0)
        self.bss = bs_processor.form_entity()
        self.n_bs = len(self.bss)

        # generate user entities
        user_processor = CSVProcessor(user_path, 1)
        self.users = user_processor.form_entity()
        self.n_user = len(self.users)

        # generate request
        self.request = [UserRequest(self.users[i].user_id) for i in range(self.n_user)]

        self.action_space = [self.cnt_bs(i) for i in range(self.n_user)]

        self.global_constraint = [0 for _ in range(self.n_user)]

    def reset(self):
        self.idle_pool = [self.request[i].precursor for i in range(self.n_user)]
        self.curr_task = [self.idle_pool[i][0] for i in range(self.n_user)]
        self.curr_index = [0 for _ in range(self.n_user)]
        self.released = [[] for _ in range(self.n_user)]

        self.obs_space = [[0 for _ in range(self.request[i].task_num)] for i in range(self.n_user)]

        self.done = [False for _ in range(self.n_user)]

        return self.obs_space

    def set_action(self):
        for i in range(self.n_user):
            if not self.done[i]:
                self.obs_space[i][self.curr_task[i]] = 1
                self.released[i].append(self.curr_task[i])

                child = []
                for j, k in enumerate(self.request[i].structure[self.curr_task[i]]):
                    if k == 1:
                        child.append(j)
                for j in child:
                    for k in range(self.request[i].task_num):
                        if self.request[i].structure[k][j] == 1 and k not in self.released[i]:
                            break
                        elif k == self.request[i].task_num - 1:
                            self.idle_pool.append(j)

                self.curr_task[i] = self.idle_pool[i][self.curr_index[i] + 1]
                # self.idle_pool[i] = self.idle_pool[i][1:]
                self.curr_index[i] += 1

    def observation(self):
        return self.obs_space

    def rewards(self, action):
        rwd = []
        for i in range(self.n_user):
            task_type = self.request[i].subtask[self.idle_pool[i][self.curr_index[i] - 1]].type
            server_type = self.bss[i][self.action_space[i][action[i]]].type
            layer = self.request[i].subtask[self.idle_pool[i][self.curr_index[i] - 1]].layers
            # performance_data = []
            # if server_type == 0:
            #     performance_data = self.amz_data
            # elif server_type == 1:
            #     performance_data = self.hw_data
            # elif server_type == 2:
            #     performance_data = self.tec_data
            # performance_data = generate_data(performance_data)
            # performance_data = performance_data[:, task_type]
            local_constraint = self.compute_local_constraint(i, self.global_constraint[i], layer)
            cum_prob = self.generate_pmf(local_constraint, server_type, task_type)

            rwd.append(cum_prob)
        return rwd

    def is_done(self):
        for i in self.n_user:
            if all(self.obs_space[i]):
                self.done[i] = True
        return self.done

    def step(self, action):
        self.set_action()
        obs = self.observation()
        reward = self.rewards(action)
        done = self.is_done()
        return obs, reward, done

    def cnt_bs(self, user):
        bss = []
        for i in range(self.n_bs):
            user_loc = self.users[user].location
            bs_loc = self.bss[i].location
            if math.sqrt((user_loc[0] - bs_loc[0]) ** 2 + (user_loc[1] - bs_loc[1]) ** 2) < self.bss[i].radius:
                bss.append(i)
        return bss

    def generate_pmf(self, x, flag1, flag2):
        data = []
        if flag1 == 0 and flag2 == 0:
            data = self.amz_data[:, 0]
        elif flag1 == 0 and flag2 == 1:
            data = self.amz_data[:, 1]
        elif flag1 == 0 and flag2 == 2:
            data = self.amz_data[:, 2]
        elif flag1 == 1 and flag2 == 0:
            data = self.hw_data[:, 0]
        elif flag1 == 1 and flag2 == 1:
            data = self.hw_data[:, 1]
        elif flag1 == 1 and flag2 == 2:
            data = self.hw_data[:, 2]
        elif flag1 == 2 and flag2 == 0:
            data = self.tec_data[:, 0]
        elif flag1 == 2 and flag2 == 1:
            data = self.tec_data[:, 1]
        elif flag1 == 2 and flag2 == 2:
            data = self.tec_data[:, 2]
        start, end = min(data), max(data)

        arg = [round(start - 0.1, 1)]
        while arg[-1] < round(end + 0.1, 1):
            arg.append(arg[-1] + 0.1)

        count = [0 for _ in range(len(arg))]
        for i in data:
            count[int((i - arg[0]) / 0.1)] += 1

        prob = []
        for i in count:
            prob.append(i / len(data))

        x = round(x, 1)
        if x in arg:
            return prob[arg.index(x)]
        else:
            return prob, arg[0]

    def compute_local_constraint(self, user, global_c, layer):
        request = self.request[user]
        qos = []
        for i in range(1, request.max_layer + 1):
            record = []
            for j in request.task_num:
                if j.layers == i:
                    record.append(self.average_qos(user, j.type))
                qos.append(max(record))
            return qos[layer] / sum(qos) * global_c

    def average_qos(self, user, task_type):
        qos = []
        bss = self.cnt_bs(user)
        server_type = []
        for bs in bss:
            if self.bss[bs].type not in server_type:
                server_type.append(self.bss[bs].type)
        for i in server_type:
            qos.append(self.mean_qos_pmf(i, task_type))
        return sum(qos) / len(qos)

    def mean_qos_pmf(self, server_type, task_type):
        prob, floor = self.generate_pmf(-100, server_type, task_type)
        # print(sum(prob), prob, floor)
        cum = 0
        ceiling = 0
        for i in prob:
            if cum + i <= 0.5:
                cum += i
                floor += 0.1
            else:
                ceiling = i
                break
        floor -= 0.1
        qos = 0.1 * (0.5 - cum) / ceiling + floor
        return qos


if __name__ == '__main__':
    env = Env()
    print(env.mean_qos_pmf(0, 0))
