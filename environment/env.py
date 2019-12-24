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

        self.obs_space = None
        self.action_space = None

        self.done = None

    def reset(self):
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
        self.idle_pool = [self.request[i].precursor for i in range(self.n_user)]
        self.curr_task = [self.idle_pool[i][0] for i in range(self.n_user)]
        self.curr_index = [0 for _ in range(self.n_user)]

        self.obs_space = [[0 for _ in range(self.request[i].task_num)] for i in range(self.n_user)]

        self.action_space = [self.cnt_bs(i) for i in range(self.n_user)]

        self.done = [False for _ in range(self.n_user)]

        return self.obs_space

    def set_action(self):
        for i in range(self.n_user):
            if not self.done[i]:
                self.obs_space[i][self.curr_task[i]] = 1
                for j, k in enumerate(self.request[i].structure[self.curr_task[i]]):
                    if k is 1:
                        self.idle_pool[i].append(j)
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
            path = data_path + SERVER[server_type]
            performance_data = CSVProcessor(path, 1).get_raw_data()
            performance_data = generate_data(performance_data)
            performance_data = performance_data[:, task_type]
            rwd.append()
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
            if math.sqrt((user_loc[0] - bs_loc[0]) ** 2 + (user_loc[1] - bs_loc[1] ** 2)) < self.bss[i].radius:
                bss.append(i)
        return bss


if __name__ == '__main__':
    pass
