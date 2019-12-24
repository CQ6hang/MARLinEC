import numpy as np

from entity.base_station import BS
from entity.user import User

bs_path = './heqiang-dataset/edge-servers/site-optus-melbCBD.csv'
user_path = './heqiang-dataset/users/users-melbcbd-generated.csv'


# extract the location information of users and base stations
class CSVProcessor:
    def __init__(self, path, flag):
        self.__data = None
        self.__flag = flag
        with open(path, encoding='utf-8') as f:
            if self.__flag == 0:  # extract base stations csv files
                self.__data = np.loadtxt(f, delimiter=',', skiprows=1, usecols=(1, 2))
            elif self.__flag == 1:  # extract users csv files
                self.__data = np.loadtxt(f, delimiter=',', skiprows=1)

    def get_raw_data(self):
        return self.__data

    def get_pruned_data(self):
        return self.__data

    def form_entity(self):
        entities = []
        for i, j in enumerate(self.__data):
            if self.__flag == 0:  # get base station entities
                entities.append(BS(i, [float(j[0]), float(j[1])], 200))
            elif self.__flag == 1:  # get user entities
                entities.append(User(i, [float(j[0]), float(j[1])]))
        return entities


if __name__ == '__main__':
    bs_processor = CSVProcessor(bs_path, 0)
    bss = bs_processor.form_entity()
    # print(bs_processor.get_raw_data())
    for bs in bss:
        bs.print()

    user_processor = CSVProcessor(user_path, 1)
    users = user_processor.form_entity()
    # print(user_processor.get_raw_data())
    for user in users:
        user.print()
