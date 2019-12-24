import math
import matplotlib.pyplot as plt
import numpy as np

from data.csv_processor import CSVProcessor

bs_path = './heqiang-dataset/edge-servers/site-optus-melbCBD.csv'
user_path = './heqiang-dataset/users/users-melbcbd-generated.csv'
amazon_path = './server-performance-dataset/AMAZON/amazon.csv'
huawei_path = './server-performance-dataset/HUAWEI/huawei.csv'
tencent_path = './server-performance-dataset/TENCENT/tencent.csv'

# measured from google map,unit(m)
horizontal_axis = 1860
vertical_axis = 945

# the fixed points chosen from Melbourne CBD area
fixed_point = [[-37.821091, 144.955074], [-37.813194, 144.951433], [-37.815319, 144.974872]]

h_d = math.sqrt((-37.821091 + 37.815319) ** 2 + (144.955074 - 144.974872) ** 2)
v_d = math.sqrt((-37.821091 + 37.813194) ** 2 + (144.955074 - 144.951433) ** 2)
h_line = np.array([-37.815319 + 37.821091, 144.974872 - 144.955074])
v_line = np.array([-37.813194 + 37.821091, 144.951433 - 144.955074])

bs_data = CSVProcessor(bs_path, 0).get_raw_data()
user_data = CSVProcessor(user_path, 1).get_raw_data()


# bs_data = CSVProcessor(bs_path, 0).get_pruned_data()
# user_data = CSVProcessor(user_path, 1).get_pruned_data()


# the distance from the point to the line
def distance(point, line):
    po = np.array([point[0] + 37.821091, point[1] - 144.955074])
    return np.linalg.norm(np.cross(po, line) / np.linalg.norm(line))


# location transform
for i in range(len(bs_data)):
    x_axis = distance(bs_data[i], v_line) / h_d * horizontal_axis
    y_axis = distance(bs_data[i], h_line) / v_d * vertical_axis
    bs_data[i] = np.array([x_axis, y_axis])

for i in range(len(user_data)):
    x_axis = distance(user_data[i], v_line) / h_d * horizontal_axis
    y_axis = distance(user_data[i], h_line) / v_d * vertical_axis
    user_data[i] = np.array([x_axis, y_axis])


# expand server performance dataset
def generate_data(data):
    col1_min = np.min(data[:, 0])
    col1_max = np.max(data[:, 0])

    col2_min = np.min(data[:, 1])
    col2_max = np.max(data[:, 1])

    col3_min = np.min(data[:, 2])
    col3_max = np.max(data[:, 2])

    for _ in range(100):
        data1 = (col1_max - col1_min) * np.random.random_sample() + col1_min
        data2 = (col2_max - col2_min) * np.random.random_sample() + col2_min
        data3 = (col3_max - col3_min) * np.random.random_sample() + col3_min
        new_data = np.array([[data1, data2, data3]])
        data = np.append(data, new_data, 0)

    return data


# plot location distribution map
def draw_loc():
    plt.figure()
    # draw partial
    # plt.xlim(250, 750)
    # plt.ylim(0, 500)
    plt.xlim(0, 1860)
    plt.ylim(0, 945)
    # plt.plot(bs_data[:, 0], bs_data[:, 1], 'ro')
    plt.scatter(bs_data[:, 0], bs_data[:, 1], c='#CDCDCD', edgecolors='', marker='o', s=12000, alpha=0.1)
    plt.scatter(bs_data[:, 0], bs_data[:, 1], c='r', edgecolors='', marker='o', s=35)
    plt.scatter(bs_data[:, 0], bs_data[:, 1], c='', edgecolors='r', marker='o', s=12000, linewidths=0.3)
    plt.show()

    # plt.figure()
    # plt.plot(user_data[:, 0], user_data[:, 1], 'ro')
    # plt.show()

    plt.figure()
    plt.xlim(0, 1860)
    plt.ylim(0, 945)
    plt.xlabel('Physical Distance(m)')
    plt.ylabel('Physical Distance(m)')
    plt.title('The locations of base stations and users in Melbourne CBD area')
    plt.scatter(user_data[:, 0], user_data[:, 1], c='b', edgecolors='', marker='o', s=20)
    plt.scatter(bs_data[:, 0], bs_data[:, 1], c='#CDCDCD', edgecolors='', marker='o', s=12000, alpha=0.1)
    plt.scatter(bs_data[:, 0], bs_data[:, 1], c='r', edgecolors='', marker='o', s=35)
    plt.scatter(bs_data[:, 0], bs_data[:, 1], c='', edgecolors='r', marker='o', s=12000, linewidths=0.3)
    plt.show()


# plot server performance fluctuation
def draw_performance():
    amazon_performance_data = CSVProcessor(amazon_path, 1).get_raw_data()
    amazon_performance_data = generate_data(amazon_performance_data)
    # print(amazon_performance_data)
    plt.figure()
    # plt.xlim(0, 21)
    plt.title('computing 8/16/32 million digits of PI on Amazon ES')
    plt.ylabel('Execution time(s)')
    plt.xlabel('Time slot')
    plt.plot([p for p in range(len(amazon_performance_data[:, 0]))], amazon_performance_data[:, 0], '.-')
    plt.plot([p for p in range(len(amazon_performance_data[:, 1]))], amazon_performance_data[:, 1], '.-')
    plt.plot([p for p in range(len(amazon_performance_data[:, 2]))], amazon_performance_data[:, 2], '.-')
    plt.legend(('8M', '16M', '32M'))
    plt.show()

    huawei_performance_data = CSVProcessor(huawei_path, 1).get_raw_data()
    huawei_performance_data = generate_data(huawei_performance_data)
    # print(amazon_performance_data)
    plt.figure()
    # plt.xlim(0, 21)
    plt.title('computing 8/16/32 million digits of PI on HuaWei ES')
    plt.ylabel('Execution time(s)')
    plt.xlabel('Time slot')
    plt.plot([p for p in range(len(huawei_performance_data[:, 0]))], huawei_performance_data[:, 0], '.-')
    plt.plot([p for p in range(len(huawei_performance_data[:, 1]))], huawei_performance_data[:, 1], '.-')
    plt.plot([p for p in range(len(huawei_performance_data[:, 2]))], huawei_performance_data[:, 2], '.-')
    plt.legend(('8M', '16M', '32M'))
    plt.show()

    tencent_performance_data = CSVProcessor(tencent_path, 1).get_raw_data()
    tencent_performance_data = generate_data(tencent_performance_data)
    # print(amazon_performance_data)
    plt.figure()
    # plt.xlim(0, 21)
    plt.title('computing 8/16/32 million digits of PI on Tencent ES')
    plt.ylabel('Execution time(s)')
    plt.xlabel('Time slot')
    plt.plot([p for p in range(len(tencent_performance_data[:, 0]))], tencent_performance_data[:, 0], '.-')
    plt.plot([p for p in range(len(tencent_performance_data[:, 1]))], tencent_performance_data[:, 1], '.-')
    plt.plot([p for p in range(len(tencent_performance_data[:, 2]))], tencent_performance_data[:, 2], '.-')
    plt.legend(('8M', '16M', '32M'))
    plt.show()


# draw_loc()

draw_performance()
