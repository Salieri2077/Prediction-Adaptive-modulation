import numpy as np
import scipy.io
import os
from collections import Counter
from matplotlib import pyplot as plt

# 预测数据读入
def load_data(file_name):
    hss = np.load('./'+file_name+'/metrics.npy')
    preds = np.load('./'+file_name+'/pred.npy')
    trues = np.load('./'+file_name+'/true.npy')
    return trues, preds

def generate_channel_state(trues, preds, model_name):
    num_tau = preds.shape[0]
    num_pred = preds.shape[1]
    origin_data = np.zeros((num_tau+num_pred, preds.shape[2]))
    pred_data = np.zeros((num_tau+num_pred, preds.shape[2]))

    for i in range(num_tau):
        if i == 0:
            origin_data[:num_pred,:] = trues[i,:num_pred,:]
            pred_data[:num_pred,:] = preds[i,:num_pred,:]
        else:
            origin_data[num_pred+i,:] = trues[i,-1,:]
            pred_data[num_pred+i,:] = preds[i,-1,:]
    time_duration = np.arange(0, len(pred_data.T)) * (128 / 2048)
    return origin_data,pred_data

def moving_average(data, window_size):
    data = np.array(data).flatten()  # 转换为 NumPy 数组
    smoothed_data = np.convolve(data, np.ones(window_size) / window_size, mode='valid')
    return smoothed_data

# 调制编码模式和信道状态对能量开销和数据传输率的影响函数
def energy_consumption(modulation, channel_state):
    energy_table = {'BPSK': 1, 'QPSK': 2, '16QAM': 3, '64QAM': 4}
    min_channel_state = 1e-2  # 设置一个最小值阈值，避免 channel_state 为 0
    effective_channel_state = max(channel_state, min_channel_state)
    return energy_table[modulation] / effective_channel_state

def data_rate(modulation, channel_state):
    rate_table = {'BPSK': 1, 'QPSK': 2, '16QAM': 4, '64QAM': 6}
    return rate_table[modulation] * channel_state

# 定义误包率函数
def packet_error_rate(modulation, snr):
    params = {
        'BPSK': (1, 1.0, 0.5),
        'QPSK': (1, 0.5, 0.25),
        '16QAM': (1, 0.1, 0.1),
        '64QAM': (1, 0.05, 0.05)
    }
    a_m, b_m, gamma_pm = params[modulation]
    if snr < gamma_pm:
        return 1.0
    else:
        return a_m * np.exp(-b_m * snr)

# 适应度函数
def fitness(chromosome, channel_states, snrs):
    total_energy = sum([energy_consumption(mc, h) for mc, h in zip(chromosome, channel_states)])
    total_rate = sum([data_rate(mc, h) * (1 - packet_error_rate(mc, snr)) for mc, h, snr in zip(chromosome, channel_states, snrs)])
    if total_rate >= A_s:
        return total_rate / total_energy
    else:
        return 0

# 选择操作
def select(population, fitnesses, K):
    selected_indices = np.argsort(fitnesses)[-K:]
    selected_population = [population[i] for i in selected_indices]
    # 确保 selected_population 的长度为偶数
    if len(selected_population) % 2 != 0:
        selected_population.append(selected_population[np.random.randint(0, len(selected_population))])
    return selected_population

# 交叉操作
def crossover(parent1, parent2):
    point = np.random.randint(1, len(parent1) - 1)
    return np.concatenate((parent1[:point], parent2[point:])), np.concatenate((parent2[:point], parent1[point:]))

# 变异操作
def mutate(chromosome, mutation_rate=0.01):
    modulation_schemes = ['BPSK', 'QPSK', '16QAM', '64QAM']
    for i in range(len(chromosome)):
        if np.random.rand() < mutation_rate:
            chromosome[i] = np.random.choice(modulation_schemes)
    return chromosome

# 主算法
def genetic_algorithm(channel_states, snrs, buffer_state, arrival_rate, max_cross_time, population_size=50, K=25):
    modulation_schemes = ['BPSK', 'QPSK', '16QAM', '64QAM']
    num_time_slots = len(channel_states)
    population = [np.random.choice(modulation_schemes, num_time_slots).tolist() for _ in range(population_size)]
    
    cross_time = 0
    
    while cross_time < max_cross_time:
        fitnesses = np.array([fitness(chrom, channel_states, snrs) for chrom in population])
        selected_population = select(population, fitnesses, K)
        next_generation = []
        for i in range(0, len(selected_population), 2):
            parent1, parent2 = selected_population[i], selected_population[i + 1]
            offspring1, offspring2 = crossover(parent1, parent2)
            next_generation.append(mutate(offspring1))
            next_generation.append(mutate(offspring2))
        population = next_generation
        cross_time += 1
    
    best_chromosome = max(population, key=lambda chrom: fitness(chrom, channel_states, snrs))
    return best_chromosome
# 示例输入
file_name = 'inpulse_informer_24'
trues, preds = load_data(file_name)
# channel_states是预测的信道数据
origin_data, channel_states = generate_channel_state(trues, preds, 'Informer-24')
output_dir = './' + file_name
# 构建保存路径
save_path = os.path.join(output_dir, 'data.mat')
# 保存数据为 .mat 文件
scipy.io.savemat(save_path, {'origin_data': origin_data, 'pred_data': channel_states})
print("Data saved to data.mat")


buffer_state = 50  # 缓存状态（示例值）
arrival_rate = 10  # 数据到达率（示例值）
max_cross_time = 100  # 设定的交叉变异轮数
A_s = 10  # 示例数据传输需求，可以根据实际情况调整

# 示例信噪比 (SNR) 数据
# snrs = np.random.uniform(0, 30, len(channel_states))  # 随机生成 SNR 数据
snrs = np.ones(len(channel_states)) * 5 # 规定相同的信噪比

best_modulation_schemes = []

# 每24个点确定一种调制方式
for i in range(0, len(channel_states), 24):
    window_channel_states = np.abs(channel_states[i:i+24])
    window_snrs = snrs[i:i+24]
    if len(window_channel_states) == 24:
        best_modulation_scheme = genetic_algorithm(window_channel_states, window_snrs, buffer_state, arrival_rate, max_cross_time)
        modulation_counter = Counter(best_modulation_scheme)
        most_common_modulation = modulation_counter.most_common(1)[0][0]
        best_modulation_schemes.append(most_common_modulation)

print("最佳调制和编码模式:", best_modulation_schemes)
