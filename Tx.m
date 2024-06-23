clc;
clear all;
close all;
set(0, 'defaultAxesXGrid','on', 'defaultAxesYGrid', 'on') %打开网格
%% 基本参数
Mod = 4; % MPSK模式
bitnum_per = log2(Mod);
fs = 48000;                                                            % 采样频率
fl = 10e3;                                                             % (LFM)下限频率
B = 4e3;                                                               % 通信带宽
fh = fl+B;                                                              % (LFM)上限频率
f0 = (fl + fh) / 2;                                                        % 中心频率（单频带传输）==12KHz
Rb = 2000;                                                             % 符号率
N_up = fs / Rb;                                                        % 升采样点数
N_bit = 3000;                                                          % 发送的比特数
N_BS = N_bit/bitnum_per;                                                  % 发送的符号数--QPSK
length_BS = N_BS * N_up*2;                                             % 使用卷积码需要乘2
N_bit = N_BS * bitnum_per;                                             % 需要生成的比特数
alpha = 1;                                                              % 滚降系数
N_filter = 512;                                                        % 滤波器阶数
% PulseShape = rcosfir(alpha, [ ], N_up, 1, 'sqrt');  % 脉冲成型滤波器（低通滤波器）
PulseShape = rcosdesign(alpha,1, N_up, 'sqrt');
b1 = fir1(N_filter, 2 * [fl fh] / fs);                               % 带通滤波器
%% --------------------发射机部分------------------------
%% 数据信号产生及编码
load information.mat
bit_generate = information(1 : N_bit);
%% 直接加扰?(scramble)
rng(1); % 种子
random_bits = randi([0, 1], 1, N_bit);
% 对bit_generate和随机数进行异或操作
scrambled_bits = xor(bit_generate, random_bits);
%% 卷积码编码
P = 0.1;
rate = 1/2;
%卷积码的生成多项式
if rate == 1
    tre1 = poly2trellis(7,[131]);
elseif rate == 1/2
    tre1 = poly2trellis(7,[133 171]);   
end
msg1 = convenc(scrambled_bits,tre1); %卷积编码
%% 进行交织
cols = length(msg1)/5;rows =  5;
interleaved_data = matintrlv(msg1, rows, cols); %解交织使用函数--matdeintrlv
%% ------------利用信道预测结果的自适应调制------------
% 准备输入数据
data = load('./inpulse_informer_24/data.mat');
channel_states = data.pred_data(:); % 转换为列向量
snrs = ones(length(channel_states), 1) * 5; % 设定相同的信噪比
buffer_state = 50; % 示例缓冲状态
arrival_rate = 10; % 示例数据到达率
max_cross_time = 100; % 最大交叉变异次数
A_s = 10; % 示例数据传输需求
% 设置 Python 环境
pyversion('E:\Anaconda\Data\envs\myenv\python.exe');
% 确保MATLAB找到Python路径
if count(py.sys.path, '') == 0
    insert(py.sys.path, int32(0), '');
end
% 导入Python模块
mod = py.importlib.import_module('genetic_algorithm_main');
py.importlib.reload(mod);
% 调用Python函数并处理结果
try
    % 将数据转换为Python可接受的类型
    py_channel_states = py.numpy.array(channel_states);
    py_snrs = py.numpy.array(snrs);
    py_buffer_state = int32(buffer_state);
    py_arrival_rate = int32(arrival_rate);
    py_max_cross_time = int32(max_cross_time);
    py_A_s = int32(A_s);
    % 调用Python函数
    result = mod.genetic_algorithm_main('inpulse_informer_24',py_snrs ,py_buffer_state, py_arrival_rate, py_max_cross_time, py_A_s);
    % 将结果转换为MATLAB类型
    best_modulation_schemes = cell(result);
    % 在MATLAB中使用结果
    disp('最佳调制和编码模式:');
    disp(best_modulation_schemes);
% 根据best_modulation_schemes选择Mod，以进行自适应调制
    switch char(best_modulation_schemes{1})
        case 'BPSK'
            Mod = 2;
        case 'QPSK'
            Mod = 4;
        case '16QAM'
            Mod = 16;
        case '64QAM'
            Mod = 64;
        otherwise
            error('Unknown modulation scheme');
    end
    % best_modulation_schemes的长度为len(channel_states)/24-预测点数
    disp(['Selected Modulation Scheme: ', char(best_modulation_schemes{1})]);
    bitnum_per = log2(Mod);
    % 更新调制模式相关参数
    N_BS = N_bit / bitnum_per;
    length_BS = N_BS * N_up * 2;
    N_bit = N_BS * bitnum_per;
catch ME
    disp('调用Python函数时出错:');
    disp(ME.message);
end

%% MPSK映射
[SymbolIn, Table] = Mapping(interleaved_data, Mod);
%% 升采样脉冲成型--对信息进行调制
signal_IQ = IQmodulate(PulseShape, N_up, SymbolIn, f0, fs);
signal_IQ = signal_IQ ./ max(abs(signal_IQ));
%% LFM信号参数设计
T_syn = 0.1;K = B / T_syn;  % LFM信号参数，B带宽，T脉宽，K调频斜率
t = 0 : 1/fs : T_syn-1/fs;
signal_measure = cos(2*pi*fl*t + pi*K*t.^2);                  
length_measure = T_syn * fs;
length_GI = 0.1 * fs;                                                %保护间隔
signal_GI = zeros(1, length_GI);
%% 发送信号构成
signal_send = [signal_measure signal_GI signal_IQ signal_GI signal_measure signal_GI];    %信号结构[测量信号 保护间隔 调制信号 保护间隔 测量信号 保护间隔]
m = -0;SNR = 15;
pass_by = Pass_Channel(fs,m,b1,length_measure,length_GI,length_BS,SNR,Rb,signal_send);

%% --------------------接收机部分------------------------
%% 时间同步
signal_receive = pass_by;
% 对signal_receive进行时间同步---要看效果的话可以先启动Pass_Channel中为信道加时延的代码
% Res_xcorr = corr_fun(signal_receive, signal_measure);
% delay_vec = 1:length(signal_receive)+length(signal_measure)-1;
% [~, index]=max(Res_xcorr(1 : length_measure+length(signal_GI)));
% est_delay=delay_vec(index)-length_measure; % 同步的位置，从此开始
% figure;plot(delay_vec(est_delay+1:end),abs(Res_xcorr(est_delay+1:end)));%接收端与发送端的互相关
% grid on;
% set(gca, 'FontSize',12,'FontWeight','bold')
% xlabel('时间索引n')
% title(['时间同步：估计的时延=' num2str(est_delay)]);
% signal_receive = signal_receive(est_delay+1:end);
% figure;plot((1:length(signal_receive))/fs,signal_receive);
% xlabel('时间 (s)');
% ylabel('幅度');
% title('同步后信号时域波形');
%% 带通滤波
signal_bandpass = filter(b1, 1, [signal_receive zeros(1,fix(length(b1)/2))]);
signal_rec_pass = signal_bandpass(fix(length(b1)/2)+1:end);

%% 多普勒测量
Res_xcorr = corr_fun(signal_rec_pass, signal_measure);
[~, pos1] = max(Res_xcorr(1 : length_measure+length(signal_GI)));             
[~, pos2] = max(Res_xcorr(length_GI+length_BS+1 : end));      
pos2 = pos2 + length_GI + length_BS;                                      
% 计算接收信号首尾LFM间隔，与发射间隔做对比
del_rec = pos2 - pos1;
del_send = length_measure + 2*length_GI + length_BS ;
% 利用间隔变化做多普勒测量
dup_det = (del_send - del_rec) / del_send;
fprintf(['多普勒因子测量值：' num2str(dup_det) '\n']);
% 利用重采样进行多普勒补偿
fs2 = fs*(1-dup_det);
fs2 = round(fs2 / factor_resample(fs)) * factor_resample(fs);    %使其满足resample精度要求防止报错
signal_rec_dc = resample(signal_rec_pass, fs, fs2);   %dc：Doppler compensation
% 提取信息符号
signal_rec_nodc_information = signal_rec_pass(length_measure+length_GI+1 : length_measure+length_GI+length_BS);
signal_rec_dc_information = signal_rec_dc(length_measure+length_GI+1 : length_measure+length_GI+length_BS);
signal_rec_origin_information = signal_send(length_measure+length_GI+1 : length_measure+length_GI+length_BS);
%% 信道均衡
Need_len =  length(signal_rec_dc_information);
% signal_rec_dc_information = LTE_LMS_fun1(25,0.05,Need_len/2,Need_len/2,signal_rec_dc_information,signal_rec_origin_information);
%% 相干解调--IQ解调+下载波
[symbol_demodulate_nodc] = IQdemodulate(signal_rec_nodc_information, fs, length_BS, f0, PulseShape, N_up);
[symbol_demodulate_dc] = IQdemodulate(signal_rec_dc_information, fs, length_BS, f0, PulseShape, N_up);
%% 抽样判决
for j = 1 : length(symbol_demodulate_nodc)
    Distance_all = abs(symbol_demodulate_nodc(j) - Table);
    Tablemin=find(Distance_all == min(Distance_all));
    symbol_decision_nodc(j) = Table(Tablemin(1));
end
for j = 1 : length(symbol_demodulate_dc)
    Distance_all = abs(symbol_demodulate_dc(j) - Table);
    Tablemin=find(Distance_all == min(Distance_all));
    symbol_decision_dc(j) = Table(Tablemin(1));
end

%% 解映射
bit_nodc  = Demapping(symbol_decision_nodc , Table , Mod);
bit_dc  = Demapping(symbol_decision_dc , Table , Mod);
%% 解交织
deinterleaved_bit_nodc = matdeintrlv(bit_nodc, rows, cols);
deinterleaved_bit_dc = matdeintrlv(bit_dc, rows, cols);
%% 译码
output_bit_nodc = vitdec(deinterleaved_bit_nodc, tre1, 7, 'trunc', 'hard'); % 卷积码
output_bit_dc = vitdec(deinterleaved_bit_dc, tre1, 7, 'trunc', 'hard'); % 卷积码
%% 解扰
final_bit_nodc = xor(output_bit_nodc, random_bits);
final_bit_dc = xor(output_bit_dc, random_bits);
%% 计算误码
BER_nodc = length(find(final_bit_nodc ~= bit_generate)) ./ N_bit;
BER_dc = length(find(final_bit_dc ~= bit_generate)) ./ N_bit;
fprintf(['多普勒不补偿且信道不均衡误码率：'  num2str(BER_nodc) '\n'] );
fprintf(['多普勒补偿且信道均衡误码率：' num2str(BER_dc) '\n']);

scatterplot(symbol_demodulate_nodc);
title('未进行多普勒补偿且未进行信道均衡前')
scatterplot(symbol_demodulate_dc);
title('采用多普勒补偿和信道均衡后')