function [signal_add_dopper] = Pass_Channel(fs,m,b1,length_measure,length_GI,length_BS,SNR,Rb,signal_send)
%     signal_add_noise = BandNoiseAdd(signal_send, SNR, b1 ,length_measure+length_GI, length_measure+length_GI+length_BS);
%     signal_add_noise=signal_send;
%     signal_add_noise = FadingNoiseAdd(signal_send,SNR,fs,Rb);
    signal_add_noise = CIRAdd(signal_send,fs);
    
    %% 加多普勒
    % 多普勒估计精度
    % 受限于重采样函数的多普勒估计精度，多普勒设计值不满足这个倍数时，resample大概率报错
    dup_precision1 = factor_resample(fs) / fs; 
    % 受限于FFT的物理分辨率的多普勒精度，不满足这个倍数时，估计会存在误差
    dup_precision2 = 1 / (length_measure + 2*length_GI + length_BS);  
    % 求最小公倍数，同时满足上述两种精度设置                                           
    dup_precision = dup_precision1;

    % 多普勒添加
    % m = 50;                      % 估计精度的整倍数
    dup_ori= m * dup_precision;  
    fs1 = fs * (1 - dup_ori); 
    signal_add_dopper = resample(signal_add_noise, fs1, fs);
    fprintf(['多普勒因子实际值：'  num2str(dup_ori) '\n']);
    %% 加开头时延
%     signal_add_dopper = [zeros(1, 0.1*fs) signal_add_dopper];
    %% 画图
    pass_by = signal_add_dopper(1:length(signal_send)); % 裁剪到和原信号同样的长度
    s_t = (1:length(signal_send))*1/fs;
    figure;
    subplot(2,1,1);
    plot(s_t, signal_send);
    xlabel('时间 (s)');
    ylabel('幅度');
    title('原始信号时域波形');
    subplot(2,1,2);
    plot(s_t, pass_by);
    xlabel('时间 (s)');
    ylabel('幅度');
    title('接收信号时域波形（经过信道）');
end