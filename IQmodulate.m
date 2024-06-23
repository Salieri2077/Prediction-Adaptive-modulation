function [signal] = IQmodulate(p,N_up,SymbolIn,f0,fs)

%%%% p:脉冲成型滤波器系数；N_pn：升采样点数；SymbolIn：映射符号（复数）；f0：载波频率；fs：采样频率；

%%%% IQ两路信号分别升采样，分别过脉冲成型滤波器
%%%% pams_I和pams_Q分别是是两路信号升采样后的信号
%%%% signal调制后信号
    %% 上采样
    pams_I = upsample(real(SymbolIn), N_up);
    pams_Q = upsample(imag(SymbolIn), N_up);
    %% 画图说明脉冲成型用
    %     p = 1; % 若不进行脉冲成型p=1,进行的则将其注释
    %     pams_I = real(SymbolIn);pams_Q = imag(SymbolIn);
    %     fz = ones(1,N_up);
    %     
    %     copy = pams_I(fz,:); % 将msgl复制相同的N_up行
    %     pams_I = reshape(copy,1,N_up*length(pams_I));
    %     copy = pams_Q(fz,:); % 将msgl复制相同的N_up行
    %     pams_Q = reshape(copy,1,N_up*length(pams_Q));
    %% 整形滤波器%
    pams_I = [pams_I zeros(1, fix(length(p)/2))];
    pams_Q = [pams_Q zeros(1, fix(length(p)/2))];
    ynI = filter(p, 1, pams_I);                                      %脉冲成形滤波
    ynQ = filter(p, 1, pams_Q);                                   %脉冲成形滤波
    ynI = ynI(fix(length(p)/2)+1 : end);
    ynQ = ynQ(fix(length(p)/2)+1 : end);
%     figure;plot(ynI);title('脉冲成型后');
    %% 调制
    t = 0 : 1/fs : length(ynI)/fs-1/fs;
    y_cos = cos(2*pi*f0*t);
    y_sin = sin(2*pi*f0*t);
    signal = ynI.*y_cos + ynQ.*y_sin;                          %%sending signal
%     figure;plot(signal);title('调制信号');
    %% 画图说明脉冲成型用
%     % 计算信号的FFT
%     fft_result = fftshift(fft(signal));
%     % 计算频率轴
%     frequencies = (-length(fft_result)/2 : length(fft_result)/2 - 1) * fs / length(fft_result);
%     % 取FFT结果的幅度谱
%     amplitude_spectrum = abs(fft_result);
%     % 画频谱图
%     figure;
%     plot(frequencies, amplitude_spectrum);
%     title('频谱图');
%     xlabel('频率 (Hz)');
%     ylabel('幅度');
%     grid on;
end



