function [SignalAftFdn] = FadingNoiseAdd(Signal_in,SNR,fs,Rb)
    % 尝试加多途噪声
    baud = Rb;
    lenChannel = 6 * fs/baud;   % # of samples, channel delay spread range
    nMultiPass = 5; % , # of multipasses
    mpChannel = zeros(lenChannel, 1);
    phi = 2*pi*rand(nMultiPass, 1);
    rng(1); % 纪录随机数
    amp = rand(nMultiPass, 1);
    amp = sort(amp./sum(amp),'descend');
    tau = floor(rand(nMultiPass, 1)*lenChannel);
    tau = sort(tau - min(tau) + 1);
    for i = 1:size(tau,1)
        mpChannel(tau(i)) = amp(i)*exp(1j*phi(i)); 
    end
    % 自然信道一般是实信道
    mpChannel = abs(mpChannel);
    figure, plot(mpChannel); % 绘制单位脉冲响应

    % Fading channel y(t) = h(t)*x(t) + n(t)
%     realTxWave = Signal_in;
%     compTxWave = hilbert(realTxWave);  % 为什么使用hibert?
    compTxWave = Signal_in;
    channelSignal = conv(compTxWave, mpChannel);
    noise = normrnd(0,1/10^(SNR/20),length(channelSignal),1);
    SignalAftFdn = real(channelSignal) + noise';
    %     SignalAftFdn = channelSignal; %  不加噪声
end
