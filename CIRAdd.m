function [pass_by] = CIRAdd(signal_send,fs,num_point)
%% 通过设计的信道模型--Watermark
    % 时变信道冲激响应
%     data = load('BCH1.mat');
%     data = load('NOF1_001.mat');
%     channel_data = data.h; % 582x2044的矩阵
%     [channel_num , ~] = size(channel_data);

    data = load('./inpulse_informer_24/data.mat');
    channel_data = data.pred_data;
    [channel_num , ~] = size(channel_data);
    
    % 传递信号通过每一个信道，所有信道累加
    % pass_one = conv2(signal_send, channel_data, 'full');
    % pass_by = sum(pass_one); % 所有信道进行求和
    
    % 通过一条信道 预测的24个点的信道
    channel_data = channel_data(num_point : num_point+24);
    pass_one = conv2(signal_send, abs(channel_data(1,:)));
    pass_by = pass_one;
    
    pass_by = pass_by(1:length(signal_send)); % 裁剪到和原信号同样的长度
    s_t = (1:length(signal_send))*1/fs;
%     figure;
%     plot(s_t,pass_by);
end
