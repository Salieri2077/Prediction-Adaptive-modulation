function [signal] = IQmodulate(p,N_up,SymbolIn,f0,fs)

%%%% p:��������˲���ϵ����N_pn��������������SymbolIn��ӳ����ţ���������f0���ز�Ƶ�ʣ�fs������Ƶ�ʣ�

%%%% IQ��·�źŷֱ����������ֱ����������˲���
%%%% pams_I��pams_Q�ֱ�������·�ź�����������ź�
%%%% signal���ƺ��ź�
    %% �ϲ���
    pams_I = upsample(real(SymbolIn), N_up);
    pams_Q = upsample(imag(SymbolIn), N_up);
    %% ��ͼ˵�����������
    %     p = 1; % ���������������p=1,���е�����ע��
    %     pams_I = real(SymbolIn);pams_Q = imag(SymbolIn);
    %     fz = ones(1,N_up);
    %     
    %     copy = pams_I(fz,:); % ��msgl������ͬ��N_up��
    %     pams_I = reshape(copy,1,N_up*length(pams_I));
    %     copy = pams_Q(fz,:); % ��msgl������ͬ��N_up��
    %     pams_Q = reshape(copy,1,N_up*length(pams_Q));
    %% �����˲���%
    pams_I = [pams_I zeros(1, fix(length(p)/2))];
    pams_Q = [pams_Q zeros(1, fix(length(p)/2))];
    ynI = filter(p, 1, pams_I);                                      %��������˲�
    ynQ = filter(p, 1, pams_Q);                                   %��������˲�
    ynI = ynI(fix(length(p)/2)+1 : end);
    ynQ = ynQ(fix(length(p)/2)+1 : end);
%     figure;plot(ynI);title('������ͺ�');
    %% ����
    t = 0 : 1/fs : length(ynI)/fs-1/fs;
    y_cos = cos(2*pi*f0*t);
    y_sin = sin(2*pi*f0*t);
    signal = ynI.*y_cos + ynQ.*y_sin;                          %%sending signal
%     figure;plot(signal);title('�����ź�');
    %% ��ͼ˵�����������
%     % �����źŵ�FFT
%     fft_result = fftshift(fft(signal));
%     % ����Ƶ����
%     frequencies = (-length(fft_result)/2 : length(fft_result)/2 - 1) * fs / length(fft_result);
%     % ȡFFT����ķ�����
%     amplitude_spectrum = abs(fft_result);
%     % ��Ƶ��ͼ
%     figure;
%     plot(frequencies, amplitude_spectrum);
%     title('Ƶ��ͼ');
%     xlabel('Ƶ�� (Hz)');
%     ylabel('����');
%     grid on;
end



