function sym_lte_lms=LTE_LMS_fun1(L,miu,len_RxWaveSym2,len_RxWaveSym1,RxWave,TxWave1)
    wn=zeros(2*L,1);
    Xn=zeros(1,len_RxWaveSym1);%输入序列
    en=zeros(1,len_RxWaveSym1);%误差序列
    RxWave = RxWave';
    TxWave1 = TxWave1';
    %% 训练过程
    for i=L+1:len_RxWaveSym1
        xn=RxWave(L+i-1:-1:i-L);
        Xn(i)=wn'*xn;
        XXn=xn;
        en(i)=TxWave1(i)-Xn(i);
        wn=wn+miu*en(i)'*XXn;%LMS算法更新公式
    end
    %% 均衡过程
    RxWave_bu = [RxWave;zeros(L,1)];
    for i=len_RxWaveSym1+1:len_RxWaveSym1+len_RxWaveSym2
        xn=RxWave_bu(L+i-1:-1:i-L);
        Xn(i)=wn'*xn;
    end
    sym_lte_lms=Xn;
end
