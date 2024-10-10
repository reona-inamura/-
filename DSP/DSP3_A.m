
T0=1;
F0 = 1/ T0;
N = 200;
Ts = 1/N;
n = 0:N-1;

yn = zeros(N,1);
for c = 1:N/2
    yn(c,1) = 1;
end
figure(1);
subplot(1,2,1);
stem(n,yn);
xlabel("時刻t[sec]");
ylabel("振幅");
%----------4------------
N=200;
k=0:N-1;
omega = 2 * pi * k /N;
f = omega / 2/ pi;
Fs = 1/Ts;
Omega = omega * Fs;
F = Omega /2/pi;
%-----------3----------------
N=200;
Xk = fft(yn,N);
amp = abs(Xk);
phase = angle(fix(Xk));
ynre = ifft(Xk,N);
subplot(1,2,2);
stem(n* 1,ynre);
xlabel("時刻t[sec]");
ylabel("振幅");
ylim([0,1]);
%^---------------------------------
figure(2);
subplot(1,2,1);
stem(F, amp);
xlabel("周波数F[Hz]");
ylabel("振幅スペクトル|X[k]|");
title("そのまま");
subplot(1,2,2);
load('DSP3_AH.mat');
lp = conv(yn,h_lpf);
Xyk = fft(lp,N);
%Xyk = fft(yn,N);
%Xyk = conv(Xyk,h_lpf);
ampy = abs(Xyk);
stem(F, ampy);
xlabel("周波数F[Hz]");
ylabel("振幅スペクトル|Y[k]|");
%------------------------------
figure(3);
subplot(1,2,1);
stem(n,ynre);
xlabel("時刻t[sec]");
ylabel("振幅")
title("そのまま逆フーリエ変換して求めた信号");
ynrek = ifft(Xyk,N);
subplot(1,2,2);
stem(n* 1,ynrek);
xlabel("時刻t[sec]");
ylabel("振幅");
title("ローパスフィルタをかけて逆フーリエ変換して求めた信号");


