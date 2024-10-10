datao = audioread("dsp2_org.wav");
Fs = 8000;
%sound(datao,Fs);
datan = audioread("dsp2_noise.wav");
%sound(datan,Fs);
N=2048;
Xok = fft(datao,N);
ampo = abs(Xok);
Xnk = fft(datan,N);
ampn = abs(Xnk);
%---------------------
Ts = 1/N;
k=0:N-1;
omega = 2 * pi * k /N;
f = omega / 2/ pi;
Fs = 1/Ts;
Omega = omega * Fs;
F = Omega /2/pi;
%-----------------
figure(1);
subplot(2,1,1);
stem(F, ampo);
title("original");
xlabel("周波数F[Hz]");
ylabel("振幅スペクトル|X[k]|");
subplot(2,1,2);
stem(F,ampn);
title("noise");
xlabel("周波数F[Hz]");
ylabel("振幅スペクトル|X[k]|");

load('bs2048.mat');


yn3 = conv(datan,h_bspf,"same");
%-----------------------
%sound(yn1 ,Fs);
%play(my1);
Fs = 8000;
sound(yn3 ,Fs);
%play(my2);
%-----------------------
Xyk = fft(yn3,N);
ampy = abs(Xyk);
figure(2);
subplot(2,1,1);
stem(F, ampo);
title("original");
xlabel("周波数F[Hz]");
ylabel("振幅スペクトル|X[k]|");
subplot(2,1,2);
stem(F,ampy);
title("noise 除去");
xlabel("周波数F[Hz]");
ylabel("振幅スペクトル|X[k]|");