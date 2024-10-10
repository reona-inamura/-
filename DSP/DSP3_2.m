N6 = 6;
N32 = 32;
n6 = 0:N6-1;
n32=0:N32-1;
F0 = 4;
Ts6 = 1/N6;
Ts32 = 1/N32;
x32n=sin(2*pi*F0*Ts32*n32);
x6n=sin(2*pi*F0*Ts6*n6);

%----------------
k32=0:N32-1;
omega32 = 2 * pi * k32 /N32;
f32 = omega32 / 2/ pi;
Fs32 = 32;
Omega32 = omega32 * Fs32;
F32 = Omega32 /2/pi;

k6=0:N6-1;
omega6 = 2 * pi * k6 /N6;
f6 = omega6 / 2/ pi;
Fs6 = 6;
Omega6 = omega6 * Fs6;
F6 = Omega6 /2/pi;
%---------------
X32k = fft(x32n,N32);
amp32 = abs(X32k);
X6k = fft(x6n,N6);
amp6 = abs(X6k);
x32ifft = ifft(X32k,N32);
x6ifft = ifft(X6k,N6);
figure(1);
subplot(1,2,1);
stem(n32* Ts32,x32ifft);
%hold on;
%N=1000;
%na = 0:N-1;
%Tsa= 1/N;
%xt = sin(2*pi*F0*na*Tsa);
%plot(na*Tsa,xt);
%legend("x[n]","x(t)");
%hold off;
title("32点FFTを逆FFTした時間信号");
xlabel("時刻t[sec]");
ylabel("振幅");
subplot(1,2,2);
stem(n6* Ts6,x6ifft);
%hold on;
%N=1000;
%na = 0:N-1;
%Tsa= 1/N;
%xt = sin(2*pi*F0*na*Tsa);
%plot(na*Tsa,xt);
%legend("x[n]","x(t)");
%hold off;
title("6点FFTを逆FFTした時間信号")
xlabel("時刻t[sec]");
ylabel("振幅");
figure(2);
stem(F6, amp6);
title("6点FFTの振幅スペクトル");
xlabel("周波数F[Hz]");
ylabel("振幅スペクトル|X[k]|");
%stem(n6* Ts6,x6n);
%title("6点FFTの時間信号")
%xlabel("時刻t[sec]");
%ylabel("振幅")
%hold on;
%stem(n6* Ts6,x6ifft);
%title("6点FFTを逆FFTした時間信号")
%xlabel("時刻t[sec]");
%ylabel("振幅");