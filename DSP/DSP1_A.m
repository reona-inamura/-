A=1;
theta=0;
D=0;
T0=1;
F0 = 1/ T0;
N = 20;
Ts = 1/N;
n = 0:N-1;

xn = A*sin(2*pi*F0*n*Ts+theta) + D;
yn = zeros(N,1);
for c = 1:N/2
    yn(c,1) = 1;
end
figure(1);
%stem(n*Ts,yn);
stem(n,yn);
xlabel("時刻t[sec]");
ylabel("振幅");
%------------2-----------
hold on;
N=1000;
na = 0:N-1;
Tsa= 1/N;
xt = A*sin(2*pi*F0*na*Tsa+theta) + D;
yt = zeros(N,1);
for c = 1:N/2
    yt(c,1) = 1;
end
%plot(na*Tsa,yt);
plot(na/50,yt);
legend("y[n]","y(t)");
hold off;
%----------4------------
N=20;
k=0:N-1;
omega = 2 * pi * k /N;
f = omega / 2/ pi;
Fs = 1/Ts;
Omega = omega * Fs;
F = Omega /2/pi;
%-----------3----------------
Xk = fft(yn,N);
amp = abs(Xk);
phase = angle(fix(Xk));
figure(2);
subplot(1,2,1);
stem(F, amp);
xlabel("周波数F[Hz]");
ylabel("振幅スペクトル|X[k]|");
subplot(1,2,2);
stem(Omega,phase);
xlabel("角周波数\Omega[rad/sec]");
ylabel("位相スペクトル\angle X[k] [rad]");
