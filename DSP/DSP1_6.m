A=1;
theta=0;
D=0;
T0=1;
F0 = 1/ T0;
N = 8;
Ts = 1/N;
n = 0:N-1;

an = A*sin(2*pi*F0*n*Ts+theta) + D;
an = 2*sin(2*pi*1*n*Ts+0)+1*sin(2*pi*2*n*Ts+pi)+0;
figure(1);
stem(n* Ts,an);
xlabel("時刻t[sec]");
ylabel("振幅");
%------------2-----------
hold on;
N=1000;
na = 0:N-1;
Tsa= 1/N;
at = A*sin(2*pi*F0*na*Tsa+theta) + D;
at = 2*sin(2*pi*1*na*Tsa+0)+1*sin(2*pi*2*na*Tsa+pi)+0;
plot(na*Tsa,at);
legend("a[n]","a(t)");
hold off;
%----------4------------
N=8;
k=0:7;
omega = 2 * pi * k /N;
f = omega / 2/ pi;
Fs = 1/Ts;
Omega = omega * Fs;
F = Omega /2/pi;
%-----------3----------------
Xk = fft(an,N);
amp = abs(Xk);
phase = angle(fix(Xk));
figure(2);
subplot(1,2,1);
stem(F, amp);
xlabel("周波数F[Hz]")
ylabel("振幅スペクトル|X[k]|")
subplot(1,2,2);
stem(Omega,phase);
xlabel("角周波数\Omega[rad/sec]")
ylabel("位相スペクトル\angle X[k] [rad]")
