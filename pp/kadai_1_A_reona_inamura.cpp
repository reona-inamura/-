#include<iostream>
#include<cmath>
#include<random>
#include<fstream>
#include<cstring>
#include<vector>
using namespace std;


vector<int> rnd_lgc(int n, int A, int B, int M, int X0){
    vector<int> X(n);
    X[0] = X0;
    for(int i = 1; i < n; i++){
        X[i] = (A * X[i-1] + B) % M;
    }
    return X;
}
vector<int> rnd_mt(int n, int M){
    random_device rd;
    mt19937 mt(rd());
    uniform_int_distribution<int> dist(0,M-1);
    vector<int> X(n);
    for(int i = 0; i < n; i++){
        X[i] = dist(mt);
    }
    return X;
}
int main(){
    int n = 100000;
    int A = 5;
    int B = 7;
    int M = 16;
    int X0 = 6;
    vector<int> X(n,0);
    vector<int> Y(n,0);
    X = rnd_lgc(n,A,B,M,X0);
    Y = rnd_mt(n,M);
    /*for(int i = 0;i<n;i++){
        cout << X[i] << endl;
    }*/
    int SetX[M][M] = {};
    int SetY[M][M] = {};
    int countx = 0;
    int county = 0;
    for(int i = 1;i<n;i++){
        SetX[X[i-1]][X[i]] = 1;
        SetY[Y[i-1]][Y[i]] = 1;
    }
    for (int i = 0;i<M;i++){
        for (int j = 0;j<M;j++){
            countx = countx + SetX[i][j];
            cout << SetX[i][j] << " ";
            county = county + SetY[i][j];
        }
        cout << endl;
    }
    cout << countx << "組：線形合同法" << endl;
    string filename = "lgc.csv";
    ofstream filex(filename);
    for (int i = 1; i<n;i++){
        filex << X[i-1] << "," <<X[i] << "\n";
    }
    filex.close();
    
    cout << county << "組：メルセンヌ・ツイスタ" << endl;
    filename = "mt.csv";
    ofstream filey(filename);
    for (int i = 1; i<n;i++){
        filey << Y[i-1] << "," <<Y[i] << "\n";
    }
    filey.close();

    return 0;
}