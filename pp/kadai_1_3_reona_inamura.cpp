#include<iostream>
#include<cmath>
#include<random>
#include<fstream>
#include<cstring>
#include<vector>
using namespace std;

void save_file(const string& filename, const vector<int>& data){
    ofstream file(filename);
    for (const auto& value : data){
        file << value << "\n";
    }
    file.close();
}

int main(){
    int m = 100;
    int n = 1001;
    int d = 1;
    double p = 0.5;
    double sum = 0;
    double average = 0;
    double varsum = 0;
    double var = 0;
    vector<int> S1000(m,0);
    for(int j = 0;j<m;j++){
        random_device rd;
        mt19937 mt(rd());
        uniform_real_distribution<double> dist(0,1);
        vector<double> X(n);
        for(int i = 0; i < n; i++){
            X[i] = dist(mt);
        }
        vector<int> S(n,0);
        for(int i = 1; i < n;i++){
            if (X[i] < 0.5){
                S[i] = S[i-1] + d;
            }
            else{
                S[i] = S[i-1] - d;
            }
        }
        S1000[j] = S[1000];
        string filename = "randomwalk.csv";
        save_file(filename,S);
    }
    
    for (int j = 0;j <m;j++){
        sum = sum + S1000[j];
    }
    average = sum / m;
    for (int j = 0;j < m; j++){
        varsum = varsum + pow((S1000[j] - average),2);
    }
    var = varsum / m;
    
    /*
    for (double value : S1000){
        sum += value;
    }
    average = sum / m;
    for (double value : S1000){
        varsum += pow((value - average),2);
    }
    var = varsum / m;
    */
    cout << "$S_{1000}$の平均値：" << average << endl;
    cout << "$S_{1000}$の分散：" << var << endl;

    


    return 0;
    
}