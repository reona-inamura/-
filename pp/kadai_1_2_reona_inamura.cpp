#include<iostream>
#include<cmath>
#include<random>
#include<fstream>
#include<cstring>
#include<vector>
using namespace std;
vector<double> exp_rand(double lambda, int n){
    random_device rd;
    mt19937 mt(rd());
    uniform_real_distribution<double> dist(0,1);
    vector<double> X(n);
    for(int i = 0; i < n; i++){
        X[i] = -1 / lambda * log(1-dist(mt));
    }
    return X;
}
void save_file(const string& filename, const vector<double>& data){
    ofstream file(filename);
    for (const auto& value : data){
        file << value << "\n";
    }
    file.close();
}


int main(){
    double lambda = 1.0;
    double n = 10000;
    vector<double> a = exp_rand(lambda,n);
    double sum = 0;
    double average = 0;
    double varsum = 0;
    double var = 0;
    while(lambda < 2.5){
        a = exp_rand(lambda,n);
        for (double value : a){
            sum += value;
        }
        average = sum / n;
        for (double value : a){
            varsum += pow((value - average),2);
        }
        var = varsum / n;
        cout << "lambda = " << lambda << "の平均値の理論値：" << 1/ lambda << endl;
        cout << "実験の平均値：" << average << endl;
        cout << "lambda = " << lambda << "の分散の理論値：" << 1/ lambda /lambda << endl;
        cout << "実験の分散：" << var << endl;
        string filename = "lambda_" + to_string(lambda) +".csv";
        /*string filename;
        if (lambda < 1.5){
            filename = "lambda10.csv";
        }
        else if (lambda < 2.0){
            filename = "lambda15.csv";
        }
        else{
            filename = "lambda20.csv";
        }*/
        save_file(filename, a);


        sum = 0;
        average = 0;
        varsum = 0;
        var = 0;
        lambda += 0.5;
    }
    return 0;
}