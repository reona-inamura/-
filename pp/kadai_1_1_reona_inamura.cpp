#include<iostream>
#include<cmath>
#include<random>
#include<vector>
using namespace std;
/*double unirand(){
    random_device rd;
    mt19937 mt(rd());
    uniform_real_distribution<double> dist(0,1);
    return dist(mt);
}
double exprand(double lamda){
    double x = -1 / lamda * log(unirand());
    return x;
}
vector<double> exp_rand(double lamda, int n){
    vector<double> X(n);
    for(int i = 0; i < n; i++){
        X[i] = exprand(lamda);
    }
    return X;
}*/
vector<double> exp_rand(double lamda, int n){
    random_device rd;
    mt19937 mt(rd());
    uniform_real_distribution<double> dist(0,1);
    vector<double> X(n);
    for(int i = 0; i < n; i++){
        X[i] = -1 / lamda * log(dist(mt));
    }
    return X;
}

int main(){
    double lamda = 1.0;
    double num = 5;
    vector<double> a = exp_rand(lamda,num);
    for(double value : a){
        cout << value <<endl;
    }
    cout << endl;
    return 0;
}