#include<iostream>
#include<vector>
#include<cstdio>
using namespace std;
int printf(const char *,...);
int n;
double p,r;
vector<int> vec;
vector<double> price; 
double getprice(int i) //递归计算结点的价格
{
    if(price[i]==0) 
        price[i] = (1 + r*0.01) * getprice(vec[i]);
    return price[i];
}

int main()
{
    cin>>n>>p>>r;
    vec.resize(n);
    price.resize(n);
    for(int i=0;i<n;i++){
        vec[i] = -1;
        price[i] = 0.0;
    }
    price[0] = p;
    
    int k,tmp;
    vector<pair<int,int> > leaf;
    for(int i=0;i<n;i++){ //使用并查集方法存储树形
        cin>>k;
        if(k==0){
            cin>>tmp;
            leaf.push_back(make_pair(i,tmp));
        }
        else{
            for(int j=0;j<k;j++){
                cin>>tmp;
                vec[tmp] = i;
            }
        }
    }
    double all = 0.0;
    for(unsigned int i=0;i<leaf.size();i++){
        all += getprice(leaf[i].first) * leaf[i].second;
    }
    printf("%.1f",all);
    return 0;
}
