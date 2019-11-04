#include<iostream>
#include<vector>
#include<cstdio>
using namespace std;
int n;
double p,r;
vector<int> vec;
vector<int> leaf;
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
    for(int i=0;i<n;i++){ //使用并查集方法存储树形
        cin>>k;
        if(k==0)
            leaf.push_back(i);
        else{
            for(int j=0;j<k;j++){
                cin>>tmp;
                vec[tmp] = i;
            }
        }
    }/*
    for(int i=0;i<n;i++){
        cout<<i<<" ";
    }
    cout<<endl;
    for(int i=0;i<n;i++)
        cout<<vec[i]<<" ";
    cout<<endl;
    */
    double min = 10000000000 + 1;
    double dt;
    int cnt = 0;
    for(unsigned int i=0;i<leaf.size();i++){
        dt = getprice(leaf[i]);
        if(dt < min)
            min = dt;
    }
    for(unsigned i=0;i<leaf.size();i++){
        dt = getprice(leaf[i]);
        if(dt == min)
            cnt++;
    }
    printf("%.4f %d",min,cnt);
    return 0;
}
