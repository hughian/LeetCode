
#include<iostream>
#include<vector>
#include<cstdio>
using namespace std;
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
	int root;
    for(int i=0;i<n;i++){ //使用并查集方法存储树形
        cin>>vec[i];
		if(vec[i]==-1)
			root = i;
    }
	price[root] = p;

    double max = 0.0;
    double dt;
    int cnt = 0;
    for(unsigned int i=0;i<n;i++){
        dt = getprice(i);
        if(dt > max)
            max = dt;
    }
    for(unsigned i=0;i<n;i++){
        dt = getprice(i);
        if(dt == max)
            cnt++;
    }
    printf("%.2f %d",max,cnt);
    return 0;
}
