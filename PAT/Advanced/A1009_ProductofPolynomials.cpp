#include<iostream>
#include<cstdio>
#include<vector>
using namespace std;
void getdata(vector<pair<int,double> > & v)
{
    int k;
    cin>>k;
    pair<int,double> pt;
    for(int i=0;i<k;i++){
        cin>>pt.first>>pt.second;
        v.push_back(pt);
    }
}

int main()
{
    vector<double> res(3001,0.0);
    vector<pair<int,double> > v1,v2;
    getdata(v1);
    getdata(v2);
    int index;
    for(int i=0;i<(int)v1.size();i++){
        for(int j=0;j<(int)v2.size();j++){
            index = v1[i].first + v2[j].first;
            res[index] += v1[i].second * v2[j].second;
        }
    }

    int num = 0;
    for(int i=0;i<3001;i++)
        if(res[i] != 0)
            num++;
    cout<<num;
    for(int i=3000;i>=0;i--){
        if(res[i] != 0)
            printf(" %d %.1lf",i,res[i]);
    }
    return 0;
}
