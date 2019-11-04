#include<iostream>
#include<vector>
#include<algorithm>
#include<cstdio>
using namespace std;
//A1070
bool mycmp(pair<float,float> a,pair<float,float> b){
    return (a.second * b.first) >= (b.second * a.first);
}

int main()
{
    vector<pair<float,float> > va;
    int n,d;
    cin>>n>>d;
    va.resize(n);
    float f,s; //题目中库存量和总售价是正数，包括小数
    for(int i=0;i<n;i++){
        cin>>f;
        va[i].first = f;
    }
    for(int i=0;i<n;i++){
        cin>>s;
        va[i].second = s;
    }
    sort(va.begin(),va.end(),mycmp);
    float sum = d;
    float res = 0;
    for(int i=0;i<n && sum>0;i++)
    {
        if(va[i].first > sum){
            res += va[i].second / va[i].first * sum;
            break;
        }else{
            res += va[i].second;
            sum -= va[i].first;
        }
    }
    printf("%.2f",res);
    return 0;

}
