#include<iostream>
#include<vector>
#include<cstdio>
using namespace std;
int scanf(const char*,...);
int printf(const char*,...);
int main()
{
    vector<int> dist(100010,0);
    int N,M;
    int sum = 0,t;
    cin>>N;
    for(int i=1;i<=N;i++){ 
        cin>>t;
        sum += t;
        dist[i+1] = sum;
    }
    cin>>M;
    int a,b;
    int len1,len2;
    for(int i=0;i<M;i++){
        scanf("%d%d",&a,&b);
        if(a==b) {printf("0\n");continue;}
        if(a>b) swap(a,b);
        len1 = dist[b] - dist[a];
        len2 = sum-len1;
        printf("%d\n",len1<len2?len1:len2);
    }
    return 0;
}
