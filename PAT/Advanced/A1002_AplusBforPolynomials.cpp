#include<iostream>
#include<cstdio>
using namespace std;

int main()
{
    float a[1001];
    for(int i=0;i<1001;i++)
        a[i]=0.0;
    int k;
    int ni;
	float ai;
    int cnt = 0;
    cin>>k;
    for(int i=0;i<k;i++){
        cin>>ni>>ai;
        a[ni] += ai;
    }
    cin>>k;
    for(int i=0;i<k;i++){
        cin>>ni>>ai;
        a[ni] += ai;
    }
	
	
    for(int i=1000;i>=0;i--){
        if(a[i] != 0.0)
            cnt ++;
    }
    cout<<cnt;
    for(int i=1000;i>=0;i--){
        if(a[i] != 0.0){
            printf(" %d %.1f",i,a[i]);
        }
    }
    return 0;
}   
