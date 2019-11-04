#include<iostream>
#include<cstdio>
using namespace std;

int main()
{
    int c1,c2;
    cin>>c1>>c2;
    int duration = (int) ((c2 - c1)/100.0 + 0.5); //ËÄÉáÎåÈë
    int h=duration / 3600;
    duration = duration % 3600;
    int m=duration / 60;
    int s = duration %60;
    printf("%02d:%02d:%02d",h,m,s);
    return 0;
}

