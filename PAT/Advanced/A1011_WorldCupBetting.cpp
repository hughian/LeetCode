#include<iostream>
#include<string>
#include<cstdio>
using namespace std;
int getMaxIndex(float a[]){
    cin>>a[0]>>a[1]>>a[2];
    if(a[0] > a[1] && a[0]>a[2])
        return 0;
    if(a[1] > a[0] && a[1] > a[2] )
        return 1;
    if(a[2] > a[0] && a[2] > a[1])
        return 2;

}
string s[3] = {"W ","T ","L "}; 
int main()
{
    float a[3],b[3],c[3];
    int am = getMaxIndex(a);
    int bm = getMaxIndex(b);
    int cm = getMaxIndex(c);
    cout<<s[am]<<s[bm]<<s[cm];
    printf("%.2f",(a[am]*b[bm]*c[cm]*0.65 -1)*2+0.005 );
    return 0;
}
