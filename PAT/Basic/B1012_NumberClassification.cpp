#include<iostream>
#include<vector>
#include<cstdio>
using namespace std;
void print(int a,int flg)
{
    if(flg==1)
        cout<<a;
    else
        cout<<"N";
}
void print(float f,int flg)
{
    if(flg==1)
        printf("%.1f",f);
    else
        cout<<"N";
}

int main()
{
    int n,tmp;
    cin>>n;
    int r = 1;
    int sum0 = 0,sum1=0,num2=0,sum3=0,num3=0,max4=-1;
    int flg[5]={0};
    for(int i=0;i<n;i++){
        cin>>tmp;
        switch(tmp%5){
            case 0:
                if(tmp%2==0){
                    sum0 += tmp;
                    flg[0] = 1;
                }
                break;
            case 1:
                flg[1] = 1;
                sum1 += r*tmp;
                r = -r;
                break;
            case 2:
                flg[2] = 1;
                num2 ++;
                break;
            case 3:
                flg[3] = 1;
                sum3 += tmp;
                num3++;
                break;
            case 4:
                flg[4] = 1;
                if(tmp>max4)
                    max4 = tmp;
                break;
        }
    }
    print(sum0,flg[0]);
    cout<<" ";
    print(sum1,flg[1]);
    cout<<" ";
    print(num2,flg[2]);
    cout<<" ";
    print((float)sum3/num3,flg[3]);
    cout<<" ";
    print(max4,flg[4]);
    return 0;
}
