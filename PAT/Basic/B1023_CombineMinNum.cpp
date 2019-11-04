#include<iostream>

using namespace std;
int main()
{
    int a[10];
    for(int i=0;i<10;i++){
        cin>>a[i];
    }
    int flg = 0 ;
    for(int i=1;i<10;i++){
        if(a[i]){
            cout<<i;
            flg = 1;
            a[i]--;
        }
        if(a[0] && flg){
            for(int j=0;j<a[0];j++)
                cout<<0;
            a[0] = 0;
        }
        if(a[i]){
            for(int j=0;j<a[i];j++)
                cout<<i;
        }
    }
    return 0;
}
