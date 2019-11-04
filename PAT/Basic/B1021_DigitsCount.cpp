#include<iostream>
#include<string>
using namespace std;

int main()
{
    string str;
    int a[10]={0};
    cin>>str;
    for(int i=0;i<str.length();i++){
        a[str[i]-'0'] ++;
    }
    for(int i=0;i<10;i++){
        if(a[i]){
            cout<<i<<":"<<a[i]<<endl;
        }
    }
    return 0;
}
