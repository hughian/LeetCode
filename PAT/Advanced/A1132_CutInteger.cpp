#include<iostream>
#include<string>
#include<cstdio>
#include<cstring>
#include<cstdlib>
using namespace std;

int main()
{
    int n;
    cin>>n;
    long long z,a,b;
    char buf[100];
    string as,bs;
    for(int i=0;i<n;i++){
        cin>>z;
        sprintf(buf,"%lld",z);
        string str=string(buf);
        as = str.substr(0,str.length()/2);
        bs = str.substr(str.length()/2,str.length()/2);

        a = atoll(as.c_str());
        b = atoll(bs.c_str());
        //����Ҫ�������Ϊ������
        if(a*b != 0 && z%(a*b)==0) 
            cout<<"Yes"<<endl;
        else
            cout<<"No"<<endl;
    }
    return 0;
}
