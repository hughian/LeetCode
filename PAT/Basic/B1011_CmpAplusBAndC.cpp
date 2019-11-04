#include<iostream>
using namespace std;

int main()
{
    int t;
    long long a,b,c;
    cin>>t;
    for(int i=0;i<t;i++){
        cin>>a>>b>>c;
        if(a+b > c)
            cout<<"Case #"<<i+1<<": true";
        else
            cout<<"Case #"<<i+1<<": false";
        cout<<endl;
    }

    return 0;
}
