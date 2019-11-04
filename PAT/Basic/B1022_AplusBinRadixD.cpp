#include<iostream>
#include<stack>
using namespace std;

int main(){
    long long a,b,d;
    char buf[100] = {'\0'};
    int top = 0;
    cin>>a>>b>>d;
    long long r = a + b;
    int q;
    while(r){
        q = r % d;
        buf[top++] = (char)(q + '0');
        r = r / d;
    }
    if(top==0)
        cout<<"0";
    else{
        while(top>0){
            cout<<buf[--top];
        }
    }
    return 0;
}
