#include<iostream>
#include<string>
using namespace std;

int main()
{
    int n;
    cin>>n;
    string str="";
    int d = n%10;
    for(int i=1;i<=d;i++)
        str += char(i+'0');
    if(n>9){
        n = n/10;
        int s = n%10; //
        for(int i=0;i<s;i++)
            str = 'S' + str;
    }
    if(n>9){
        n = n/10;
        for(int i=0;i<n;i++)
            str = 'B' + str;
    }
    cout<<str;
    return 0;

}
