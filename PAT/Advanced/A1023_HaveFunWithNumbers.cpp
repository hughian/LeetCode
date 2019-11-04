#include<iostream>
#include<string>
#include<vector>
using namespace std;
vector<int> a(11,0);
vector<int> b(11,0);
int main()
{
    string num;
    cin>>num;
    int i=0;
    for(;i<(int)num.length();i++)
        a[num[i]-'0']++;

    int t =0,c = 0;
    for(i=i-1;i>=0;i--){
        t = 2 * (num[i]-'0') + c;
        c = t/10;
        num[i] = (t%10 + '0'); 
    }
    if(c){
        num = to_string(c)+num;
    }
    for(i=0;i<(int)num.length();i++)
        b[num[i]-'0']++;

    for(int i=0;i<10;i++)
        if(a[i] != b[i]){
            cout<<"No\n";
            cout<<num;
            return 0;
        }
    cout<<"Yes"<<endl;
    cout<<num;
    return 0;
}
