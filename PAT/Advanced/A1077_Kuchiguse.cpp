#include<iostream>
#include<string>
#include<vector>
using namespace std;
vector<string> vs(101);
int samelen(string a,string b)
{
    int i = a.length()-1;
    int j = b.length()-1;
    while(i>=0 && j>=0){
        if(a[i]==b[j]){
            i--;j--;
        }else
            break;
    }
    if(i<0)
        return a.length();
    if(j<0)
        return b.length();
    return (a.length()-i-1);
}

int main()
{
    string ns;
    getline(cin,ns);
    int n = stoi(ns);
    for(int i=0;i<n;i++){
        getline(cin,vs[i]);
    }
    int ans = samelen(vs[0],vs[1]);
    for(int i=1;i<n-1;i++){
        ans = min(ans,samelen(vs[i],vs[i+1]));
    }
    if(ans==0)
        cout<<"nai";
    else
        cout<<vs[0].substr(vs[0].length()-ans,ans);
    return 0;
}
