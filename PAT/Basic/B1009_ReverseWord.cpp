#include<iostream>
#include<string>
#include<vector>
using namespace std;

int main()
{
    string str;
    vector<string> vs;
    getline(cin,str);
    str += " ";
    int i=0,j=str.length();
    while(i<j){
        int r = str.find(" ",i);
        string t = str.substr(i,r-i);
        i = r+1;
        vs.push_back(t);
    }
    for(int i=vs.size()-1;i>0;i--)
    {
        cout<<vs[i]<<" ";
    }
    cout<<vs[0];
    return 0;
}
