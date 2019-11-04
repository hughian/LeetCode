#include<iostream>
#include<string>
#include<vector>
using namespace std;
vector<bool> mark(130,false);

int main()
{
    string s1,s2,ans;
    getline(cin,s1);
    getline(cin,s2);
    unsigned i=0;
    for(;i<s2.length();i++){
        mark[(int)s2[i]] = true;
    }
    for(i=0;i<s1.length();i++){
        if(!mark[(int)s1[i]])
            ans.push_back(s1[i]);
    }
    cout<<ans;
    return 0;
}
