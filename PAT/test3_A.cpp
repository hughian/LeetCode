#include<iostream>
#include<string>
#include<vector>
using namespace std;

int c2i(char c){
    if(c>='a' && c<='z')
        return c-'a';
    else if(c>='A' && c<='Z')
        return c-'A'+26;
    else
        return 52;
}

int main()
{
    int n;
    cin>>n;
    int ans = 0;
    for(int i=0;i<n;i++){
        string str;
        vector<int> vec(70,0);
        cin>>str;
        for(unsigned j=0;j<str.length();j++){
            vec[c2i(str[j])]++;
        }
        for(int j=0;j<52;j++){
            if(vec[j]>0) ans++;
        }
    }
    cout<<ans;
    return 0;
}
