#include<iostream>
#include<vector>
#include<string>
using namespace std;

char conv(char c)
{
    if(c>='a' && c<= 'z')
        return c-'a' + 'A';
    else
        return c;
}

int main()
{
    string sen,str;
    vector<char> ans;
    cin>>sen>>str;
    unsigned int i,j;
    int flg,t;
    for(i=0;i<sen.length();i++){
        flg = 0;
        for(j=0;j<str.length();j++){
            if(conv(sen[i])==conv(str[j])){
                flg = 1; break;
            }
        }
        if(flg==0){
            t = 0;
            for(int k=0;k<(int)ans.size();k++){
                if(conv(sen[i])==ans[k])
                    t = 1;
            }
            if(t==0){
                ans.push_back(conv(sen[i]));
            }
        }
    }
    for(int i=0;i<(int)ans.size();i++){
        cout<<ans[i];
    }
    return 0;
}
