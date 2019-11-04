#include<iostream>
#include<map>
#include<string>
using namespace std;

//#define __NODEBUG__

class Solution{
    string tmplts[111];
    map<string,string> vars;
public:
    void dealwith(string &str,int begin){
        int i=0;
        string tmp;
        string tt = str;
        int len = tt.length();
        i = tt.find('{',begin);
        if(i != -1 &&(i+1<len && tt[i+1] == '{') &&(i+2<len && tt[i+2]==' ')){
            int r = tt.find('}',i+1);
            if(r==-1 || (r+1 >len) || tt[r+1] != '}') return;
            tmp = tt.substr(i+3,r-4-i);
            str = tt.substr(0,i) + vars[tmp] + tt.substr(r+2,len-r-2);
            dealwith(str,0);
        }
        else {
            if(i != -1)
                dealwith(str,i+2);
        }

    }
    void Templates(){
        int n,m;
        cin>>m>>n;
        string str;
        cin.get();//don't forget this again
        for(int i=0;i<m;i++){
            getline(cin,str);  //this function was defined in <string>
            tmplts[i] = str;
        }
        for(int i=0;i<n;i++){
            string var,value;
            string tmp;
            getline(cin,tmp);
            int r = tmp.find(" ",0);
            var=tmp.substr(0,r);
            value = tmp.substr(r+2,tmp.length()-3-r);
            vars[var] = value;
        }
        for(int i=0;i<m;i++){
            dealwith(tmplts[i],0);
            cout<<tmplts[i]<<endl;
        }
    }
};

#include<stdio.h>
int main()
{
#ifndef __NODEBUG__
freopen("test.txt","r",stdin);
#endif
    Solution s;
    s.Templates();
    return 0;
}
