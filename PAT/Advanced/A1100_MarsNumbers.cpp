#include<iostream>
#include<string>
#include<map>
using namespace std;

string lower[] = {"tret","jan", "feb", "mar", "apr", "may", "jun", "jly", "aug", "sep", "oct", "nov", "dec"};
string higher[] ={"tret","tam", "hel", "maa", "huh", "tou", "kes", "hei", "elo", "syy", "lok", "mer", "jou"};
map<string,int> mplow,mphigh;
void Earth2Mars(string str)
{
    int x = stoi(str);
    string ans;
    if(x<13)
        ans = lower[x];
    else if(x%13==0)
        ans = higher[x/13];
    else{
        ans = lower[x%13];
        ans = higher[x/13] + " " + ans;
    }
    cout<<ans<<endl;
}
void Mars2Earth(string str){
    if(str.length()>3){
        string high = str.substr(0,3);
        string low = str.substr(4,3);
        cout<<(mphigh[high]*13+mplow[low])<<endl;
    }else{
        if(mplow.count(str)!=0)
            cout<<mplow[str]<<endl;
        else
            cout<<mphigh[str]*13<<endl;
    }
}

int main()
{
    string ns;
    getline(cin,ns);
    int n = stoi(ns);
    for(int i=0;i<13;i++){
        mplow[lower[i]] = i;
        mphigh[higher[i]] = i;
    }
    string str;
    for(int i=0;i<n;i++){
        getline(cin,str);
        if(str[0] >='0' && str[0] <='9')
            Earth2Mars(str);
        else
            Mars2Earth(str);
    }
}
