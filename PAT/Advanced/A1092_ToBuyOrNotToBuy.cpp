#include<iostream>
#include<vector>
#include<string>
using namespace std;
vector<int> shops(62,0);
int getidx(char c)
{
    if(c>='0' && c<='9') return c-'0';
    else if(c>='a' && c<='z') return c-'a'+10;
    else return c-'A' + 36;
}
int main()
{
    string st,se;
    cin>>st>>se;
    for(unsigned i=0;i<st.length();i++)
        shops[getidx(st[i])]++;
    for(unsigned i=0;i<se.length();i++){
        shops[getidx(se[i])]--;
    }
    int pos=0,neg=0;
    for(int i=0;i<62;i++){
        if(shops[i]>0)
            pos+=shops[i];
        else
            neg+=shops[i];
    }
    if(neg!=0)
        cout<<"No "<<abs(neg)<<endl;
    else
        cout<<"Yes "<<abs(pos)<<endl;
    return 0;
}
