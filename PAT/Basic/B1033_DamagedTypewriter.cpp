#include<iostream>
#include<string>
using namespace std;
char conv(char c)
{
    if(c>='a' && c<='z')
        return c-'a' +'A';
    return c;
}

int main()
{
    char buf[100002];
    cin.getline(buf,100002); //坏键串有可能为空串
    string s(buf);           //所以不能使用cin或者scanf直接读取，
    cin.getline(buf,100002); //而要使用getline或者gets来读一行
    string t(buf);
    int flg = 0;
    for(int i=0;i<(int)s.length();i++){
        if(s[i]=='+'){
            flg = 1;
        }
        string::iterator j=t.begin();
        while(j!=t.end()){
            if(s[i]==conv(*j)){
                j=t.erase(j);
            }
            else
                j++;
        }
    }
    if(flg){
        string::iterator j = t.begin();
        while(j!=t.end()){
            if(*j>='A' && *j<='Z')
                j=t.erase(j);
            else
                j++;
        }
    }
    cout<<t<<endl;
    return 0;
}
