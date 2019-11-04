#include<iostream>
#include<string>
#include<vector>
#include<algorithm>
#include<cstdio>
#include<cstdlib>
using namespace std;
//A1069
bool mycmp(char a,char b){
    return a>b;
}

string bubble(string &str)
{
    string res = str;
    for(int i=0;i<(int)str.length();i++){
        for(int j=str.length()-1;j>=i;j--){
            if(str[i] > str[j]){
                char t = str[i];
                str[i] = str[j];
                str[j] = t;
            }
            if(res[i] < res[j]){
                char t = res[i];
                res[i] = res[j];
                res[j] = t;
            }
        }
    }
    return res;
}

int main()
{
    vector<char> vgreater;
    string str;
    cin>>str;
    while(str.length()<4) str.append("0");
    char buf[100] = {'\0'};
    string strbig = bubble(str);
    int big = atoi(strbig.c_str());
    int small = atoi(str.c_str());
    int diff = big - small;
    printf("%04d - %04d = %04d\n",big,small,diff);
    while(diff!=0 && diff!=6174){
        sprintf(buf,"%d",diff);
        str = string(buf);
        while(str.length()<4) str.append("0");
        strbig = bubble(str);
        big = atoi(strbig.c_str());
        small = atoi(str.c_str());
        diff = big - small;
        printf("%04d - %04d = %04d\n",big,small,diff);
    }
    return 0;
}
