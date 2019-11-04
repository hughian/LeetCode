#include<iostream>
#include<stdio.h>
#include<cstdlib>
#include<string>
#include<string.h>
#include<map>
#define BUY 0
#define SELL 1
#define CANCEL 2
using namespace std;
long long s[5000];
double p[5000];
int op[5000];
map<string,int> Trans;
class Solution{
public:
    void CallAuction()
    {
        memset(s,5000*sizeof(long long),-1);
        memset(p,5000*sizeof(double),-1);
        memset(op,5000*sizeof(int),-1);
        Trans["buy"] = BUY;
        Trans["sell"] = SELL;
        Trans["cancel"] = CANCEL;
        char buf[100];
        string str;
        string s[3];
        int cnt = 0;
        while(getline(cin,str)){
            sprintf(buf,"%s",str.data());
            int i = 0;
            char *tmp = strtok(buf," ");
            op[cnt] = Trans[string(tmp)];
            tmp = strtok(NULL," ");
            p[cnt] = atof(tmp);
            tmp = strtok(NULL," ");
            s[cnt] = (long long)atoi(tmp);
        }
    }
};

int main()
{
    Solution s;
    s.CallAuction();
    return 0;
}
