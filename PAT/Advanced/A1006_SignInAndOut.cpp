#include<iostream>
#include<vector>
#include<string>
using namespace std;
int str2Sec(string str){
    int hour = (str[0]-'0' )*10 + str[1] - '0';
    int min = (str[3]-'0')  * 10 + str[4] - '0';
    int sec = (str[6] -'0') * 10 + str[7] - '0';
    int time =( hour * 60 + min ) * 60 + sec;
    return time;
}

int main()
{
    int m;
    cin>>m;
    string str,IDmax,IDmin;
    int max = 0,min = 86400;
    pair<string,pair<int,int> > tmp;
    for(int i=0;i<m;i++){
        cin>>str;
        tmp.first = str;
        cin>>str;
        tmp.second.first = str2Sec(str);
        cin>>str;
        tmp.second.second =str2Sec(str);
        if(min > tmp.second.first){
            min = tmp.second.first;
            IDmin = tmp.first;
        }
        if(max < tmp.second.second){
            max = tmp.second.second;
            IDmax = tmp.first;
        }
        //v.push_back(tmp);
    }
    cout<<IDmin<<" "<<IDmax;
    return 0;
}
