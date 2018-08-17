#include<iostream>
#include<string>
#include<map>
using namespace std;

class Solution{
public:
    int lengthOfLongestSubstring(string s){
        string fans(""),ans("");
        double count=0.0;
        short ascii[128];
        char c;
        int max=(int)s.length();
        int pos=0;  //
        int len=0;
        int i,j;
        for(j=0;j<128;j++)
            ascii[j]=-1;
        for(i=0; i < max;i++)
        {
            count += 1.0;
            c=s.at(i);
            if(ascii[(int)c] == -1){
                ascii[(int)c]=i;
                ans.push_back(c);
                pos++;
            }else{
                if(pos>len){
                    len=pos;
                    fans=ans;
                }
                i=(int)ascii[(int)c];
                pos=0;
                ans.clear();
                for(j=0;j<128;j++){
                    ascii[j]=-1;
                }
            }
        }
        if(pos>len){
            len=pos;
            fans = ans;
        }
        cout<<count<<endl;
        return len;
    }
};
