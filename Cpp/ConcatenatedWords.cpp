#include<iostream>
#include<vector>
#include<string>
using namespace std;

class Solution {
public:
    vector<string> findAllConcatenatedWordsInADict(vector<string> &words){
        int len;
        int i,j;
        int flg=0;
        vector<string> vs;
        len = words.size();
        string str;
        for(i=0;i<len;i++)
        {
            str = words[i];
            flg=0;
            for(j=0;j<len;j++)
            {
                if(str.find(words[j]) != string::npos)    
                    flg++;
            }
            if(flg>2) //at least two shorter words in the given array
                vs.push_back(str);
        }
        return vs;
    }
};
