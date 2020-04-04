//
// Created by Hughian on 2020/4/4.
//


// 用一个 map 来记录字符的次数，然后找第一个只出现一次的
class Solution {
public:
    int FirstNotRepeatingChar(string str) {
        map<char, int> mp;
        if (str.length() == 0)
            return -1;
        for (auto i = 0; i < str.length(); i++) {
            mp[str[i]]++;
        }
        for (auto i = 0; i < str.length(); i++) {
            if (mp[str[i]] == 1)
                return i;
        }
        return -1;
    }
};


// 题目二：
// 字符流中第一个只出现一次的字符，还是书中的写法
class Solution
{
    int index = 1;
    map<char, int> mp;
public:
    //Insert one char from stringstream
    void Insert(char ch)
    {
        if (mp[ch] == 0)
            mp[ch] = index;
        else if(mp[ch] > 0)
            mp[ch] = -1;
        index ++;
    }
    //return the first appearence once char in current stringstream
    char FirstAppearingOnce(){
        char ch = 0;
        int min_idx = INT_MAX;
        for(int i=0;i<256;i++){
            if (mp[i] > 0 && mp[i] < min_idx){
                ch = (char)i;
                min_idx = mp[i];
            }
        }
        if (min_idx < INT_MAX)
            return ch;
        return '#';
    }

};