//
// Created by Hughian on 2020/4/4.
//

// 翻转单词顺序
class Solution {
public:
    string ReverseSentence(string str) {
        // 空格分词
        if(str.size() == 0)
            return str;
        int i =0, j=0;
        vector<string> words;
        while((j=str.find(' ', i))!= string::npos){
            string t = str.substr(i, j-i);
            words.push_back(t);
            i = j + 1;
        }
        if(i<= str.size())
            words.push_back(str.substr(i, str.size()-i));
        string res;
        while(words.size() > 0){
            res += words.back();
            words.pop_back();
            if (words.size())
                res += " ";
        }
        return res;
    }
};

// 题目二：左旋字符串
// 直接两个子串相加就 ok
class Solution {
public:
    string LeftRotateString(string str, int n) {
        if (str.size() == 0)
            return str;
        int k = n % str.size();
        return str.substr(k, str.size()-k) + str.substr(0, k);
    }
};