//
// Created by Hughian on 2020/4/4.
//

// 枚举排列的方法就很简单，但是字母可能会有重复的情况，因此我们需要使用一个set来去重
class Solution {
public:
    void swap(string& str, int i, int j){
        char c = str[i];
        str[i] = str[j];
        str[j] = c;
    }

    void helper(set<string>& res, string str, int i){
        if(i >= str.length()){
            res.insert(str);
        }else{
            for (int k=i;k<str.length();k++){
                swap(str, i, k);
                helper(res, str, i+1);
                swap(str, i, k);
            }
        }
    }

    vector<string> Permutation(string str) {
        if(str.length() == 0)
            return vector<string>();

        set<string> s;
        helper(s, str, 0);

        return vector<string>(s.begin(), s.end());
    }
};