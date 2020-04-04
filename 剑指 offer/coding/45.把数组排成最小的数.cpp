//
// Created by Hughian on 2020/4/4.
//

// 可以对两个串 a,b 进行循环比较，也可以像下面这样。
int my_cmp(string &a, string &b){
    return a+b < b+a;  // trick 的点
}

class Solution {
public:
    string PrintMinNumber(vector<int> numbers) {
        vector<string> vec;
        for (auto x: numbers){
            vec.emplace_back(to_string(x));
            sort(vec.begin(), vec.end(), my_cmp);
        }
        stringstream ss;
        for(auto &s: vec){
            ss << s;
        }
        return ss.str();
    }
};