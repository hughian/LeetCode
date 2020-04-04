//
// Created by Hughian on 2020/4/4.
//

// 位运算，n = n & (n-1) 表示将 n 的最低位的 1 置零。
class Solution {
public:
    int  NumberOf1(int n) {
        int cnt = 0;
        while(n){
            n = n & (n-1);
            cnt ++;
        }
        return cnt;
    }
};