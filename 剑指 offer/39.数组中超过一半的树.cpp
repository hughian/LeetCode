//
// Created by Hughian on 2020/4/4.
//

// 数组众数，使用一个计数变量，最后找到的数字再数一遍确保超过一半了
class Solution {
public:
    int MoreThanHalfNum_Solution(vector<int> numbers) {
        if (numbers.size() == 0)
            return 0;
        int cnt = 0, num = numbers[0];
        for (auto x: numbers) {
            if (x == num) {
                cnt += 1;
            } else {
                cnt -= 1;
                if (cnt == 0) {
                    cnt = 1;
                    num = x;
                }
            }
        }
        cnt = 0;
        for (auto x: numbers) {
            if (x == num) {
                cnt++;
            }
        }
        if (cnt > numbers.size() / 2) {
            return num;
        } else {
            return 0;
        }

    }
};