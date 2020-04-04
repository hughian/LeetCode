//
// Created by Hughian on 2020/4/4.
//

// 尽可能的剪成 3 or 4， 凑段数可以将 4 剪成两个 2
// 剪成 m 段，m > 1
class Solution {
public:
    int cutRope(int number) {
        if (number < 2)
            return 0;
        else if (number <= 3)
            return number - 1;
        else if (number == 4)
            return 4;
        else {
            int res = 1;
            while (number > 4) {
                res *= 3;
                number -= 3;
            }
            if (number > 0)
                res *= number;
            return res;
        }
    }
};