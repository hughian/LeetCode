//
// Created by Hughian on 2020/4/4.
//


class Solution {
public:
    // 使用个 list（链表）模拟
    int _LastRemaining_Solution(int n, int m) {
        if (n < 1 || m < 1)
            return -1;

        int i = 0;
        list<int> arr;
        for (int i = 0; i < n; i++) {
            arr.push_back(i);
        }
        auto it = arr.begin();

        while (arr.size() > 1) {
            for (int i = 0; i < m - 1; i++) {
                it++;
                if (it == arr.end())
                    it = arr.begin();
            }
            auto nx = ++it;
            if (nx == arr.end())
                nx = arr.begin();

            --it;
            arr.erase(it);
            it = nx;
        }
        return *arr.begin();
    }

    //  约瑟夫环
    // f(n, m) = { 0                     ; n= 1
    //           { [f(n-1, m) + m] % n   ; n>1
    int LastRemaining_Solution(int n, int m) {
        if (n < 1 || m < 1)
            return -1;
        int last = 0;
        for (int i = 2; i <= n; i++) {
            last = (last + m) % i;
        }
        return last;
    }
};

