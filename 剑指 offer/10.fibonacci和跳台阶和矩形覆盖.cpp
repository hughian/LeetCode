//
// Created by Hughian on 2020/4/4.
//


// 根据递推 fib(x) = fib(x-1) + fib(x-2) 使用循环或者递推求解就 ok 了。
// fibonacci 有 log n 的解法，使用矩阵形式，然后借助矩阵快速幂来实现。
// 矩阵形式：
//    [[f(n)  f(n-1)],[ f(n-1) f(n-2)]]  = [[1  1], [1  0]] ^(n-1)
// 上述形式可以用归纳法证明，然后我们直接求右半部分就可以得到 f(n)

class Solution {
public:
    int Fibonacci(int n) {
        int a0 = 0, a1 = 1;
        if (n==0){
            return 0;
        }else if (n==1){
            return 1;
        }else{
            int tmp = 0;
            for(int i=2; i<=n;i++){
                tmp = a0 + a1;
                a0 = a1;
                a1 = tmp;
            }
            return tmp;
        }
    }
};

// 跳台阶，依然是个 fibonacci 数列，思路一样的，注意下初始值就 ok 了。
class Solution {
public:
    int jumpFloor(int number) {
        int a0 = 1, a1 = 2;
        if(number==1 or number ==2)
            return number;
        for(int i=3;i<=number;i++){
            int t = a1;
            a1 = a0 + a1;
            a0 = t;
        }
        return a1;
    }
};


// 变态跳台阶，简单 dp, fibonacci 是记前面两个，这里是记前面所有的
class Solution {
public:
    int jumpFloorII(int number) {
        int res = 0;
        vector<int> dp(number+1, 0);

        for(int i=1; i<=number; i++){
            dp[i] = 1;
            for(int j=1;j<i;j++){
                dp[i] += dp[j];
            }
        }
        return dp[number];
    }
};


// 矩形覆盖，很容易看出递推式是 f(x) = f(x-1) + f(x-2), 依然是 fibonacci
class Solution {
public:
    int rectCover(int number) {
        vector<int> dp(number+1, 0);
        dp[1] = 1;
        dp[2] = 2;
        dp[3] = 3;
        for(int i=4;i<=number;i++){
            dp[i] = dp[i-1] + dp[i-2];
        }
        return dp[number];
    }
};
