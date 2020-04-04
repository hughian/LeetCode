//
// Created by Hughian on 2020/4/4.
//

// 使用一个额外的栈保存当前的最小值，出栈的时候一起弹出。
class Solution {
    stack<int> stk, s;
public:
    void push(int value) {
        if (stk.empty())
            s.push(value);
        else
            s.push(::min(value, stk.top())); // 用到了全局的 ::min()
        stk.push(value);
    }
    void pop() {
        stk.pop();
        s.pop();
    }
    int top() {
        return stk.top();
    }
    int min() {
        return s.top();
    }
};