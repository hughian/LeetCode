//
// Created by Hughian on 2020/4/4.
//


// 使用栈 1 作为入队列栈，使用栈 2 作为出队列
// 因此入队时，push 进栈 1
//     出队时，如果栈 2 为空，将栈 1 中的元素倒入栈 2，再从栈 2 弹出一个元素出队
//             如果栈 2 不为空，直接从栈 2 弹出一个元素出队
class Solution
{
public:
    void push(int node) {
        stack1.push(node);
    }

    int pop() {
        if (stack2.empty()){
            while (!stack1.empty()){
                int t = stack1.top();
                stack1.pop();
                stack2.push(t);
            }
        }
        int r = stack2.top();
        stack2.pop();
        return r;
    }

private:
    stack<int> stack1;
    stack<int> stack2;
};