//
// Created by Hughian on 2020/4/4.
//


// 使用一个栈来模拟这个过程，
class Solution {
public:
    bool IsPopOrder(vector<int> pushV,vector<int> popV) {
        stack<int> s;
        int i = 0, j=0;
        while (i < pushV.size() && j < popV.size()){
            if (pushV[i] == popV[j]){ // 相等，入栈出栈
                i ++;
                j ++;
            }else if(!s.empty()){     // 不相等，栈不为空
                if(s.top() == popV[j]){     // 等于栈顶。出栈
                    s.pop();
                    j++;
                }else{                      // 不等于栈顶，入栈
                    s.push(pushV[i++]);
                }
            }else{                    // 不相等，栈为空，入栈
                s.push(pushV[i++]);
            }
        }
        // pop 序列中还有元素
        while (j < popV.size()){
            // 栈中有元素且相等就弹出
            if(!s.empty() && s.top() == popV[j]){
                s.pop();
                j++;
            }else{ // 不等于或者栈空了就不可能
                return false;
            }
        }
        // 栈为空，且 push 和 pop 序列都遍历完了
        return (s.empty() && i == pushV.size() && j == popV.size());
    }
};