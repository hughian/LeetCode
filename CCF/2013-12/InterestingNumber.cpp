/*	
问题描述
　　我们把一个数称为有趣的，当且仅当：
　　1. 它的数字只包含0, 1, 2, 3，且这四个数字都出现过至少一次。
　　2. 所有的0都出现在所有的1之前，而所有的2都出现在所有的3之前。
　　3. 最高位数字不为0。
　　因此，符合我们定义的最小的有趣的数是2013。除此以外，4位的有趣的数还有两个：2031和2301。
　　请计算恰好有n位的有趣的数的个数。由于答案可能非常大，只需要输出答案除以1000000007的余数。
输入格式
　　输入只有一行，包括恰好一个正整数n (4 ≤ n ≤ 1000)。
输出格式
　　输出只有一行，包括恰好n 位的整数中有趣的数的个数除以1,000,000,007的余数。
样例输入
    4
样例输出
    3
*/
/*
使用动态规划，根据规则，记状态如下：
	S0：数字2已经使用，余下0,1,3未使用
	S1：数字2,0已经使用，余下1,3未使用
	S2：数字2,3已经使用，余下0,1未使用
	S3：数字0,1,2已经使用，余下3未使用
	S4：数字0,2,3已经使用，余下1未使用
	S5：数字0,1,2,3已全部使用
根据状态变化来计算数目,递推式为：
    S[n][0] = 1
	S[n][1] = S[n-1][0] + S[n-1][1] * 2
	S[n][2] = S[n-1][0] + S[n-1][2]
    S[n][3] = S[n-1][1] + S[n-1][3] * 2
    S[n][4] = S[n-1][1] + S[n-1][2] + S[n-1][4] * 2
    S[n][5] = S[n-1][3] + S[n-1][4] + S[n-1][5] * 2
*/
#include<iostream>
#include<vector>
using namespace std;
class Solution{
	
public:
	void InterestingNum(){
		long long mod = 1000000007;
		long long n;
		cin>>n;
		long long ** states = new long long *[n+1];
		for(long long i=0;i<n+1;i++)
			states[i] = new long long[6];
		for(long long i=0;i<6;i++)
			states[0][i] = 0;
		
		for(long long i=1;i<=n;i++){
			long long j = i - 1;
			states[i][0] = 1;
			states[i][1] = (states[j][0] + states[j][1] * 2) % mod;
			states[i][2] = (states[j][0] + states[j][2] ) % mod;
			states[i][3] = (states[j][1] + states[j][3] * 2) % mod;
			states[i][4] = (states[j][1] + states[j][2] + states[j][4] * 2) % mod;
			states[i][5] = (states[j][3] + states[j][4] + states[j][5] * 2) % mod;
		}
		cout<<states[n][5]<<endl;
	}
};

int main()
{
    Solution s;
    s.InterestingNum();
	return 0;
}

