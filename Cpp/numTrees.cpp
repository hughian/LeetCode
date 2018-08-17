#include"LeetCode.h"
#include<cstdio>
class Solution {
public:
    vector<double> v;
    Solution(){
        v.resize(0);
    }
    double fact(int n){
        if(n<v.size()) return v[n];
        double fd = v.size()>0?v.back():1;
        for(int i=v.size();i<=n;i++){
            fd *= i;
            v.push_back(fd);
        }
        return fd;
    }
    int numTrees(int n) {
        return fact(2*n)/(fact(n+1)*fact(n))+0.5;
    }
};

int main()
{
	Solution s;
	cout<<s.fact(0)<<endl;
	cout<<s.fact(1)<<endl;
	cout<<s.fact(2)<<endl;
	cout<<s.fact(3)<<endl;
	cout<<s.fact(4)<<endl;
	cout<<s.fact(5)<<endl;
	cout<<s.fact(6)<<endl;
	cout<<s.fact(19)<<endl;
	printf("%d",s.numTrees(1));
	return 0;
}