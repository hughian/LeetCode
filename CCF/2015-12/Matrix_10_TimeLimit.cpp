/*
问题描述
　　创造一个世界只需要定义一个初状态和状态转移规则。
　　宏观世界的物体运动规律始终跟物体当前的状态有关，也就是说只要知道物体足够多的状态信息，例如位置、速度等，我们就能知道物体之后任意时刻的状态。
　　现在小M创造了一个简化的世界。
　　这个世界中，时间是离散的，物理规律是线性的：世界的初始状态可以用一个m维向量b(0)表示，状态的转移方式用m×m的矩阵A表示。
　　若已知这个世界当前的状态是b，那么下一时刻就等于b左乘状态转移矩阵A，即Ab。
　　这个世界中，物体的状态也是离散的，也就是说可以用整数表示。再进一步，整数都可以用二进制编码拆分为有限位0和1。因此，这里的矩阵A和向量b的每个元素都是0或1，矩阵乘法中的加法运算视为异或运算（xor），乘法运算视为与运算（and）。
　　具体地，设矩阵A第i行第j列的元素为ai, j，向量b的第i个元素为bi。那么乘法Ab所得的第k个元素为
　　(ak,1 and b1) xor (ak,2 and b2) xor ⋯ xor (ak,m and bm)
　　矩阵和矩阵的乘法也有类似的表达。
　　小M发现，这样的矩阵运算也有乘法结合律，例如有A(Ab)=(AA)b=A2b。
　　为了保证自己创造的世界维度不轻易下降，小M保证了矩阵A可逆，也就是说存在一个矩阵A-1，使得对任意向量d，都有A-1Ad=d。
　　小M想了解自己创造的世界是否合理，他希望知道这个世界在不同时刻的状态。
　　具体地，小M有n组询问，每组询问会给出一个非负整数k，小M希望你帮他求出Akb。
输入格式
　　输入第一行包含一个整数m，表示矩阵和向量的规模。
　　接下来m行，每行包含一个长度为m的01串，表示矩阵A。
　　接下来一行，包含一个长度为m的01串，表示初始向量b(0)。（b(0)是列向量，这里表示它的转置）
　　注意：01串两个相邻的数字之间均没有空格。
　　接下来一行，包含一个正整数n，表示询问的个数。
　　最后n行，每行包含一个非负整数k，表示询问Akb(0)。
　　注意：k可能为0，此时是求A0b(0) =b(0)。
输出格式
　　输出n行，每行包含一个01串，表示对应询问中Akb(0)的结果。
　　注意：01串两个相邻的数字之间不要输出空格。
样例输入
	3
	110
	011
	111
    101
    10
    0
    2
    3
    14
    1
    1325
    6
    124124
    151
    12312
样例输出
    101
    010
    111
    101
    110
    010
    100
    101
    001
    100
评测用例规模与约定
　　本题使用10个评测用例来测试你的程序。
　　对于评测用例1，m = 10，n = 100，k ≤ 103。
　　对于评测用例2，m = 10，n = 100，k ≤ 104。
　　对于评测用例3，m = 30，n = 100，k ≤ 105。
　　对于评测用例4，m = 180，n = 100，k ≤ 105。
　　对于评测用例5，m = 10，n = 100，k ≤ 109。
　　对于评测用例6，m = 30，n = 100，k ≤ 109。
　　对于评测用例7，m = 180，n = 100，k ≤ 109。
　　对于评测用例8，m = 600，n = 100，k ≤ 109。
　　对于评测用例9，m = 800，n = 100，k ≤ 109。
　　对于评测用例10，m = 1000，n = 100，k ≤ 109。
*/
#include<iostream>
#include<vector>
#include<string>
using namespace std;
class Solution{
    vector<string> A;
    vector<int> aBit;
    string b;
    vector< vector<string> > Atv;
    vector<int> tv;
    char bitMulti(string a,string b,int m){
        char tmp,res = 0;
        for(int i=0;i<m;i++){
            tmp = (a[i]-'0') & (b[i] - '0');
            res = tmp ^ res;
        }
        return res+'0';
    }
    void matTrans(vector<string>& M){
        int size = M.size();
        for(int i=0;i<size;i++){
            for(int j =0;j<size;j++)
                M[i][j] = M[j][i];
        }
    }
	string matMulti(vector<string>& M,string b){
		char tmp[M.size()+1];
        int i;
		for(i=0;i<(int)M.size();i++)
			tmp[i] = bitMulti(M.at(i),b,M.size());
		tmp[i] = '\0';
		return string(tmp);
	}
    vector<string> matMulti(vector<string>&M,vector<string>& N){
        vector<string> C;
        int size = M.size();
        char tmp[size+1];
        tmp[size+1] = '\0';
        for(int i=0;i<size;i++){
            for(int j=0;j<size;j++){
                tmp[j] = N[i][j];
            }
            string str = matMulti(M,string(tmp));
            C.at(i) = str;
        }
        matTrans(C);
        return C;
    }
public:
    void Matrix(){
        int m,n;
        cin>>m;
        A.resize(m);
        b.resize(m);
        for(int i=0;i<m;i++)
            cin>>A[i];
        cin>>b;
        cin>>n;
        int query[n];
        int kmax=0;
        for(int i=0;i<n;i++){
            cin>>query[i];
            if(query[i] > kmax)
                kmax = query[i];
        }

        int r = 1,t = 1;
        while(r < kmax) r *= 2; //  2^r <= k
        r /= 2;
        while(t < r) t *= 2; //2^t <= r;
        Atv.resize(t+1);
        Atv.at(0) = A;
        tv.at(0) = 0;
        r = 1;
        for(int i=1;i<=t;i++){
            A = matMulti(A,A);
            r *= 2;
            Atv.at(i) = A;
            tv.at(i) = r;
        }
        int k;
        for(int i=0;i<t;i++){
            cout<<tv.at(i)<<endl<<"===================";
            for(int j=0;j<m;j++)
                cout<<Atv.at(i).at(j)<<endl;
            cout<<"==================="<<endl;
        }
        for(int i=0;i<n;i++){
            int j=0;
            k = query[i];
            string bk = b;
            
            while(j<query[i]){
                bk = matMulti(A,bk);
                j++;
            }
            cout<<bk<<endl;
        }
    }
};
int main()
{
    Solution s;
    s.Matrix();
    return 0;
}

