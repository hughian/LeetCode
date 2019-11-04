class Solution {
public:
    vector<long long> v;
    double fact(int n){
        double d=1;
        for(int i=1;i<=n;i++){
            d *= i;
            
        }
        return d;
    }
    int numTrees(int n) {
        return fact(2*n)/(fact(n+1)*fact(n))+0.5;
    }
};

int main(){
    int n = 3;
    s = Solution();
    int ret = s.numTrees(n);
    printf("%d",ret);
    return 0;
}