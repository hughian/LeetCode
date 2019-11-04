#include<iostream>
#include<vector>
#include<string>
using namespace std;

int main()
{
    string str;
    cin>>str;
    int n = str.length();
    int n1=0,n2=0,n3=0;
    for(n2=3;n2<=n;n2++){
        int k = (n+2-n2)/2;
        if(k<=n2 && k>n1)
            n1 = k;
    }
    n2 = n + 2 - 2 * n1; 
    n3 = n1;
    vector< vector<char> > vvc(n1+1,vector<char>(n2+1,' '));
    int k = 0;
    for(int i=0;i<n1-1;i++)
        vvc[i][0] = str[k++];
    for(int i=0;i<n2;i++)
        vvc[n1-1][i] = str[k++];
    for(int i=n1-2;i>=0;i--)
        vvc[i][n2-1] = str[k++];

    for(int i=0;i<n1;i++){
        for(int j=0;j<n2;j++)
            cout<<vvc[i][j];
        cout<<endl;
    }
    return 0;
}
