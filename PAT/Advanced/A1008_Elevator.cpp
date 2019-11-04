#include<iostream>
#include<vector>
using namespace std;

int main()
{
    int n;
    int sum = 0;
    vector<int> v(101,0);
    cin>>n;
    for(int i=1;i<=n;i++)
        cin>>v[i];
    for(int i=1;i<=n;i++){
        if(v[i] > v[i-1]) { //up
            sum +=( v[i] - v[i-1] ) * 6 + 5;
        }
        else if(v[i] < v[i-1]) {//down
            sum += (v[i-1] - v[i]) * 4 + 5;
        }
        else
            sum += 5;
    }
    cout<<sum;
    return 0;
}
