#include<iostream>
#include<vector>
#include<algorithm>

using namespace std;

int main()
{
    int n;
    cin>>n;
    vector<int> v(n);
	for(int i=0;i<n;i++)
        cin>>v[i];
    sort(v.begin(),v.end());
    float ans = v[0];
	for(int i=1;i<n;i++){
		ans = 0.5 * (ans + v[i]);
	}
    cout<<floor(ans);
    return 0;
}
