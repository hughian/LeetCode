#include<iostream>
#include<string>
#include<map>
using namespace std;
map<string,int> mp;
bool isAlphaDigit(char c){
    if(c>='0' && c<='9')
        return true;
    if(c>='a' && c<='z')
        return true;
    if(c>='A' && c<='Z')
        return true;
    return false;
}

string tolower(string &word){
    for(unsigned i=0;i<word.length();i++){
        if(word[i]>='A' && word[i]<='Z')
            word[i] = word[i] - 'A' + 'a';
    }
    while(word[0]==' ') word.erase(word.begin());
    return word;
}

int main()
{
    string s;
    getline(cin,s);
    unsigned last = 0,i = 0;
    bool flag = false;
    while(i<s.length()){
        if(isAlphaDigit(s[i]) && !flag){
            flag = true;
            last = i;
            i++;
        }else if(!isAlphaDigit(s[i]) && flag){
            flag = false;
            string word = s.substr(last,i-last);
            tolower(word);
            //cout<<word<<endl;
            if(word != "")
                mp[word]++;
            last = i+1;
            i++;
        }else{
            i++;
        }
    }
    string word = s.substr(last,i-last);
    tolower(word);
    //cout<<word<<endl;
    if(word!= "") mp[word]++;
    int max = 0;
    string maxs;
    map<string,int>::iterator it = mp.begin();
    while(it!=mp.end()){
        if(it->second > max){
            max = it->second;
            maxs = it->first;
        }
        it++;
    }
    cout<<maxs<<" "<<max<<endl;
    return 0;
}
