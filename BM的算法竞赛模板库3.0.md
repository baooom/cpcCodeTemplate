

[TOC]

#  代码头

```c++
#include <cstdio>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <queue>
#include <map>
#include <set>
#include <stack>
#include <vector>
#include <string>
#include <iostream>
#include <list>
#include <cstdlib>
#include <bitset>
//#define getchar() (p1 == p2 && (p2 = (p1 = buf) + fread(buf, 1, 1 << 21, stdin), p1 == p2) ? EOF : *p1++)
//char buf[(1 << 21) + 1], * p1 = buf, * p2 = buf;
//#define int long long
#define lowbit(x) (x&(-x))
#define lson root<<1,l,mid
#define rson root<<1|1,mid+1,r
#define pb push_back
typedef unsigned long long ull;
typedef long long ll;
typedef pair<int, int> pii;
#define bug puts("BUG");
const long long INF = 0x3f3f3f3f3f3f3f3fLL;
const int inf = 0x3f3f3f3f;
const int mod = 1000000007;
const double eps = 1e-6;
template<class T>inline void read(T& x) {
	int sign = 1; char c = getchar(); x = 0;
	while (c > '9' || c < '0') { if (c == '-')sign = -1; c = getchar(); }
	while (c >= '0' && c <= '9') { x = x * 10 + c - '0'; c = getchar(); }
	x *= sign;
}
using namespace std;

```

#  数据结构

##  可持久化线段树

```c++
struct Seg
{
	int v, ls, rs;
}val[maxn * 20];
int rot[maxn], cnt;
inline void pushup(int root)
{
	val[root].v = val[val[root].ls].v + val[val[root].rs].v;
}
int buildemp(int l, int r)
{
	int id = cnt++;
	if (l == r)
	{
		val[id].v = 0;
		return id;
	}
	int mid = l + r >> 1;
	val[id].ls = buildemp(l, mid);
	val[id].rs = buildemp(mid + 1, r);
	pushup(id);
	return id;
}

int addnode(int pre, int l, int r, int v)
{
	int id = cnt++;
	if (l == r)
	{
		val[id].v = val[pre].v + 1;
		return id;
	}
	int mid = l + r >> 1;
	val[id] = val[pre];
	if (v <= mid)
		val[id].ls = addnode(val[pre].ls, l, mid, v);
	else
		val[id].rs = addnode(val[pre].rs, mid + 1, r, v);
	pushup(id);
	return id;
}

int query(int l, int r, int lt, int rt, int k)
{
	if (l == r)return l;
	int tt = val[val[rt].ls].v - val[val[lt].ls].v;
	int mid = l + r >> 1;
	if (tt >= k)
		return query(l, mid, val[lt].ls, val[rt].ls, k);
	else
		return query(mid + 1, r, val[lt].rs, val[rt].rs, k - tt);
}

```

##  Splay

```c++
const int maxn = 1e5 + 10;
int n, m, rt, tot;
int fa[maxn], siz[maxn], ch[maxn][2], val[maxn], cnt[maxn];
inline void pushup(int x)
{
	siz[x] = siz[ch[x][0]] + siz[ch[x][1]] + cnt[x];
}

inline void connect(int son, int father, int d)
{
	ch[father][d] = son;
	fa[son] = father;
}

inline int ws(int x)
{
	return ch[fa[x]][1] == x;
}

inline void rot(int x)
{
	int f = fa[x], ff = fa[f];
	int c = ws(x), cc = ws(f);
	connect(ch[x][c ^ 1], f, c);
	connect(f, x, c ^ 1);
	connect(x, ff, cc);
	pushup(f);
	pushup(x);
}

inline void splay(int v, int goal)
{
	while (fa[v] != goal)
	{
		int f = fa[v], ff = fa[f];
		if (ff != goal)
		{
			if (ws(v) == ws(f))rot(f);
			else rot(v);
		}
		rot(v);
	}
	if (goal == 0)rt = v;
}

inline void insert(int x)
{
	int u = rt, f = 0;
	while (u && val[u] != x)
	{
		f = u;
		u = ch[u][x > val[u]];
	}
	if (u)++cnt[u];
	else
	{
		u = ++tot;
		if (f)ch[f][x > val[f]] = u;
		ch[u][0] = ch[u][1] = 0;
		fa[u] = f, val[u] = x, siz[u] = cnt[u] = 1;
	}
	splay(u, 0);
}

inline void find(int x)
{
	int u = rt;
	if (!u)return;
	while (ch[u][x > val[u]] && val[u] != x)
		u = ch[u][x > val[u]];
	splay(u, 0);
}

inline int pre(int x)
{
	find(x);
	int u = rt;
	if (val[u] < x)return u;
	u = ch[u][0];
	while (ch[u][1])u = ch[u][1];
	return u;
}

inline int nxt(int x)
{
	find(x);
	int u = rt;
	if (val[u] > x)return u;
	u = ch[u][1];
	while (ch[u][0])u = ch[u][0];
	return u;
}

inline void delet(int x)
{
	int last = pre(x), next = nxt(x);
	splay(last, 0), splay(next, last);
	int del = ch[next][0];
	if (cnt[del] > 1)--cnt[del], splay(del, 0);
	else ch[next][0] = 0;
}

inline int kth(int x)
{
	int u = rt;
	if (siz[u] < x)return 0;
	while (1)
	{
		int y = ch[u][0];
		if (x > siz[y] + cnt[u])
		{
			x -= siz[y] + cnt[u];
			u = ch[u][1];
		}
		else
		{
			if (x <= siz[y])
				u = y;
			else
				return val[u];
		}
	}
}

```

##  树链剖分

```c++
void dfs1(int u,int pre,int d)
{
	f[u] = pre;
	dep[u] = d;
	sz[u] = 1;
	for (int i=head[u];~i;i=edge[i].next)
	{
		int v = edge[i].to;
        if(v==pre)continue;
		dfs1(v, u,d + 1);
        if (sz[v] > sz[son[u]])son[u]=v;
		sz[u] += sz[v];
	}
}
void dfs2(int u, int to)
{
	top[u] = to;
	dfn[u] = ++cnt;
	if(son[u])
		dfs2(son[u], to);
	for (int i=head[u];~i;i=edge[i].next)
	{
        int v=edge[i].to;
        if(v==fa[u]||v==son[u])
            continue;
		dfs2(v, v);
	}
}

int LCA(int u, int v)
{
	while (top[u] != top[v])
	{
		if (dep[top[u]] < dep[top[v]])swap(u, v);
		u = f[top[u]];
	}
	return dep[u] < dep[v] ? u : v;
}
```

##  虚树

```c++
void insert(int u){
    if(top <= 1) {stk[++top] = u;return ;}
    int lca = LCA(u,stk[top]);
    if(lca == stk[top]) {stk[++top] = u;return ;}
    while(top > 1 && dfn[lca] <= stk[top-1]) {
        addedge(stk[top-1],stk[top]);
        --top;
    }
    if(lca != stk[top]) stk[++top] = lca;
    stk[++top] = u;
}
```

##  树套树(树状数组套权值线段树)

```c++
int k;
const int maxn = 2e5 + 10;
struct NODE
{
    int x, y, z;
    int ansidx;
    bool operator < (const NODE& other)const
    {
        if (x != other.x)
            return x < other.x;
        if (y != other.y)
            return y < other.y;
        return z < other.z;
    }
    bool operator == (const NODE& other)const
    {
        return x == other.x && y == other.y && z == other.z;
    }

} node[maxn];
int nodez[maxn * 90], tot, ls[maxn * 90], rs[maxn * 90];
int rt[maxn << 2];
int ans[maxn], ansout[maxn];
void pushup(int root)
{
    nodez[root] = nodez[ls[root]] + nodez[rs[root]];
}
void updatez(int root, int l, int r, int z, int v)
{
    if (l == r)
    {
        nodez[root] += v;
        return;
    }
    int mid = l + r >> 1;
    if (z <= mid)
    {
        if(!ls[root])
            ls[root] = ++tot;
        updatez(ls[root], l, mid, z, v);
    }
    else
    {
        if(!rs[root])
            rs[root] = ++tot;
        updatez(rs[root], mid + 1, r, z, v);
    }
    pushup(root);
}

void updatey(int y,int z, int v)
{
    for (; y <= k; y += lowbit(y))
    {
        if(!rt[y])
            rt[y] = ++tot;
        updatez(rt[y], 1, k, z, v);
        pushup(rt[y]);
    }
}

int queryz(int root, int l, int r, int lz, int rz)
{
    if (l == lz && r == rz)
    {
        return nodez[root];
    }
    int mid = l + r >> 1;
    if (rz <= mid)
        return queryz(ls[root], l, mid, lz, rz);
    else if (lz > mid)
        return queryz(rs[root], mid + 1, r, lz, rz);
    else
    {
        return queryz(ls[root], l, mid, lz, mid) + queryz(rs[root], mid + 1, r, mid + 1, rz);
    }
}

int queryy(int y, int lz, int rz)
{
    int res = 0;
    for (; y > 0; y -= lowbit(y))
    {
        if(!rt[y])
            continue;
        res += queryz(rt[y], 1, k, lz, rz);
    }
    return res;
}
```

#  数论

##  扩展中国剩余定理

$$
x\equiv c_i(\mod m_i)
$$

```c++
const LL MAXN = 1e6 + 10;
LL K, C[MAXN], M[MAXN], x, y;
LL gcd(LL a, LL b) {
	return b == 0 ? a : gcd(b, a % b);
}
LL exgcd(LL a, LL b, LL& x, LL& y) {
	if (b == 0) { x = 1, y = 0; return a; }
	LL r = exgcd(b, a % b, x, y), tmp;
	tmp = x; x = y; y = tmp - (a / b) * y;
	return r;
}
LL inv(LL a, LL b) {
	LL r = exgcd(a, b, x, y);
	while (x < 0) x += b;
	return x;
}
int main() {	while (~scanf("%lld", &K)) {
		for (LL i = 1; i <= K; i++) scanf("%lld%lld", &M[i], &C[i]);
		bool flag = 1;
		for (LL i = 2; i <= K; i++) {
			LL M1 = M[i - 1], M2 = M[i], C2 = C[i], C1 = C[i - 1], T = gcd(M1, M2);
			if ((C2 - C1) % T != 0) { flag = 0; break; }
			M[i] = (M1 * M2) / T;
			C[i] = (inv(M1 / T, M2 / T) * (C2 - C1) / T) % (M2 / T) * M1 + C1;
			C[i] = (C[i] % M[i] + M[i]) % M[i];
		}
		printf("%lld\n", flag ? C[K] : -1);
	}
	return 0;
}
```

##  线性递推逆元

```c++
int inv[100001];
int p;
int main()
{
	cin >> p;//模数
	inv[1] = 1;//1的逆元为1
	for (int i = 2; i <= 10; i++)
	{
		inv[i] = (p - (p / i)) * inv[p % i] % p;//递推公式
		printf("%d %d %d\n", i, inv[i], (i * inv[i]) % p);
	}
}
```

##  欧拉降幂

$$
\left.\begin{cases}
A^B\equiv A^{B \mod \phi(C)}(\mod C)\qquad B\lt\phi(C)\\
A^B\equiv A^{B \mod \phi(C)+\phi(C)}(\mod C)\qquad B\ge\phi(C)
\end{cases}\right.
$$

```c++
#define MOD(a,b) (a<b?a:a%b+b)
ll phi(ll n)
{
	ll ans = n;
	for (ll i = 2; i * i <= n; ++i)
	{
		if (n % i == 0)
		{
			ans -= ans / i;
			while (n % i == 0)n /= i;
		}
	}
	if (n > 1)ans -= ans / n;
	return ans;
}
ll qmod(ll a, ll n, ll mod)
{
	ll ans = 1;
	while (n)
	{
		if (n & 1)
			ans = MOD(ans * a, mod);
		a = MOD(a * a, mod);
		n >>= 1;
	}
	return ans;
}
ll slove(ll a, ll b, ll mod)
{
	if (b == 0)return MOD(1, mod);
	if (mod == 1)return MOD(a, mod);
	return qmod(a, slove(a, b - 1, phi(mod)), mod);
}
int main()
{
	int t;
	for (read(t); t--;)
	{
		ll a, b, p;
		read(a), read(b), read(p);
		printf("%lld\n", slove(a, b, p) % p);
	}
}
```

##  二次剩余

$$
X^2\equiv d(\mod p)
$$

```c++
#include <time.h>
struct num {
	ll x, y;
};

num mul(num a, num b, ll p)
{
	num ans = { 0,0 };
	ans.x = ((a.x * b.x % p + a.y * b.y % p * w % p) % p + p) % p;
	ans.y = ((a.x * b.y % p + a.y * b.x % p) % p + p) % p;
	return ans;
}

ll powwR(ll a, ll b, ll p) {
	ll ans = 1;
	while (b) {
		if (b & 1)ans = 1ll * ans % p * a % p;
		a = a % p * a % p;
		b >>= 1;
	}
	return ans % p;
}
ll powwi(num a, ll b, ll p) {
	num ans = { 1,0 };
	while (b) {
		if (b & 1)ans = mul(ans, a, p);
		a = mul(a, a, p);
		b >>= 1;
	}
	return ans.x % p;
}

ll solve(ll n, ll p)
{
	n %= p;
	if (p == 2)return n;
	if (powwR(n, (p - 1) / 2, p) == p - 1)return -1;//不存在
	ll a;
	while (1)
	{
		a = rand() % p;
		w = ((a * a % p - n) % p + p) % p;
		if (powwR(w, (p - 1) / 2, p) == p - 1)break;
	}
	num x = { a,1 };
	return powwi(x, (p + 1) / 2, p);
}

int main()
{
	srand(time(0));
	int t;
	scanf("%d", &t);
	while (t--)
	{
		ll n, p;
		scanf("%lld%lld", &n, &p);
		if (!n) {
			printf("0\n"); continue;
		}
		ll ans1 = solve(n, p), ans2;
		if (ans1 == -1)printf("Hola!\n");
		else
		{
			ans2 = p - ans1;
			if (ans1 > ans2)swap(ans1, ans2);
			if (ans1 == ans2)printf("%lld\n", ans1);
			else printf("%lld %lld\n", ans1, ans2);
		}
	}
}

```

##  类欧几里得

$$
f(a,b,c,n)=\sum_{i=0}^{i\le n}\left\lfloor\frac{ai+b}{c}\right\rfloor\\
$$

$$
f(a,b,c,n)=\left.\begin{cases}
\frac{a}{c}*\frac{n*(n+1)}{2}+\frac{b}{c}*(n+1)+f(a\%c,b\%c,c,n)\qquad a\ge c\quad or\quad b\ge c\\
nm-f(c,c-b-1,a,m-1)\qquad else\\
\end{cases}\right.
$$

```c++
ll SUM(ll k)
{
    return k * (k + 1) / 2;
}

ll f(ll a,ll b,ll c,ll n)
{
    if(!a)
        return 0;
    if (a >= c || b >= c)
        return ((a / c) * SUM(n) + (n + 1) * (b / c) + f(a % c, b % c, c, n));
    ll m = (a * n + b) / c;
    return m * n - f(c, c - b - 1, a, m - 1);
}
```


$$
g(a,b,c,n)=\sum_{i=0}^{i\le n}i\left\lfloor\frac{ai+b}{c}\right\rfloor\\
$$

$$
g(a,b,c,n)=\left.\begin{cases}
\frac{a}{c}*\frac{n(n+1)(2n+1)}{6}+\frac{b}{c}*\frac{n(n+1)}{2}+g(a\%c,b\%c,c,n)\qquad a\ge c\quad or\quad b\ge c\\
\frac{nm(n+1)-f(c,c-b-1,a,m-1)-h(c,c-b-1,a,m-1)}{2}\qquad else\\
\end{cases}\right.
$$


$$
h(a,b,c,n)=\sum_{i=0}^{i\le n}\left\lfloor\frac{ai+b}{c}\right\rfloor^2\\
$$

$$
h(a,b,c,n)=\left.\begin{cases}
(\frac{a}{c})^2*\frac{n(n+1)(2n+1)}{6}+(\frac{b}{c})^2*(n+1)+(\frac{a}{c})(\frac{b}{c})*n(n+1)+h(a\%c,b\%c,c,n)+\\2*(\frac{a}{c})^2*g(a\%c,b\%c,c,n)\qquad a\ge c\quad or\quad b\ge c\\
nm(m+1)-2g(c,c-b-1,a,m-1)-2f(c,c-b-1,a,m-1)-f(a,b,c,n)\qquad else\\
\end{cases}\right.
$$

##   序列中互质数个数

$$
Comprime(x)=\sum_{d|x}{\mu(d)*cnt_d}
$$

```c++
const int maxn = 1e5 + 10;
int mu[maxn];
int cnt[maxn];
void init()
{
    for (int i = 1; i < maxn; ++i)
    {
        for (int j = i; j < maxn; j += i)
            d[j].pb(i);
        if (i == 1)
            mu[i] = 1;
        else if (i % (1ll * d[i][1] * d[i][1]) == 0)
            mu[i] = 0;
        else
            mu[i] = -mu[i / d[i][1]];
    }
}

void update(int x, int v)
{
    for(int k:d[x])
        cnt[k] += v;
}
 
int cal(int x)
{
    int res = 0;
    for(int y:d[x])
        res += mu[y] * cnt[y];
    return res;
}
```

##  积性函数

$$
\epsilon(x)=\sum_{d|x}\mu(d)\\
\epsilon(x)=\left.\begin{cases}
1\qquad x=1\\
0\qquad else
\end{cases}\right.\\
ID(x)=\sum_{d|x}\phi(d)=x\\
\phi(x)=\sum_{d|x}ID(d)\mu(\frac{x}{d})=\sum_{d|x}d\mu(\frac{x}{d})\\
$$

###  欧拉筛积性函数

$$
t[i]表示i的最小质因子出现的次数，因为欧拉筛始终用最小质因子来筛\\当i\%prime[j]=0时prime[j]自然为i的最小质因子，也是i*prime[j]的最小质因子\\
当i\%prime[j]\ne 0时prime[j]是i*prime[j]的最小质因子
$$



```c++
const int maxn = 1e5 + 10;
int prime[maxn],tot;
bool notprime[maxn];
int mu[maxn];
int t[maxn];
ll d[maxn];
void init()
{
    mu[1] = 1;
    d[1] = 1;
    for (ll i = 2; i < maxn; ++i)
    {
        if(!notprime[i])
        {
            prime[tot++] = i;
            mu[i] = -1;
            d[i] = 2;
            t[i] = 1;
        }
        for (int j = 0; j < tot && i * prime[j] < maxn; ++j)
        {
            notprime[i * prime[j]] = 1;
            if (i % prime[j] == 0)
            {
                mu[i * prime[j]] = 0;
                d[i * prime[j]] = d[i] / (t[i] + 1) * (t[i] + 2);
                t[i * prime[j]] = t[i] + 1;
                break;
            }
            d[i * prime[j]] = d[i] * 2;
            mu[i * prime[j]] = -mu[i];
            t[i * prime[j]] = 1;
        }
    }
    for (int i = 1; i < maxn; ++i)
        mu[i] += mu[i - 1], d[i] += d[i - 1];
}
```



##  杜教筛

$$
g(x)为积性函数\\
f(x)=\sum_{i=1}^{x}g(i)\\
S(n)=\sum_{i=1}^{n}\sum_{d|i}g(d)=\sum_{d=1}^{n}\sum_{k=1}^{\left\lfloor\frac{n}{d}\right\rfloor}g(k)=\sum_{d=1}^nf(\left\lfloor\frac{n}{d}\right\rfloor)\\
f(x)=S(x)-\sum_{d=2}^nf(\left\lfloor\frac{n}{d}\right\rfloor)
$$



#  多项式

##  拉格朗日插值

$$
f(k)=\sum_{i=1}^{n}y_i\prod_{j\ne i}\frac{k-x[j]}{x[i]-x[j]}
$$

**n次多项式前缀和是n+1次多项式**

###  模板

$$
已知[f(0),f(n)]求f(k)
$$



```c++
const int maxn = 1e3 + 10;
ll f[maxn];
ll pre[maxn];
ll suf[maxn];
ll fac[maxn], infac[maxn];
void init()
{
    fac[0] = 1;
    for (int i = 1; i < maxn; ++i)
    {
        fac[i] = fac[i - 1] * i % mod;
    }
    infac[maxn - 1] = qmod(fac[maxn - 1], mod - 2);
    for (int i = maxn - 2; i >= 0; --i)
    {
        infac[i] = infac[i + 1] * (i + 1) % mod;
    }
}
ll cal(int n, int k)
{
    pre[0] = 1;
    suf[n] = 1;
    for (int i = 1; i <= n; ++i)
        pre[i] = pre[i - 1] * (k - i + 1) % mod;
    for (int i = n; i >= 1; --i)
        suf[i - 1] = suf[i] * (k - i) % mod;
    ll res = 0;
    for (int i = 0; i <= n; ++i)
    {
        ll ret = f[i] * pre[i] % mod * suf[i] % mod * infac[i] % mod * infac[n - i] % mod;
        if((n-i)&1)
            res = (res - ret + mod) % mod;
        else
            res = (res + ret) % mod;
    }
    return res;
}
```



##  虚数类

```c++
const double PI = acos(-1.0);
struct comp
{
    double r,i;
    comp(double _r = 0.0,double _i = 0.0)
    {
        r = _r; i = _i;
    }
    comp operator +(const comp &b)
    {
        return comp(r+b.r,i+b.i);
    }
    comp operator -(const comp &b)
    {
        return comp(r-b.r,i-b.i);
    }
    comp operator *(const comp &b)
    {
        return comp(r*b.r-i*b.i,r*b.i+i*b.r);
    }
};
```



##  快速傅里叶变换(FFT)

```c++
const int maxn = 4e5 + 10;
comp a[maxn], b[maxn];
char s1[maxn], s2[maxn];
int ans[maxn];
int r[maxn];
void init(int len)
{
    int l = 0;
    while ((1 << l) < len)
        ++l;
    for (int i = 0; i < len; ++i)
    {
        r[i] = (r[i >> 1] >> 1) | ((i & 1) << (l - 1));
    }
}
void change(comp y[],int len)
{
    for (int i = 0; i < len; ++i)
    {
        if(i<r[i])
            swap(y[i], y[r[i]]);
    }
}

void DFT(comp y[],int len,int rev)//rev=1为DFT；-1为IDFT，逆变换；
{
    change(y,len);
    for(int h = 2; h <= len; h <<= 1)
    {
        comp wn(cos(2 * PI / h), sin(rev * 2 * PI / h));
        for (int j = 0; j < len; j += h)
        {
            comp w(1,0);
            for (int k = j; k < j + h / 2; k++)
            {
                comp u = y[k];
                comp t = w*y[k+h/2];
                y[k] = u+t;
                y[k+h/2] = u-t;
                w = w * wn;
            }
        }
    }
    if (rev == -1)
        for(int i = 0;i < len;i++)
            y[i].r /= len;
}
```

##  快速数论变换(NTT)

```c++
void NTT(ll y[],int len,int rev)
{
    change(y,len);
    for(int h = 2; h <= len; h <<= 1)
    {
        ll wn = qmod(3, (mod - 1) / h);
        for (int j = 0; j < len; j += h)
        {
            ll w = 1;
            for (int k = j; k < j + h / 2; k++)
            {
                ll u = y[k];
                ll t = (w * y[k + h / 2]) % mod;
                y[k] = (u + t) % mod;
                y[k + h / 2] = (u - t + mod) % mod;
                w = (w * wn) % mod;
            }
        }
    }
    if (rev == -1)
    {
        reverse(y + 1, y + len);
        ll inv = qmod(len, mod - 2);
        for(int i = 0;i < len;i++)
            (y[i] *= inv) %= mod;
    }
}
```

# 字符串

##  后缀数组

```c++
const int maxn = 1e5 + 10;
char s[maxn];
int x[maxn], y[maxn], sa[maxn], c[maxn], height[maxn], rak[maxn];
int st[maxn][20];
int n, m;
inline void getsa()
{
    int *xx=x,*yy=y;
	memset(c, 0, sizeof c);
	for (int i = 1; i <= n; ++i)++c[xx[i] = s[i]];
	for (int i = 2; i <= m; ++i)c[i] += c[i - 1];
	for (int i = n; i >= 1; --i)sa[c[xx[i]]--] = i;
	for (int k = 1; k <= n; k <<= 1)
	{
		int num = 0;
		for (int i = n - k + 1; i <= n; ++i)yy[++num] = i;
		for (int i = 1; i <= n; ++i)if (sa[i] > k)yy[++num] = sa[i] - k;
		for (int i = 1; i <= m; ++i)c[i] = 0;
		for (int i = 1; i <= n; ++i)++c[xx[i]];
		for (int i = 2; i <= m; ++i)c[i] += c[i - 1];
		for (int i = n; i >= 1; --i)sa[c[xx[yy[i]]]--] = yy[i], yy[i] = 0;//sort
		swap(xx, yy);
		xx[sa[1]] = 1; num = 1;
		for (int i = 2; i <= n; ++i)
			xx[sa[i]] = (yy[sa[i]] == yy[sa[i - 1]] && yy[sa[i] + k] == yy[sa[i - 1] + k]) ? num : ++num;
		if (num == n)break;
		m = num;
	}
}

inline void getheight()
{
	int k = 0;
	for (int i = 1; i <= n; ++i)rak[sa[i]] = i;
	for (int i = 1; i <= n; ++i)
	{
		if (rak[i] == 1)continue;
		if (k)--k;
		int j = sa[rak[i] - 1];
		while (j + k <= n && i + k <= n && s[i + k] == s[j + k])++k;
		height[rak[i]] = k;
	}
}

inline void buildst(int n)
{
	for (int i = 1; i <= n; ++i)st[i][0] = height[i];
	for (int k = 1; k < 20; ++k)
	{
		for (int i = 1; i + (1 << k) - 1 <= n; ++i)
			st[i][k] = min(st[i][k - 1], st[i + (1 << k - 1)][k - 1]);
	}
}

inline int lcp(int x, int y)
{
	int l = rak[x], r = rak[y];
	if (l > r)swap(l, r);
	if (l == r)return n - sa[l] + 1;
	int t = log2(r - l);
	return min(st[l + 1][t], st[r - (1 << t) + 1][t]);
}
```

##  后缀自动机(SAM)

```c++
struct NODE
{
	int ch[26];
	int len, fa;
	NODE() { memset(ch, 0, sizeof(ch)); len = 0; }
}node[MAXN << 1];
int las = 1, tot = 1;
void add(int c)
{
	int p = las; int np = las = ++tot;
	node[np].len = node[p].len + 1;
	for (; p && !node[p].ch[c]; p = node[p].fa)node[p].ch[c] = np;
	if (!p)node[np].fa = 1;//以上为case 1
	else
	{
		int q = node[p].ch[c];
		if (node[q].len == node[p].len + 1)node[np].fa = q;//以上为case 2
		else
		{
			int nq = ++tot; node[nq] = node[q];
			node[nq].len = node[p].len + 1;
			node[q].fa = node[np].fa = nq;
			for (; p && node[p].ch[c] == q; p = node[p].fa)node[p].ch[c] = nq;//以上为case 3
		}
	}
}
/*struct SAM
{
	int ch[maxn << 1][26];
	int fa[maxn << 1], len[maxn << 1];
	int las = 1, tot = 1;
	void add(int c)
	{
		int p = las; int np = las = ++tot;
		len[np] = len[p] + 1;
		for (; p && !ch[p][c]; p = fa[p])ch[p][c] = np;
		if (!p)fa[np] = 1;
		else
		{
			int q = ch[p][c];
			if (len[q] == len[p] + 1)fa[np] = q;
			else
			{
				int nq = ++tot;
				memcpy(ch[nq], ch[q], sizeof ch[nq]);
				fa[nq] = fa[q];
				len[nq] = len[p] + 1;
				fa[q] = fa[np] = nq;
				for (; p && ch[p][c] == q; p = fa[p])ch[p][c] = nq;
			}
		}
	}
	int c[maxn << 1], b[maxn << 1];
	inline void sort()
	{
		for (int i = 1; i <= tot; ++i)++c[len[i]];
		for (int i = 1; i <= tot; ++i)c[i] += c[i - 1];
		for (int i = 1; i <= tot; ++i)b[c[len[i]]--] = i;
	}
};*/
char s[MAXN]; int len;
int main()
{
	scanf("%s", s); len = strlen(s);
	for (int i = 0; i < len; i++)add(s[i] - 'a');
}
```

## AC自动机

```c++
const int maxn = 2e4 + 10;
const int maxm = 1e6 + 10;
int trie[maxn << 1][26];
int fail[maxn << 1];
int cntword[maxn << 1];
int ans[maxn];
int cnt = 0;
char s[200][100], t[maxm];
void insertwords(char str[], int idx)
{
	int u = 0;
	for (int i = 0; str[i]; ++i)
	{
		int v = str[i] - 'a';
		if (!trie[u][v])
			trie[u][v] = ++cnt;
		u = trie[u][v];
	}
	cntword[u] = idx;
}

void getfail()
{
	queue<int> q;
	for (int i = 0; i < 26; ++i)if (trie[0][i])
	{
		fail[trie[0][i]] = 0;
		q.push(trie[0][i]);
	}
	while (!q.empty())
	{
		int u = q.front();
		q.pop();
		for (int i = 0; i < 26; ++i)
		{
			if (trie[u][i])
			{
				fail[trie[u][i]] = trie[fail[u]][i];
				q.push(trie[u][i]);
			}
			else
				trie[u][i] = trie[fail[u]][i];
		}
	}
}
int maxx = 0;
int query(char str[])
{
	int u = 0, res = 0;
	for (int i = 0; str[i]; ++i)
	{
		u = trie[u][str[i] - 'a'];
		for (int j = u; j; j = fail[j])
		{
			++ans[cntword[j]];
			if (cntword[j])
				maxx = max(maxx, ans[cntword[j]]);
		}
	}
	return res;
}

int main()
{
	int n;
	while (scanf("%d", &n), n)
	{
		cnt = 0;
		maxx = 0;
		memset(cntword, 0, sizeof cntword);
		memset(trie, 0, sizeof trie);
		memset(fail, 0, sizeof fail);
		memset(ans, 0, sizeof ans);
		for (int i = 1; i <= n; ++i)
		{
			scanf("%s", s[i]);
			insertwords(s[i], i);
		}
		scanf("%s", t);
		fail[0] = 0;
		getfail();
		query(t);
		printf("%d\n", maxx);
		for (int i = 1; i <= n; ++i)if (ans[i] == maxx)
			printf("%s\n", s[i]);
	}
}
```

## 最小表示法

```C++
int zxbsf(string s)
{
	int len = s.length() - 1;
	s.append(s);
	int i = 0, j = 1, k;
	while (i <= len && j <= len)
	{
		for (k = 0; k < len && s[i + k] == s[j + k]; k++);
		if (k == len)break;
		if (s[i + k] > s[j + k])
		{
			i = i + k + 1;
			if (i == j)i++;
		}
		else
		{
			j = j + k + 1;
			if (i == j)j++;
		}
	}
	return min(i, j);
}
```

#  图论

## 网络流

###  Dinic

```c++
const int maxm = 1e5 + 10;
const int maxn = 1e4 + 10;
struct EDGE
{
	int next, to, w;
}edge[maxm << 1];
int head[maxn], tot;
int cur[maxn];
int n, m;
inline void addedge(int u, int v, int w)
{
	edge[tot] = { head[u],v,w };
	head[u] = tot++;
	edge[tot] = { head[v],u,0 };
	head[v] = tot++;
}

int dep[maxn];

bool bfs(int s, int t)
{
	memset(dep, 0x3f, sizeof dep);
	queue<int> q;
	for (int i = 1; i <= n; ++i)cur[i] = head[i];
	dep[s] = 0;
	q.push(s);
	while (!q.empty())
	{
		int u = q.front();
		q.pop();
		for (int i = head[u]; ~i; i = edge[i].next)
		{
			int v = edge[i].to;
			if (dep[v] == inf && edge[i].w)
			{
				dep[v] = dep[u] + 1;
				q.push(v);
			}
		}
	}
	return dep[t] < inf;
}
int dfs(int u, int t, int limit)
{
	if (!limit || u == t)return limit;
	int flow = 0, f;
	for (int i = cur[u]; ~i; i = edge[i].next)
	{
		cur[u] = i;
		int v = edge[i].to;
		if (dep[v] == dep[u] + 1 && (f = dfs(v, t, min(limit, edge[i].w))))
		{
			flow += f;
			limit -= f;
			edge[i].w -= f;
			edge[i ^ 1].w += f;
			if (!limit)break;
		}
	}
	return flow;
}

ll Dinic(int s, int t)
{
	ll maxflow = 0;
	while (bfs(s, t))
		maxflow += dfs(s, t, inf);
	return maxflow;
}

```

##  最小费用最大流

###   spfa+EK

```c++
bool vis[maxn];
int dis[maxn], flow[maxn], pre[maxn], last[maxn];
bool spfa(int s, int t)
{
    queue<int> q;
    memset(dis, 0x3f, sizeof dis);
    memset(flow, 0x3f, sizeof flow);
    memset(vis, 0, sizeof vis);
    q.push(s);
    vis[s] = 1;
    dis[s] = 0;
    pre[t] = -1;
    while (!q.empty())
    {
        int u = q.front();
        q.pop();
        vis[u] = 0;
        for (int i = head[u]; ~i; i = edge[i].next)
        {
            int v = edge[i].to;
            if (edge[i].w > 0 && dis[v] > dis[u] + edge[i].c)
            {
                dis[v] = dis[u] + edge[i].c;
                pre[v] = u;
                last[v] = i;
                flow[v] = min(flow[u], edge[i].w);
                if(!vis[v])
                {
                    q.push(v);
                    vis[v] = 1;
                }
            }
        }
    }
    return pre[t] != -1;
}
ll maxflow, maxcost;
void mcmf(int s, int t)
{
    maxflow = 0, maxcost = 0;
    while(spfa(s,t))
    {
        int u=t;
        maxflow += flow[t];
        maxcost += flow[t] * dis[t];
        while (u != s)
        {
            edge[last[u]].w -= flow[t];
            edge[last[u] ^ 1].w += flow[t];
            u = pre[u];
        }
    }
}
```

## 启发式搜索(K短路/A*)

```c++
const int maxn = 1e5 + 10;
int tot1 = 0, tot2 = 0;
int head1[1010], head2[1010];
int dis[1010];
bool vis[1010];
struct Edge
{
	int next, to, w;
	bool operator < (const Edge& other)const
	{
		return w + dis[to] > other.w + dis[other.to];
	}
} edge1[maxn], edge2[maxn];
struct cmp
{
	bool operator () (const Edge& a, const Edge& b)
	{
		return a.w > b.w;
	}
};
inline void addedge1(int u, int v, int w)
{
	edge1[tot1] = { head1[u],v,w };
	head1[u] = tot1++;
}
inline void addedge2(int u, int v, int w)
{
	edge2[tot2] = { head2[u],v,w };
	head2[u] = tot2++;
}
void dijk(int s, int t)
{
	memset(dis, 0x3f, sizeof dis);
	dis[s] = 0;
	priority_queue<Edge, vector<Edge>, cmp > q;
	for (int i = head2[s]; ~i; i = edge2[i].next)
	{
		int v = edge2[i].to;
		dis[v] = min(dis[v], edge2[i].w);
		q.push(edge2[i]);
	}
	Edge tmp;
	while (!q.empty())
	{
		tmp = q.top();
		q.pop();
		int u = tmp.to;
		vis[u] = true;
		for (int i = head2[u]; ~i; i = edge2[i].next)
		{
			int v = edge2[i].to, w = edge2[i].w;
			if (vis[v])continue;
			if (dis[v] > dis[u] + w)
			{
				dis[v] = dis[u] + w;
				q.push(Edge{ 0,v,dis[u] + w });
			}
		}
	}
}

int Astar(int s, int t, int k)
{
	int cnt = 0;
	priority_queue<Edge> q;
	for (int i = head1[s]; ~i; i = edge1[i].next)
	{
		q.push(edge1[i]);
	}
	Edge tmp;
	while (!q.empty())
	{
		tmp = q.top();
		q.pop();
		if (tmp.to == t)
		{
			++cnt;
			if (cnt == k)
				return tmp.w;
		}
		int u = tmp.to;
		int w = tmp.w;
		for (int i = head1[u]; ~i; i = edge1[i].next)
		{
			q.push(Edge{ 0,edge1[i].to,w + edge1[i].w });
		}
	}
	return -1;
}

int main()
{
	int n, m;
	int u, v, w;
	int s, t, k;
	memset(head1, -1, sizeof head1);
	memset(head2, -1, sizeof head2);
	read(n), read(m);
	for (int i = 0; i < m; ++i)
	{
		read(u), read(v), read(w);
		addedge1(u, v, w);
		addedge2(v, u, w);
	}
	read(s), read(t), read(k);
	dijk(t, s);
	printf("%d\n", Astar(s, t, k));
}

```

#  计算几何

## 三点定圆

```c++
const double eps = 1e-6;
struct Point {
	double x, y;
	Point(double _x = 0, double _y = 0)
	{
		x = _x;
		y = _y;
	}
	Point operator-(const Point& p1)const
	{
		return Point(x - p1.x, y - p1.y);
	}
	Point operator+(const Point& p1)const
	{
		return Point(x + p1.x, y + p1.y);
	}
	bool friend operator<(const Point& p1, const Point& p2)
	{
		if (p1.x != p2.y)return p1.x < p2.x;
		return p1.y < p2.y;
	}
	bool friend operator==(const Point& p1, const Point& p2)
	{
		if (fabs(p1.x - p2.x) > eps)return 0;
		if (fabs(p1.y - p2.y) > eps)return 0;
		return 1;
	}
	double friend operator^(const Point& p1, const Point& p2)
	{
		return p1.x * p2.y - p2.x * p1.y;
	}
	Point friend operator*(const Point& p1, const double& k)
	{
		return Point(p1.x * k, p1.y * k);
	}
	double friend operator*(const Point& p1, const Point& p2)
	{
		return p1.x * p2.x + p1.y * p2.y;
	}
	void transXY(double B)//绕原点旋转B弧度 
	{
		double tx = x, ty = y;
		x = tx * cos(B) - ty * sin(B);
		y = tx * sin(B) + ty * cos(B);
	}
}aa[maxn];
double dis(const Point& p1, const Point& p2)
{
	return sqrt((p1 - p2) * (p1 - p2));
}
bool in_row(const Point& p1, const Point& p2, const Point& p3)//判断三点是否共线 
{
	return fabs((p1.x - p2.x) * (p2.y - p3.y) - (p2.x - p3.x) * (p1.y - p2.y)) <= eps;
}
Point GetCenter(const Point& p1, const Point& p2, const Point& p3)//求三点圆心 （三角形外心） 
{
	Point cp;
	double a1 = p2.x - p1.x, b1 = p2.y - p1.y, c1 = (a1 * a1 + b1 * b1) / 2;
	double a2 = p3.x - p1.x, b2 = p3.y - p1.y, c2 = (a2 * a2 + b2 * b2) / 2;
	double d = a1 * b2 - a2 * b1;
	cp.x = p1.x + (c1 * b2 - c2 * b1) / d;
	cp.y = p1.y + (a1 * c2 - a2 * c1) / d;
	return cp;
}

```

# 其它

## 关于括号匹配

1.  取**‘（’**为1，**‘）’**为-1，若序列每个前缀和均非负，切总和为0，则括号可成功匹配
2.  接上条，对于总和为0的匹配序列，序列中前缀和最小值的个数即为循环匹配的个数