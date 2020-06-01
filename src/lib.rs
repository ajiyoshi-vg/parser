pub struct Source<T = char> {
    s: Vec<T>,
    pos: usize,
}

pub fn source(s: &str) -> Source<char> {
    Source::from(s.to_string().chars().collect())
}

impl<T: Clone> Source<T> {
    pub fn from(s: Vec<T>) -> Self {
        Source { s: s, pos: 0 }
    }

    pub fn peek(&self) -> Option<T> {
        self.s.get(self.pos).map(|x| x.clone())
    }

    pub fn ahead(&mut self) {
        self.pos += 1;
    }

    pub fn pop(&mut self) -> Option<T> {
        let ret = self.peek();
        self.ahead();
        ret
    }

    pub fn save(&self) -> usize {
        self.pos
    }

    pub fn restore(&mut self, pos: usize) {
        self.pos = pos;
    }

    pub fn is_finished(&self) -> bool {
        self.s.get(self.pos).is_none()
    }
}

pub struct Parser<T> {
    body: Box<dyn Fn(&mut Source) -> Option<T>>,
}

impl<T: 'static> Parser<T> {
    pub fn new<F>(f: F) -> Self
    where
        F: Fn(&mut Source) -> Option<T> + 'static,
    {
        Parser { body: Box::new(f) }
    }

    pub fn many(self) -> Parser<Vec<T>> {
        many(self)
    }
    pub fn or(self, it: Self) -> Self {
        or(self, it)
    }
    pub fn next<U: 'static>(self, it: Parser<U>) -> Parser<U> {
        next(self, it)
    }
    pub fn prev<U: 'static>(self, it: Parser<U>) -> Self {
        prev(self, it)
    }
    pub fn repeat(self, n: usize) -> Parser<Vec<T>> {
        repeat(n, self)
    }
    pub fn tryp(self) -> Self {
        tryp(self)
    }
    pub fn apply<U, F>(self, f: F) -> Parser<U>
    where
        U: 'static,
        F: 'static + Fn(T) -> U,
    {
        apply(self, f)
    }

    pub fn then(self, them: Parser<Vec<T>>) -> Parser<Vec<T>> {
        then(self, them)
    }
}

pub fn lazy<T, F>(f: F) -> Parser<T>
where
    T: 'static,
    F: Fn() -> Parser<T> + 'static,
{
    parser(move |s| f().parse(s))
}

pub fn parser<T, F>(f: F) -> Parser<T>
where
    T: 'static,
    F: Fn(&mut Source) -> Option<T> + 'static,
{
    Parser::new(f)
}

pub fn many<T: 'static>(p: Parser<T>) -> Parser<Vec<T>> {
    parser(move |s| {
        let mut ret = Vec::new();
        while let Some(x) = p.parse(s) {
            ret.push(x);
        }
        Some(ret)
    })
}

pub fn many1<T: 'static>(p: Parser<T>) -> Parser<Vec<T>> {
    parser(move |s| {
        let mut ret = match p.parse(s) {
            Some(x) => vec![x],
            None => return None,
        };

        while let Some(x) = p.parse(s) {
            ret.push(x);
        }
        Some(ret)
    })
}

pub fn repeat<T: 'static>(n: usize, p: Parser<T>) -> Parser<Vec<T>> {
    parser(move |s| {
        let mut ret = Vec::with_capacity(n);
        for _ in 0..n {
            match p.parse(s) {
                Some(x) => ret.push(x),
                None => return None,
            }
        }
        Some(ret)
    })
    .tryp()
}

pub fn or<T: 'static>(a: Parser<T>, b: Parser<T>) -> Parser<T> {
    let a = a.tryp();
    parser(move |s| a.parse(s).or_else(|| b.parse(s)))
}

pub fn tryp<T: 'static>(p: Parser<T>) -> Parser<T> {
    parser(move |s| {
        let save = s.save();
        p.parse(s).or_else(|| {
            s.restore(save);
            None
        })
    })
}

pub fn next<A: 'static, B: 'static>(a: Parser<A>, b: Parser<B>) -> Parser<B> {
    parser(move |s| a.parse(s).and_then(|_| b.parse(s))).tryp()
}

pub fn prev<A: 'static, B: 'static>(a: Parser<A>, b: Parser<B>) -> Parser<A> {
    parser(move |s| {
        a.parse(s)
            .and_then(|ret| b.parse(s).and_then(|_| Some(ret)))
    })
    .tryp()
}

pub fn apply<A, B, F>(p: Parser<A>, f: F) -> Parser<B>
where
    A: 'static,
    B: 'static,
    F: Fn(A) -> B + 'static,
{
    parser(move |s| p.parse(s).map(|x| f(x)))
}
pub fn apply_option<A, B, F>(p: Parser<A>, f: F) -> Parser<B>
where
    A: 'static,
    B: 'static,
    F: Fn(A) -> Option<B> + 'static,
{
    parser(move |s| p.parse(s).and_then(|x| f(x))).tryp()
}

pub fn then<T: 'static>(head: Parser<T>, tail: Parser<Vec<T>>) -> Parser<Vec<T>> {
    parser(move |s| {
        let mut ret = match head.parse(s) {
            Some(x) => vec![x],
            None => return None,
        };

        tail.parse(s).map(|xs| {
            for x in xs {
                ret.push(x);
            }
            ret
        })
    })
    .tryp()
}

impl<T> Parser<T> {
    pub fn parse(&self, s: &mut Source) -> Option<T> {
        (self.body)(s)
    }
}

impl Iterator for Source {
    type Item = char;

    fn next(&mut self) -> Option<Self::Item> {
        self.pop()
    }
}

pub fn satisfy<F>(f: F) -> Parser<char>
where
    F: Fn(char) -> bool + 'static,
{
    /*
    parser(move |s| s.pop().filter(|x| f(x.to_owned())))
        .tryp()
    と等価だと思うけどベタにやった方が速い気がするので
    */
    parser(move |s| match s.peek() {
        Some(c) => {
            if f(c) {
                s.pop()
            } else {
                None
            }
        }
        None => None,
    })
}

pub fn any_char() -> Parser<char> {
    satisfy(|_| true)
}

pub fn char1(x: char) -> Parser<char> {
    satisfy(move |c| c == x)
}

pub fn digit() -> Parser<char> {
    satisfy(|c| c.is_digit(10))
}

pub fn number() -> Parser<String> {
    many1(digit()).apply(|x| x.into_iter().collect())
}

pub fn int<T>() -> Parser<T>
where
    T: std::str::FromStr + 'static,
{
    apply_option(number(), |x| maybe(x.parse()))
}

fn maybe<T, E>(x: Result<T, E>) -> Option<T> {
    match x {
        Ok(ret) => Some(ret),
        Err(_) => None,
    }
}

pub fn sequence<T>(ps: Vec<Parser<T>>) -> Parser<Vec<T>>
where
    T: 'static,
{
    parser(move |s| {
        let mut tmp = Vec::new();
        for p in ps.iter() {
            match p.parse(s) {
                Some(x) => tmp.push(x),
                None => return None,
            }
        }
        Some(tmp)
    })
    .tryp()
}

pub fn string(st: &str) -> Parser<String> {
    // Vec<Parser<char>> を作って
    let ps = st.chars().map(|c| char1(c)).collect();
    // Parser<Vec<char>> にして
    sequence(ps)
        // パース結果の Vec<char> から String を作る
        .apply(|v| v.iter().collect())
}

pub fn space() -> Parser<char> {
    satisfy(|c| c.is_ascii_whitespace())
}

pub fn spaces() -> Parser<Vec<char>> {
    space().many()
}

pub fn sandbox() {
    assert!(true);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_source() {
        let mut s = source("abc");
        assert_eq!(s.next(), Some('a'));
        assert_eq!(s.next(), Some('b'));
        assert_eq!(s.next(), Some('c'));
        assert_eq!(s.next(), None);
        assert_eq!(s.next(), None);
    }

    #[test]
    fn test_parse() {
        let mut s = source("abc");
        let any_char = any_char();
        assert_eq!(any_char.parse(&mut s), Some('a'));
        assert_eq!(any_char.parse(&mut s), Some('b'));
        assert_eq!(any_char.parse(&mut s), Some('c'));
        assert_eq!(any_char.parse(&mut s), None);
    }

    #[test]
    fn test_char1() {
        let mut s = source("xyz");
        let is_x = char1('x');
        let any_char = any_char();
        assert_eq!(is_x.parse(&mut s), Some('x'));
        assert_eq!(is_x.parse(&mut s), None);
        assert_eq!(any_char.parse(&mut s), Some('y'));
        assert_eq!(any_char.parse(&mut s), Some('z'));
        assert_eq!(any_char.parse(&mut s), None);
    }

    #[test]
    fn test_many() {
        let mut s = source("123x");
        assert_eq!(many(digit()).parse(&mut s), Some(vec!['1', '2', '3']));
        assert_eq!(any_char().parse(&mut s), Some('x'));

        let mut s = source("123x");
        assert_eq!(digit().many().parse(&mut s), Some(vec!['1', '2', '3']));
        assert_eq!(any_char().parse(&mut s), Some('x'));

        let p = many(char1('a').or(char1('b')));
        let mut s = source("abaabbbbabc");
        assert!(p.parse(&mut s).is_some());
        assert_eq!(p.parse(&mut s), Some(vec![]));
        assert_eq!(any_char().parse(&mut s), Some('c'));
    }

    #[test]
    fn test_repeat() {
        let p4 = digit().repeat(4);
        let p3 = digit().repeat(3);
        let mut s = source("123x");
        assert_eq!(p4.parse(&mut s), None); // repeat途中で失敗したら何も消費しない
        assert_eq!(p3.parse(&mut s), Some(vec!['1', '2', '3']));
        assert_eq!(p3.parse(&mut s), None);
    }

    #[test]
    fn test_or() {
        let p = or(char1('a'), char1('b'));
        let mut s = source("abc");
        assert_eq!(p.parse(&mut s), Some('a'));
        assert_eq!(p.parse(&mut s), Some('b'));
        assert_eq!(p.parse(&mut s), None);
        assert!(!s.is_finished());

        let p = char1('a').or(char1('b'));
        let mut s = source("abc");
        assert_eq!(p.parse(&mut s), Some('a'));
        assert_eq!(p.parse(&mut s), Some('b'));
        assert_eq!(p.parse(&mut s), None);
        assert!(!s.is_finished());
    }

    #[test]
    fn test_string() {
        let mut s = source("if cond then 1 else 2");
        assert_eq!(string("if cond {").parse(&mut s), None);
        assert_eq!(
            string("if cond ").parse(&mut s),
            Some("if cond ".to_string())
        );
        assert_eq!(string("then").parse(&mut s), Some("then".to_string()));
    }

    #[test]
    fn test_number() {
        let mut s = source("123x");
        assert_eq!(number().parse(&mut s), Some("123".to_string()));
        assert_eq!(any_char().parse(&mut s), Some('x'));
        assert!(s.is_finished());

        let mut s = source("123x");
        assert_eq!(int().parse(&mut s), Some(123));
        assert_eq!(any_char().parse(&mut s), Some('x'));
        assert!(s.is_finished());

        let mut s = source("abc");
        assert_eq!(number().parse(&mut s), None);
        assert!(string("abc").parse(&mut s).is_some()); // ↑のパース失敗で何も消費されない

        let mut s = source("abc");
        assert_eq!(int().parse(&mut s), None as Option<i32>);
        assert!(string("abc").parse(&mut s).is_some()); // ↑のパース失敗で何も消費されない
    }

    #[test]
    fn test_next_prev() {
        let p = char1('+').next(number());
        let mut s = source("+123");
        assert_eq!(p.parse(&mut s), Some("123".to_string()));
        assert!(s.is_finished());
        assert_eq!(p.parse(&mut source("123")), None);

        let p = number().prev(char1(','));
        let mut s = source("123,");
        assert_eq!(p.parse(&mut s), Some("123".to_string()));
        assert!(s.is_finished());
        assert_eq!(p.parse(&mut source("123")), None);

        let p = char1('+').next(number()).prev(char1(','));
        let mut s = source("+123,");
        assert_eq!(p.parse(&mut s), Some("123".to_string()));
        assert!(s.is_finished());
        assert_eq!(p.parse(&mut source("+123")), None);
        assert_eq!(p.parse(&mut source("123,")), None);

        let p = char1('+').next(int()).prev(char1(','));
        let mut s = source("+123,");
        assert_eq!(p.parse(&mut s), Some(123));
        assert!(s.is_finished());
        assert_eq!(p.parse(&mut source("+123")), None);
        assert_eq!(p.parse(&mut source("123,")), None);
    }

    #[test]
    fn test_expr() {
        let mut s = source("1+2+3");
        assert_eq!(expr().parse(&mut s).map(|x| eval(x)), Some(6));
        assert_eq!(
            expr().parse(&mut source("1-(2+3)")).map(|x| eval(x)),
            Some(-4)
        );
        assert_eq!(
            expr().parse(&mut source("2*(1+3)")).map(|x| eval(x)),
            Some(8)
        );
        assert_eq!(
            expr().parse(&mut source("-2*(1+3)")).map(|x| eval(x)),
            Some(-8)
        );
        assert_eq!(
            expr().parse(&mut source("2 + 3 * 4")).map(|x| eval(x)),
            Some(14)
        );
        assert_eq!(
            expr().parse(&mut source("100 / 10 / 2")).map(|x| eval(x)),
            Some(5)
        );
        assert_eq!(
            expr().parse(&mut source("100 / (10 / 2)")).map(|x| eval(x)),
            Some(20)
        );
        assert_eq!(expr().parse(&mut source("-1")).map(|x| eval(x)), Some(-1));
    }

    // パーサコンビネータを使った数式構文解析
    enum Expr {
        Const(i32),
        Add(Box<Expr>, Box<Expr>),
        Sub(Box<Expr>, Box<Expr>),
        Mul(Box<Expr>, Box<Expr>),
        Div(Box<Expr>, Box<Expr>),
    }

    enum Op {
        Add,
        Sub,
        Mul,
        Div,
    }

    fn eval(exp: Expr) -> i32 {
        match exp {
            Expr::Const(n) => n,
            Expr::Add(a, b) => eval(*a) + eval(*b),
            Expr::Sub(a, b) => eval(*a) - eval(*b),
            Expr::Mul(a, b) => eval(*a) * eval(*b),
            Expr::Div(a, b) => eval(*a) / eval(*b),
        }
    }

    fn expr() -> Parser<Expr> {
        // 1+2+3
        // number ( [+|-] number )*
        term()
            .apply(|x| (Op::Add, x))
            .then(many(or(
                char1('+').next(term()).apply(|x| (Op::Add, x)),
                char1('-').next(term()).apply(|x| (Op::Sub, x)),
            )))
            .apply(|ts| {
                let mut ret = Expr::Const(0);
                for t in ts {
                    ret = match t.0 {
                        Op::Add => Expr::Add(Box::new(ret), Box::new(t.1)),
                        Op::Sub => Expr::Sub(Box::new(ret), Box::new(t.1)),
                        _ => ret,
                    }
                }
                ret
            })
    }

    fn term() -> Parser<Expr> {
        factor()
            .apply(|x| (Op::Mul, x))
            .then(many(or(
                char1('*').next(factor()).apply(|x| (Op::Mul, x)),
                char1('/').next(factor()).apply(|x| (Op::Div, x)),
            )))
            .apply(|ts| {
                let mut ret = Expr::Const(1);
                for t in ts {
                    ret = match t.0 {
                        Op::Mul => Expr::Mul(Box::new(ret), Box::new(t.1)),
                        Op::Div => Expr::Div(Box::new(ret), Box::new(t.1)),
                        _ => ret,
                    }
                }
                ret
            })
    }

    fn factor() -> Parser<Expr> {
        spaces()
            .next(or(
                char1('(').next(lazy(|| expr())).prev(char1(')')),
                num_expr(),
            ))
            .prev(spaces())
    }

    fn num_expr() -> Parser<Expr> {
        or(
            int().apply(|n: i32| Expr::Const(n)),
            char1('-').next(int().apply(|n: i32| Expr::Const(-n))),
        )
    }

    #[test]
    fn test_sandbox() {
        sandbox();
    }
}
