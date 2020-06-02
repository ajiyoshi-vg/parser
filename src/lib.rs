pub struct Source<T = char> {
    s: Vec<T>,
    pos: usize,
}

pub fn source(s: &str) -> Source<char> {
    Source::from(s.to_string().chars().collect())
}

impl<T: Clone> Source<T> {
    pub fn from(s: Vec<T>) -> Self {
        Source { s, pos: 0 }
    }

    pub fn peek(&self) -> Option<T> {
        self.s.get(self.pos).cloned()
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

#[derive(Debug, PartialEq)]
pub enum ParseError {
    Unknown,
}

pub struct Parser<T> {
    body: Box<dyn Fn(&mut Source) -> Result<T, ParseError>>,
}

impl<T: 'static> Parser<T> {
    pub fn new<F>(f: F) -> Self
    where
        F: Fn(&mut Source) -> Result<T, ParseError> + 'static,
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
    F: Fn(&mut Source) -> Result<T, ParseError> + 'static,
{
    Parser::new(f)
}

pub fn many<T: 'static>(p: Parser<T>) -> Parser<Vec<T>> {
    parser(move |s| {
        let mut ret = Vec::new();
        while let Ok(x) = p.parse(s) {
            ret.push(x);
        }
        Ok(ret)
    })
}

pub fn many1<T: 'static>(p: Parser<T>) -> Parser<Vec<T>> {
    parser(move |s| {
        let mut ret = vec![p.parse(s)?];
        while let Ok(x) = p.parse(s) {
            ret.push(x);
        }
        Ok(ret)
    })
}

pub fn repeat<T: 'static>(n: usize, p: Parser<T>) -> Parser<Vec<T>> {
    parser(move |s| {
        let mut ret = Vec::with_capacity(n);
        for _ in 0..n {
            ret.push(p.parse(s)?);
        }
        Ok(ret)
    })
    .tryp()
}

pub fn or<T: 'static>(a: Parser<T>, b: Parser<T>) -> Parser<T> {
    let a = a.tryp();
    parser(move |s| a.parse(s).or_else(|_| b.parse(s)))
}

pub fn tryp<T: 'static>(p: Parser<T>) -> Parser<T> {
    parser(move |s| {
        let save = s.save();
        p.parse(s).or_else(|e| {
            s.restore(save);
            Err(e)
        })
    })
}

pub fn next<A: 'static, B: 'static>(a: Parser<A>, b: Parser<B>) -> Parser<B> {
    parser(move |s| a.parse(s).and_then(|_| b.parse(s))).tryp()
}

pub fn prev<A: 'static, B: 'static>(a: Parser<A>, b: Parser<B>) -> Parser<A> {
    parser(move |s| a.parse(s).and_then(|ret| b.parse(s).and_then(|_| Ok(ret)))).tryp()
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
    F: Fn(A) -> Result<B, ParseError> + 'static,
{
    parser(move |s| p.parse(s).and_then(|x| f(x))).tryp()
}

pub fn then<T: 'static>(head: Parser<T>, tail: Parser<Vec<T>>) -> Parser<Vec<T>> {
    parser(move |s| {
        let mut ret = vec![head.parse(s)?];
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
    pub fn parse(&self, s: &mut Source) -> Result<T, ParseError> {
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
    F: Fn(&char) -> bool + 'static,
{
    parser(move |s| {
        s.peek()
            .filter(|x| f(x))
            .map_or(Err(ParseError::Unknown), |x| {
                s.ahead();
                Ok(x)
            })
    })
}

pub fn any_char() -> Parser<char> {
    satisfy(|_| true)
}

pub fn char1(x: char) -> Parser<char> {
    satisfy(move |c| x.eq(c))
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
    apply_option(number(), |x| x.parse().map_err(|_| ParseError::Unknown))
}

pub fn sequence<T>(ps: Vec<Parser<T>>) -> Parser<Vec<T>>
where
    T: 'static,
{
    parser(move |s| {
        let mut tmp = Vec::new();
        for p in ps.iter() {
            tmp.push(p.parse(s)?);
        }
        Ok(tmp)
    })
    .tryp()
}

pub fn string(st: &str) -> Parser<String> {
    // Vec<Parser<char>> を作って
    let ps = st.chars().map(char1).collect();
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
        assert_eq!(any_char.parse(&mut s).unwrap(), 'a');
        assert_eq!(any_char.parse(&mut s).unwrap(), 'b');
        assert_eq!(any_char.parse(&mut s).unwrap(), 'c');
        assert!(any_char.parse(&mut s).is_err());
    }

    #[test]
    fn test_char1() {
        let mut s = source("xyz");
        let is_x = char1('x');
        let any_char = any_char();
        assert_eq!(is_x.parse(&mut s).unwrap(), 'x');
        assert!(is_x.parse(&mut s).is_err());
        assert_eq!(any_char.parse(&mut s).unwrap(), 'y');
        assert_eq!(any_char.parse(&mut s).unwrap(), 'z');
        assert!(any_char.parse(&mut s).is_err());
    }

    #[test]
    fn test_many() {
        let mut s = source("123x");
        assert_eq!(many(digit()).parse(&mut s).unwrap(), vec!['1', '2', '3']);
        assert_eq!(any_char().parse(&mut s).unwrap(), 'x');

        let mut s = source("123x");
        assert_eq!(digit().many().parse(&mut s).unwrap(), vec!['1', '2', '3']);
        assert_eq!(any_char().parse(&mut s).unwrap(), 'x');

        let p = many(char1('a').or(char1('b')));
        let mut s = source("abaabbbbabc");
        assert!(p.parse(&mut s).is_ok());
        assert_eq!(p.parse(&mut s).unwrap(), vec![]);
        assert_eq!(any_char().parse(&mut s).unwrap(), 'c');
    }

    #[test]
    fn test_repeat() {
        let p4 = digit().repeat(4);
        let p3 = digit().repeat(3);
        let mut s = source("123x");
        assert!(p4.parse(&mut s).is_err()); // repeat途中で失敗したら何も消費しない
        assert_eq!(p3.parse(&mut s).unwrap(), vec!['1', '2', '3']);
        assert!(p3.parse(&mut s).is_err());
    }

    #[test]
    fn test_or() {
        let p = or(char1('a'), char1('b'));
        let mut s = source("abc");
        assert_eq!(p.parse(&mut s).unwrap(), 'a');
        assert_eq!(p.parse(&mut s).unwrap(), 'b');
        assert!(p.parse(&mut s).is_err());
        assert!(!s.is_finished());

        let p = char1('a').or(char1('b'));
        let mut s = source("abc");
        assert_eq!(p.parse(&mut s).unwrap(), 'a');
        assert_eq!(p.parse(&mut s).unwrap(), 'b');
        assert!(p.parse(&mut s).is_err());
        assert!(!s.is_finished());
    }

    #[test]
    fn test_string() {
        let mut s = source("if cond then 1 else 2");
        assert!(string("if cond {").parse(&mut s).is_err());
        assert_eq!(
            string("if cond ").parse(&mut s).unwrap(),
            "if cond ".to_string()
        );
        assert_eq!(string("then").parse(&mut s).unwrap(), "then".to_string());
    }

    #[test]
    fn test_number() {
        let mut s = source("123x");
        assert_eq!(number().parse(&mut s).unwrap(), "123".to_string());
        assert_eq!(any_char().parse(&mut s).unwrap(), 'x');
        assert!(s.is_finished());

        let mut s = source("123x");
        assert_eq!(int::<i32>().parse(&mut s).unwrap(), 123);
        assert_eq!(any_char().parse(&mut s).unwrap(), 'x');
        assert!(s.is_finished());

        let mut s = source("abc");
        assert!(number().parse(&mut s).is_err());
        assert!(string("abc").parse(&mut s).is_ok()); // ↑のパース失敗で何も消費されない

        let mut s = source("abc");
        assert!(int::<i32>().parse(&mut s).is_err());
        assert!(string("abc").parse(&mut s).is_ok()); // ↑のパース失敗で何も消費されない
    }

    #[test]
    fn test_next_prev() {
        let p = char1('+').next(number());
        let mut s = source("+123");
        assert_eq!(p.parse(&mut s).unwrap(), "123".to_string());
        assert!(s.is_finished());
        assert!(p.parse(&mut source("123")).is_err());
        assert!(p.parse(&mut source("+")).is_err());

        let p = number().prev(char1(','));
        let mut s = source("123,");
        assert_eq!(p.parse(&mut s).unwrap(), "123".to_string());
        assert!(s.is_finished());
        assert!(p.parse(&mut source("123")).is_err());

        let p = char1('+').next(number()).prev(char1(','));
        let mut s = source("+123,");
        assert_eq!(p.parse(&mut s).unwrap(), "123".to_string());
        assert!(s.is_finished());
        assert!(p.parse(&mut source("+123")).is_err());
        assert!(p.parse(&mut source("123,")).is_err());

        let p = char1('+').next(int::<i32>()).prev(char1(','));
        let mut s = source("+123,");
        assert_eq!(p.parse(&mut s).unwrap(), 123);
        assert!(s.is_finished());
        assert!(p.parse(&mut source("+123")).is_err());
        assert!(p.parse(&mut source("123,")).is_err());
    }

    #[test]
    fn test_expr() {
        struct Case {
            name: String,
            src: String,
            expect: i32,
        }
        let cases = vec![
            Case {
                name: "simple".to_string(),
                src: "1+2+3".to_string(),
                expect: 6,
            },
            Case {
                name: "with ()".to_string(),
                src: "1-(2+3)".to_string(),
                expect: -4,
            },
            Case {
                name: "mul".to_string(),
                src: "2*(2+3)".to_string(),
                expect: 10,
            },
            Case {
                name: "unary minus".to_string(),
                src: "-2*(-1+3)".to_string(),
                expect: -4,
            },
            Case {
                name: "unary minus".to_string(),
                src: "-2*(-1+3)".to_string(),
                expect: -4,
            },
            Case {
                name: "spaces".to_string(),
                src: "2 + 3 * 4".to_string(),
                expect: 14,
            },
            Case {
                name: "minus".to_string(),
                src: "2 - 3 * 4".to_string(),
                expect: -10,
            },
            Case {
                name: "div".to_string(),
                src: "100 / 10/2".to_string(),
                expect: 5,
            },
            Case {
                name: "div+paren".to_string(),
                src: "100 / (10 / 2)".to_string(),
                expect: 20,
            },
        ];
        for c in cases {
            assert_eq!(
                expr()
                    .parse(&mut source(c.src.as_str()))
                    .map(|x| eval(x))
                    .expect(c.name.as_str()),
                c.expect
            )
        }

        let cases = vec!["+", "1+"];
        for c in cases {
            let mut s = source(c);
            let _ = expr().parse(&mut s);
            assert!(!s.is_finished());
        }
    }

    // パーサコンビネータを使った数式構文解析
    #[derive(Debug, PartialEq)]
    enum Expr {
        Const(i32),
        BinOp(Op, Box<Expr>, Box<Expr>),
    }

    #[derive(Debug, PartialEq)]
    enum Op {
        Add,
        Sub,
        Mul,
        Div,
    }

    fn eval(exp: Expr) -> i32 {
        match exp {
            Expr::Const(n) => n,
            Expr::BinOp(op, a, b) => match op {
                Op::Add => eval(*a) + eval(*b),
                Op::Sub => eval(*a) - eval(*b),
                Op::Mul => eval(*a) * eval(*b),
                Op::Div => eval(*a) / eval(*b),
            },
        }
    }

    fn expr() -> Parser<Expr> {
        // 1+2+3
        // number ( [+|-] number )*
        apply(term(), |x| (Op::Add, x))
            .then(many(or(
                apply(char1('+').next(term()), |x| (Op::Add, x)),
                apply(char1('-').next(term()), |x| (Op::Sub, x)),
            )))
            .apply(|ts| {
                ts.into_iter().fold(Expr::Const(0), |acc, t| {
                    Expr::BinOp(t.0, Box::new(acc), Box::new(t.1))
                })
            })
    }

    fn term() -> Parser<Expr> {
        apply(factor(), |x| (Op::Mul, x))
            .then(many(or(
                apply(char1('*').next(factor()), |x| (Op::Mul, x)),
                apply(char1('/').next(factor()), |x| (Op::Div, x)),
            )))
            .apply(|ts| {
                ts.into_iter().fold(Expr::Const(1), |acc, t| {
                    Expr::BinOp(t.0, Box::new(acc), Box::new(t.1))
                })
            })
    }

    fn factor() -> Parser<Expr> {
        let expr = lazy(|| expr());
        spaces()
            .next(or(char1('(').next(expr).prev(char1(')')), num_expr()))
            .prev(spaces())
    }

    fn num_expr() -> Parser<Expr> {
        or(
            apply(int(), |n| Expr::Const(n)),
            apply(char1('-').next(int()), |n: i32| Expr::Const(-n)),
        )
    }
}
