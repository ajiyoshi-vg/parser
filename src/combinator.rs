use super::parser::{parser, ParseError, Parser};

pub fn lazy<T, F>(f: F) -> Parser<T>
where
    T: 'static,
    F: Fn() -> Parser<T> + 'static,
{
    parser(move |s| f().parse(s))
}

pub fn many<T>(p: Parser<T>) -> Parser<Vec<T>>
where
    T: 'static
{
    parser(move |s| {
        let mut ret = Vec::new();
        while let Ok(x) = p.parse(s) {
            ret.push(x);
        }
        Ok(ret)
    })
}

pub fn many1<T>(p: Parser<T>) -> Parser<Vec<T>>
where
    T: 'static
{
    parser(move |s| {
        let mut ret = vec![p.parse(s)?];
        while let Ok(x) = p.parse(s) {
            ret.push(x);
        }
        Ok(ret)
    })
}

pub fn repeat<T>(n: usize, p: Parser<T>) -> Parser<Vec<T>>
where
    T: 'static,
{
    parser(move |s| {
        let mut ret = Vec::with_capacity(n);
        for _ in 0..n {
            ret.push(p.parse(s)?);
        }
        Ok(ret)
    })
    .tryp()
}

pub fn sequence<T>(ps: Vec<Parser<T>>) -> Parser<Vec<T>>
where
    T: 'static,
{
    parser(move |s| {
        let mut tmp = Vec::with_capacity(ps.len());
        for p in ps.iter() {
            tmp.push(p.parse(s)?);
        }
        Ok(tmp)
    })
    .tryp()
}

pub fn or<T>(a: Parser<T>, b: Parser<T>) -> Parser<T>
where
    T: 'static,
{
    let a = a.tryp();
    parser(move |s| a.parse(s).or_else(|_| b.parse(s)))
}

pub fn tryp<T>(p: Parser<T>) -> Parser<T>
where
    T: 'static,
{
    parser(move |s| {
        let save = s.save();
        p.parse(s).or_else(|e| {
            s.restore(save);
            Err(e)
        })
    })
}

pub fn next<A, B>(a: Parser<A>, b: Parser<B>) -> Parser<B>
where
    A: 'static,
    B: 'static,
{
    parser(move |s| a.parse(s).and_then(|_| b.parse(s))).tryp()
}

pub fn prev<A, B>(a: Parser<A>, b: Parser<B>) -> Parser<A>
where
    A: 'static,
    B: 'static,
{
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

pub fn then<T>(head: Parser<T>, tail: Parser<Vec<T>>) -> Parser<Vec<T>>
where
    T: 'static,
{
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

pub fn satisfy<F>(f: F) -> Parser<char>
where
    F: Fn(&char) -> bool + 'static,
{
    parser(move |s| s.pop().filter(|x| f(x)).ok_or(ParseError::Unknown)).tryp()
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
