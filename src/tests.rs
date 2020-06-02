#[cfg(test)]
use super::combinator::*;
use super::source::{source, Source};
use super::parser::{Parser, ParseError};

#[test]
fn test_source() {
    let mut s = source("abc");
    assert_eq!(s.to_string(), "abc".to_string());
    assert_eq!(s.pop(), Some('a'));
    assert_eq!(s.to_string(), "bc".to_string());
    assert_eq!(s.pop(), Some('b'));
    assert_eq!(s.to_string(), "c".to_string());
    assert_eq!(s.pop(), Some('c'));
    assert_eq!(s.to_string(), "".to_string());
    assert_eq!(s.pop(), None);
    assert!(s.is_finished());
    assert_eq!(s.to_string(), "".to_string());
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
    assert_eq!(is_x.parse(&mut s).unwrap(), 'x');
    assert!(is_x.parse(&mut s).is_err());
    assert_eq!(s.to_string(), "yz".to_string());
}

struct Case<T> {
    parser: Parser<T>,
    source: Source,
    expect: Result<T, ParseError>,
    source_after: String,
}

#[test]
fn test_many() {
    let cs = vec![
        Case {
            parser: many(digit()),
            source: source("123x"),
            expect: Ok(vec!['1', '2', '3']),
            source_after: "x".to_string(),
        },
        Case {
            parser: digit().many(),
            source: source("123x"),
            expect: Ok(vec!['1', '2', '3']),
            source_after: "x".to_string(),
        },
        Case {
            parser: many(char1('a').or(char1('b'))),
            source: source("abaabcab"),
            expect: Ok(vec!['a', 'b', 'a', 'a', 'b']),
            source_after: "cab".to_string(),
        },
    ];

    for mut c in cs {
        assert_eq!(c.parser.parse(&mut c.source), c.expect);
        assert_eq!(c.source.to_string(), c.source_after)
    }
}

#[test]
fn test_repeat() {
    let cs = vec![
        Case {
            parser: digit().repeat(4),
            source: source("123x"),
            expect: Err(ParseError::Unknown),
            source_after: "123x".to_string(),
        },
        Case {
            parser: digit().repeat(3),
            source: source("123x"),
            expect: Ok(vec!['1', '2', '3']),
            source_after: "x".to_string(),
        },
        Case {
            parser: repeat(3, digit()),
            source: source("123x"),
            expect: Ok(vec!['1', '2', '3']),
            source_after: "x".to_string(),
        },
        Case {
            parser: digit().repeat(0),
            source: source("abc"),
            expect: Ok(vec![]),
            source_after: "abc".to_string(),
        },
    ];

    for mut c in cs {
        assert_eq!(c.parser.parse(&mut c.source), c.expect);
        assert_eq!(c.source.to_string(), c.source_after)
    }
}

#[test]
fn test_or() {
    let cs = vec![
        Case {
            parser: or(string("ab"), string("cd")),
            source: source("abcd"),
            expect: Ok("ab".to_string()),
            source_after: "cd".to_string(),
        },
        Case {
            parser: or(string("ab"), string("cd")),
            source: source("cdcd"),
            expect: Ok("cd".to_string()),
            source_after: "cd".to_string(),
        },
        Case {
            parser: or(string("ab"), string("ac")),
            source: source("abc"),
            expect: Ok("ab".to_string()),
            source_after: "c".to_string(),
        },
        Case {
            parser: or(string("ab"), string("ac")),
            source: source("acb"),
            expect: Ok("ac".to_string()),
            source_after: "b".to_string(),
        },
        Case {
            parser: string("ab").or(string("ac")),
            source: source("acb"),
            expect: Ok("ac".to_string()),
            source_after: "b".to_string(),
        },
    ];

    for mut c in cs {
        assert_eq!(c.parser.parse(&mut c.source), c.expect);
        assert_eq!(c.source.to_string(), c.source_after)
    }
}

#[test]
fn test_number() {
    let cs = vec![
        Case {
            parser: number(),
            source: source("123c"),
            expect: Ok("123".to_string()),
            source_after: "c".to_string(),
        },
        Case {
            parser: number(),
            source: source("abc"),
            expect: Err(ParseError::Unknown),
            source_after: "abc".to_string(),
        },
    ];
    for mut c in cs {
        assert_eq!(c.parser.parse(&mut c.source), c.expect);
        assert_eq!(c.source.to_string(), c.source_after)
    }
}

#[test]
fn test_int() {
    let cs = vec![
        Case {
            parser: int(),
            source: source("123c"),
            expect: Ok(123),
            source_after: "c".to_string(),
        },
        Case {
            parser: int(),
            source: source("abc"),
            expect: Err(ParseError::Unknown),
            source_after: "abc".to_string(),
        },
    ];
    for mut c in cs {
        assert_eq!(c.parser.parse(&mut c.source), c.expect);
        assert_eq!(c.source.to_string(), c.source_after)
    }
}

#[test]
fn test_next_prev() {
    let cs = vec![
        Case {
            parser: char1('+').next(number()),
            source: source("+123c"),
            expect: Ok("123".to_string()),
            source_after: "c".to_string(),
        },
        Case {
            parser: char1('+').next(number()),
            source: source("+c"),
            expect: Err(ParseError::Unknown),
            source_after: "+c".to_string(),
        },
        Case {
            parser: char1('+').next(number()),
            source: source("123"),
            expect: Err(ParseError::Unknown),
            source_after: "123".to_string(),
        },
        Case {
            parser: number().prev(char1(',')),
            source: source("42,"),
            expect: Ok("42".to_string()),
            source_after: "".to_string(),
        },
        Case {
            parser: number().prev(char1(',')),
            source: source("42"),
            expect: Err(ParseError::Unknown),
            source_after: "42".to_string(),
        },
        Case {
            parser: char1('(').next(number()).prev(char1(')')),
            source: source("(42)"),
            expect: Ok("42".to_string()),
            source_after: "".to_string(),
        },
        Case {
            parser: char1('(').next(number()).prev(char1(')')),
            source: source("(42"),
            expect: Err(ParseError::Unknown),
            source_after: "(42".to_string(),
        },
        Case {
            parser: char1('(').next(number()).prev(char1(')')),
            source: source("()"),
            expect: Err(ParseError::Unknown),
            source_after: "()".to_string(),
        },
    ];
    for mut c in cs {
        assert_eq!(c.parser.parse(&mut c.source), c.expect);
        assert_eq!(c.source.to_string(), c.source_after)
    }
}

#[test]
fn test_expr() {
    struct Case {
        name: String,
        src: Source,
        expect: i32,
    }
    let cases = vec![
        Case {
            name: "simple".to_string(),
            src: source("1+2+3"),
            expect: 6,
        },
        Case {
            name: "with ()".to_string(),
            src: source("1-(2+3)"),
            expect: -4,
        },
        Case {
            name: "mul".to_string(),
            src: source("2*(2+3)"),
            expect: 10,
        },
        Case {
            name: "unary minus".to_string(),
            src: source("-2*(-1+3)"),
            expect: -4,
        },
        Case {
            name: "unary minus".to_string(),
            src: source("-2*(-1+3)"),
            expect: -4,
        },
        Case {
            name: "spaces".to_string(),
            src: source("2 + 3 * 4"),
            expect: 14,
        },
        Case {
            name: "minus".to_string(),
            src: source("2 - 3 * 4"),
            expect: -10,
        },
        Case {
            name: "div".to_string(),
            src: source("100 / 10/2"),
            expect: 5,
        },
        Case {
            name: "div+paren".to_string(),
            src: source("100 / (10 / 2)"),
            expect: 20,
        },
    ];
    for mut c in cases {
        assert_eq!(
            expr()
                .parse(&mut c.src)
                .map(|x| x.eval())
                .expect(c.name.as_str()),
            c.expect
        )
    }

    let cases = vec!["+", "1+", "a+b", "1+(2+3"];
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
    Zero,
    One,
}

impl Expr {
    fn bin(op: Op, lhs: Expr, rhs: Expr) -> Self {
        Expr::BinOp(op, Box::new(lhs), Box::new(rhs))
    }
    fn eval(&self) -> i32 {
        match &self {
            Expr::Zero => 0,
            Expr::One => 1,
            Expr::Const(n) => n.clone(),
            Expr::BinOp(op, a, b) => match op {
                Op::Add => a.eval() + b.eval(),
                Op::Sub => a.eval() - b.eval(),
                Op::Mul => a.eval() * b.eval(),
                Op::Div => a.eval() / b.eval(),
            },
        }
    }
}

#[derive(Debug, PartialEq)]
enum Op {
    Add,
    Sub,
    Mul,
    Div,
}

fn expr() -> Parser<Expr> {
    // term ( [+|-] term )*
    // 2 + 3 * 4
    apply(term(), |x| (Op::Add, x)) // (Op, Expr) を作る
        .then(many(or(
            apply(char1('+').next(term()), |x| (Op::Add, x)), // (Op, Expr) を作る
            apply(char1('-').next(term()), |x| (Op::Sub, x)), // (Op, Expr) を作る
        ))) // Vec<(Op, Expr)> ができる
        .apply(|ts| {
            //構文木を作る
            // "1+2-3" はS式で書くと (- (+ (+ 1 0) 2) 3)  のような構文木になる
            ts.into_iter()
                .fold(Expr::Zero, |acc, t| Expr::bin(t.0, acc, t.1))
        })
}

fn term() -> Parser<Expr> {
    // factor { [*|/] facter }*
    // 1 * 2
    // 2 * ( 3 + 4 )
    apply(factor(), |x| (Op::Mul, x))
        .then(many(or(
            apply(char1('*').next(factor()), |x| (Op::Mul, x)),
            apply(char1('/').next(factor()), |x| (Op::Div, x)),
        )))
        .apply(|ts| {
            //構文木を作る
            // "2*3/4" はS式で書くと (/ (* (* 2 1) 3) 4)  のような構文木になる
            ts.into_iter()
                .fold(Expr::One, |acc, t| Expr::bin(t.0, acc, t.1))
        })
}

fn factor() -> Parser<Expr> {
    let expr = lazy(|| expr());
    // number | '(' expr ')'
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
