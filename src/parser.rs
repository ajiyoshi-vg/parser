use super::source::{Source};
use super::combinator::*;

#[derive(Debug, PartialEq)]
pub enum ParseError {
    Unknown,
}

pub struct Parser<T> {
    body: Box<dyn Fn(&mut Source) -> Result<T, ParseError>>,
}

pub fn parser<T, F>(f: F) -> Parser<T>
where
    T: 'static,
    F: Fn(&mut Source) -> Result<T, ParseError> + 'static,
{
    Parser::new(f)
}

impl<T> Parser<T> {
    pub fn parse(&self, s: &mut Source) -> Result<T, ParseError> {
        (self.body)(s)
    }
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
