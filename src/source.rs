pub struct Source {
    pub s: Vec<char>,
    pub pos: usize,
}

pub fn source(s: &str) -> Source {
    Source::from(s.to_string().chars().collect())
}

impl Source {
    pub fn from(s: Vec<char>) -> Self {
        Source { s, pos: 0 }
    }

    pub fn peek(&self) -> Option<char> {
        self.s.get(self.pos).cloned()
    }

    pub fn ahead(&mut self) {
        self.pos += 1;
    }

    pub fn pop(&mut self) -> Option<char> {
        self.peek().and_then(|ret| {
            self.ahead();
            Some(ret)
        })
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

impl ToString for Source {
    fn to_string(&self) -> String {
        self.s.as_slice()[self.pos..].iter().collect()
    }
}

